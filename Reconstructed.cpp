#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

#include <chrono> 
using namespace std::chrono;

typedef Eigen::Matrix<double, 10, 3> Filter; // filter used for convolution
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Layer;
typedef std::vector<std::vector<std::vector<double>>> Tensor;

// returns a filter matrix of shape (10, 3)
Filter get_filter()
{
	Filter f;
	std::ifstream file("../weights/weight_1.txt");

	for (unsigned int i = 0; i < 10; i++) {
		for (unsigned int j = 0; j < 3; j++) {
			file >> f(i, j);
		}
	}
	return f;
}

Layer get_weight(int n, int row, int col)
{
	Eigen::MatrixXd f;
	f.resize(row, col);

	std::string path = "../weights/weight_" + std::to_string(n) + ".txt";
	std::ifstream file(path);

	for (unsigned int i = 0; i < row; i++) 
		for (unsigned int j = 0; j < col; j++)
			file >> f(i, j);
	return f;
}

Eigen::VectorXd get_bias(int n, int col)
{
	Eigen::VectorXd f;
	f.resize(col);

	std::string path = "../weights/weight_" + std::to_string(n) + ".txt";
	std::ifstream file(path);

	for (unsigned int i = 0; i < col; i++)
		file >> f(i);

	return f;
}

Filter F = get_filter();

Layer W1 = get_weight(2, 1840, 64);
Layer W2 = get_weight(4, 64, 32);
Layer W3 = get_weight(6, 32, 5);

Eigen::VectorXd b1 = get_bias(3, 64).transpose();
Eigen::VectorXd b2 = get_bias(5, 32).transpose();
Eigen::VectorXd b3 = get_bias(7, 5).transpose();

// ReLU activation for conv_1d
void relu_3d(Tensor& x)
{
	int channels = x.size();
	int rows = x[0].size();
	int cols = x[0][0].size();

	Tensor y;
	for (unsigned int c = 0; c < channels; c++)
	{
		for (unsigned int i = 0; i < rows; i++)
		{
			for (unsigned int j = 0; j < cols; j++)
			{
				if (x[c][i][j] < 0) x[c][i][j] = 0;
			}
		}
	}
}

// ReLU activation function
inline void relu(Layer& x)
{
	int rows = x.rows();
	int cols = x.cols();

	for (unsigned int i = 0; i < rows; i++)
		for (unsigned int j = 0; j < cols; j++)
			if (x(i,j) < 0) x(i,j) = 0;
}

Eigen::VectorXd max(Layer x)
{
	int I = x.rows(); int J = x.cols();
	double m = -1.0;
	Eigen::VectorXd V;
	V.resize(5);
	for (unsigned int i = 0; i < I; i++)
		for (unsigned int j = 0; j < J; j++)
			if (x(i, j) > m) m = x(i, j);
	V << m, m, m, m, m;
	
	return V;
}

Eigen::VectorXd Exp(Eigen::VectorXd V)
{
	//int n = V.size();
	for (unsigned int i = 0; i < 5; i++)
		V[i] = std::exp(V[i]);
	return V;
}

// softmax activation for final layer
void softmax(Layer& x)
{
	Eigen::VectorXd m = max(x);

	for (int f = 0; f < x.rows(); f++)
		x.row(f) = Exp(x.row(f) - m.transpose());
	x /= x.sum();
}

// is expected to return a tensor of shape (10, batch_size, 185)
Tensor conv_1d(Layer input, int filters, int kernel_size)
{
	int batch = input.rows();
	int col = input.cols();
	Tensor conv_1d_output;
	
	for (unsigned int i = 0; i < filters; i++) {
		std::vector<std::vector<double>> Matrix;
		for (unsigned int j = 0; j < batch; j++) {		
			std::vector<double> row;
			for (unsigned int c = 0; c < col - kernel_size + 1; c++) {
				double conv = F(i, 0) * input(j, c) + F(i, 1) * input(j, c+1) + F(i, 2) * input(j, c+2);
				row.push_back(conv);
			}
			Matrix.push_back(row);
		}
		conv_1d_output.push_back(Matrix); 
	}
	std::cout << "> Convolution complete" << std::endl;
	relu_3d(conv_1d_output);

	return conv_1d_output;
}

// is expected to return tensor of shape (10, batch_size, 184)
Tensor max_pooling1d(Tensor x, int pool_size, int strides)
{	
	int channels = x.size();
	int batch = x[0].size();
	int columns = x[0][0].size();
	
	Tensor pooling_output;
	for (unsigned int c = 0; c < channels; c++) {
		std::vector<std::vector<double>> matrix;
		for (unsigned int b = 0; b < batch; b++) {
			std::vector<double> row;
			for (unsigned int i = 0; i < columns - pool_size + 1; i++) {
				double val = (x[c][b][i] > x[c][b][i + 1]) ? x[c][b][i] : x[c][b][i + 1];
				row.push_back(val);
			}
			matrix.push_back(row);
		}
		pooling_output.push_back(matrix);
	}
	std::cout << "> Pooling complete" << std::endl;
	return pooling_output;
}

// turns 3-dimensional tensor into 2-dimensional Eigen matrix
Layer flatten(Tensor x)
{
	int channels = x.size(); // always 10
	int batch = x[0].size(); // varies
	int columns = x[0][0].size(); // always 184

	Layer matrix;
	matrix.resize(batch, channels*columns);

	for (unsigned int b = 0; b < batch; b++) {
		int k = 0;
		for (unsigned int i = 0; i < columns; i++)
			for (unsigned int c = 0; c < channels; c++) {
				matrix(b, k) = x[c][b][i];
				++k;
			}			
	}
	std::cout << "> Flattening complete" << std::endl;
	return matrix;
}

inline void dense1(Layer& x)
{
	x *= W1;
	int I = x.rows();

	for (int i = 0; i < I; i++)
		x.row(i) += b1;
	std::cout << "> Dense 1 complete" << std::endl;
	relu(x);
}

void dense2(Layer& x)
{
	x *= W2;
	int I = x.rows();
	for (int i = 0; i < I; i++)
		x.row(i) += b2;
	std::cout << "> Dense 2 complete" << std::endl;
	relu(x);
}

void dense3(Layer& x)
{
	x *= W3;
	int I = x.rows();

	for (int i = 0; i < I; i++)
		x.row(i) += b3;
	std::cout << "> Dense 3 complete" << std::endl;
	softmax(x);
}


int main()
{
	Eigen::MatrixXd input = Eigen::MatrixXd::Ones(100, 187); // random input for testing

	Tensor x; Layer m;
	
	auto start = high_resolution_clock::now();
	x = conv_1d(input, 10, 3);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<seconds>(stop - start);
	std::cout << "Elapsed time: " << duration.count() << std::endl;

	x = max_pooling1d(x, 2, 1);
	m = flatten(x);
	dense1(m);
	dense2(m);
	dense3(m);

	stop = high_resolution_clock::now();
	duration = duration_cast<seconds>(stop - start);
	std::cout << "Elapsed time: " << duration.count() << std::endl;
	std::cout << m.row(0) << std::endl;
}
