#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

#include <chrono> 
using namespace std::chrono;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Layer;
typedef std::vector<std::vector<std::vector<double>>> Tensor;

// returns a filter matrix of shape (10, 3)
Eigen::MatrixXd get_filter1()
{
	Eigen::MatrixXd f;
	f.resize(10, 3);
	std::ifstream file("../weights/filter1.txt");

	for (unsigned int i = 0; i < 10; i++) {
		for (unsigned int j = 0; j < 3; j++) {
			file >> f(i, j);
		}
	}
	return f;
}

// returns tensor of shape (3, 10, 5)
Tensor get_filter2()
{
	Tensor f;
	std::ifstream file1("../weights/filter2_1.txt");
	std::ifstream file2("../weights/filter2_2.txt");
	std::ifstream file3("../weights/filter2_3.txt");

	std::vector<std::vector<double>> temp;
	for (unsigned int i = 0; i < 10; i++) {
		std::vector<double> t; double x;
		t.reserve(5);
		for (unsigned int j = 0; j < 5; j++) {
			file1 >> x;
			t.push_back(x);
		}
		temp.push_back(t);
	}
	f.push_back(temp);

	temp = std::vector<std::vector<double>>();
	for (unsigned int i = 0; i < 10; i++) {
		std::vector<double> t;
		t.reserve(5); double x;
		for (unsigned int j = 0; j < 5; j++) {
			file2 >> x;
			t.push_back(x);
		}
		temp.push_back(t);
	}
	f.push_back(temp);

	temp = std::vector<std::vector<double>>();
	for (unsigned int i = 0; i < 10; i++) {
		std::vector<double> t; double x;
		t.reserve(5);
		for (unsigned int j = 0; j < 5; j++) {
			file3 >> x;
			t.push_back(x);
		}
		temp.push_back(t);
	}
	f.push_back(temp);

	return f;
}

Layer get_weight(int n, int row, int col)
{
	Eigen::MatrixXd f;
	f.resize(row, col);

	std::string path = "../weights/weight" + std::to_string(n) + ".txt";
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

	std::string path = "../weights/bias" + std::to_string(n) + ".txt";
	std::ifstream file(path);

	for (unsigned int i = 0; i < col; i++)
		file >> f(i);

	return f;
}

Layer W1 = get_weight(1, 935, 64);
Layer W2 = get_weight(2, 64, 32);
Layer W3 = get_weight(3, 32, 5);

Eigen::VectorXd b1 = get_bias(1, 64).transpose();
Eigen::VectorXd b2 = get_bias(2, 32).transpose();
Eigen::VectorXd b3 = get_bias(3, 5).transpose();

// ReLU activation for conv_1d
void relu_3d(Tensor& x)
{
	int channels = x.size();
	int rows = x[0].size();
	int cols = x[0][0].size();

	Tensor y;
	for (unsigned int c = 0; c < channels; c++)
		for (unsigned int i = 0; i < rows; i++)
			for (unsigned int j = 0; j < cols; j++)
				if (x[c][i][j] < 0) x[c][i][j] = 0;
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

Eigen::VectorXd exp(Eigen::VectorXd V)
{
	for (unsigned int i = 0; i < 5; i++)
		V[i] = std::exp(V[i]);
	return V;
}

// softmax activation for final layer
void softmax(Layer& x)
{
	Eigen::VectorXd m = max(x);

	for (int f = 0; f < x.rows(); f++)
		x.row(f) = exp(x.row(f) - m.transpose());
	x /= x.sum();
}

// is expected to return a tensor of shape (10, batch_size, 187) (same padding)
Tensor conv_1d(Layer input, int filters, int kernel_size, Eigen::MatrixXd F)
{
	int batch = input.rows();
	int col = input.cols();
	Tensor conv_1d_output;
	
	for (unsigned int i = 0; i < filters; i++) {
		std::vector<std::vector<double>> Matrix;
		for (unsigned int j = 0; j < batch; j++) {		
			std::vector<double> row;
			for (unsigned int c = 0; c < col - kernel_size + 1; c++) {
				double conv;
				if (c == 0) {
					conv = F(i, 1) * input(j, c + 1) + F(i, 2) * input(j, c + 2);
					row.push_back(conv);
				}
				conv = F(i, 0) * input(j, c) + F(i, 1) * input(j, c+1) + F(i, 2) * input(j, c+2);
				row.push_back(conv);
				if (c == col - kernel_size) {
					conv = F(i, 0) * input(j, c) + F(i, 1) * input(j, c + 1);
					row.push_back(conv);
				}
			}
			Matrix.push_back(row);
		}
		conv_1d_output.push_back(Matrix); 
	}
	std::cout << "> Convolution 1 complete" << std::endl;
	relu_3d(conv_1d_output);

	return conv_1d_output;
}

// returns tensor of shape (5, batch_size, 187)
Tensor conv_1d(Tensor input, int filters, int kernel_size, Tensor Filter)
{
	int channels = input.size();
	int batch = input[0].size();
	int columns = input[0][0].size();
	Tensor conv_1d_output;

	for (unsigned int f = 0; f < filters; f++) {
		std::vector<std::vector<double>> Matrix;
		for (unsigned int i = 0; i < batch; i++) {
			std::vector<double> row;
			for (unsigned int j = 0; j < columns - kernel_size + 1; j++) {
				double conv = 0, pad1 = 0, pad2 = 0;
				for (unsigned int c = 0; c < channels; c++) {
					if (j == 0)
						pad1 += input[c][i][j + 1] * Filter[1][c][f] + input[c][i][j + 2] * Filter[2][c][f];
					if (j == columns - kernel_size)
						pad2 += input[c][i][j] * Filter[0][c][f] + input[c][i][j + 1] * Filter[1][c][f];
					conv += input[c][i][j] * Filter[0][c][f] + input[c][i][j + 1] * Filter[1][c][f] + input[c][i][j + 2] * Filter[2][c][f];
				}
				if (j == 0)
					row.push_back(pad1);
				row.push_back(conv);
				if (j == columns - kernel_size)
					row.push_back(pad2);
			}
			Matrix.push_back(row);
		}
		conv_1d_output.push_back(Matrix);
	}
	std::cout << "> Convolution 2 complete" << std::endl;
	relu_3d(conv_1d_output);

	return conv_1d_output;
}

// is expected to return tensor of shape (10, batch_size, 187) (same padding)
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
				if (i == columns - 1)
					row.push_back(x[c][b][i]);
			}
			matrix.push_back(row);
		}
		pooling_output.push_back(matrix);
	}
	std::cout << "> Pooling complete" << std::endl;
	return pooling_output;
}

// turns 3-dimensional tensor into 2-dimensional Eigen matrix
inline Layer flatten(Tensor x)
{
	int channels = x.size(); // always 10
	int batch = x[0].size(); // varies
	int columns = x[0][0].size(); // always 184

	Layer matrix;
	matrix.resize(batch, channels*columns);

	for (unsigned int b = 0; b < batch; b++) {
		int k = 0;
		for (unsigned int i = 0; i < columns; i++) {
			for (unsigned int c = 0; c < channels; c++) {
				matrix(b, k) = x[c][b][i];
				++k;
			}
		}
	}
	std::cout << "> Flattening complete" << std::endl;
	return matrix;
}

void dense1(Layer& x)
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

// test data and evaluation

// returns input matrix of shape (21891, 187)
Eigen::MatrixXd get_X()
{
	Eigen::MatrixXd X;
	X.resize(21891, 187);

	std::ifstream file("../weights/X.txt");

	for (unsigned int i = 0; i < 21891; i++) {
		for (unsigned int j = 0; j < 187; j++) {
			file >> X(i, j);
		}
	}
	return X;
}

// returns target vector of size 21891
Eigen::VectorXd get_y()
{
	Eigen::VectorXd y;
	y.resize(21891);

	std::string path = "../weights/y.txt";
	std::ifstream file(path);

	for (unsigned int i = 0; i < 21891; i++)
		file >> y(i);

	return y.transpose();
}

double accuracy_score(Layer y_pred, Eigen::VectorXd y_true)
{
	double score = 0.0;
	int batch = y_pred.rows();
	int cols = y_pred.cols();
	for (unsigned int i = 0; i < batch; i++) {
		int predicted;
		double m = y_pred.row(i).maxCoeff();
		for (unsigned int j = 0; j < cols; j++) {
			if (y_pred(i, j) == m)
				predicted = j;
		}
		if (predicted == y_true(i))
			score += 1;
	}
	return score / batch;
}


int main()
{
	//Eigen::MatrixXd input = Eigen::MatrixXd::Random(1000, 187); // input for testing
	Eigen::MatrixXd input = get_X(); Eigen::VectorXd y = get_y();
	Eigen::MatrixXd filter1 = get_filter1();
	Tensor filter2 = get_filter2();

	Tensor x; Layer m;
	
	auto start = high_resolution_clock::now();

	x = conv_1d(input, 10, 3, filter1);
	x = conv_1d(x, 5, 3, filter2);
	x = max_pooling1d(x, 2, 1);
	m = flatten(x);
	dense1(m);
	dense2(m);
	dense3(m);

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<seconds>(stop - start);
	std::cout << "\n$> Batch size: " << input.rows() << ", Elapsed time: " << duration.count() << " seconds" << std::endl;
	std::cout << "Accuracy: " << accuracy_score(m, y) << std::endl;
}
