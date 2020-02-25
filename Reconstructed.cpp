#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

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

// weight for first dense layer
Eigen::MatrixXd get_weight1()
{
	Eigen::MatrixXd f;
	f.resize(1840, 64);
	std::ifstream file("../weights/weight_2.txt");

	for (unsigned int i = 0; i < 1840; i++) {
		for (unsigned int j = 0; j < 64; j++) {
			file >> f(i, j);
		}
	}
	return f;
}

// weight for second dense layer
Eigen::MatrixXd get_weight2()
{
	Eigen::MatrixXd f;
	f.resize(64, 32);
	std::ifstream file("../weights/weight_4.txt");

	for (unsigned int i = 0; i < 64; i++) {
		for (unsigned int j = 0; j < 32; j++) {
			file >> f(i, j);
		}
	}
	return f;
}

// weight for third dense layer
Eigen::MatrixXd get_weight3()
{
	Eigen::MatrixXd f;
	f.resize(32, 5);
	std::ifstream file("../weights/weight_6.txt");

	for (unsigned int i = 0; i < 32; i++) {
		for (unsigned int j = 0; j < 5; j++) {
			file >> f(i, j);
		}
	}
	return f;
}

Eigen::VectorXd get_bias1()
{
	Eigen::VectorXd f;
	f.resize(64);
	std::ifstream file("../weights/weight_3.txt");

	for (unsigned int i = 0; i < 64; i++)  file >> f(i);

	return f;
}

Eigen::VectorXd get_bias2()
{
	Eigen::VectorXd f;
	f.resize(32);
	std::ifstream file("../weights/weight_5.txt");

	for (unsigned int i = 0; i < 32; i++)  file >> f(i);

	return f;
}

Eigen::VectorXd get_bias3()
{
	Eigen::VectorXd f;
	f.resize(5);
	std::ifstream file("../weights/weight_7.txt");

	for (unsigned int i = 0; i < 5; i++)  file >> f(i);

	return f;
}

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
Eigen::MatrixXd relu(Layer x)
{
	int rows = x.rows();
	int cols = x.cols();

	for (unsigned int i = 0; i < rows; i++)
	{
		for (unsigned int j = 0; j < cols; j++)
		{
			if (x(i,j) < 0) x(i,j) = 0;
		}
	}
	return x;
}

double max(Layer x)
{
	int I = x.rows(); int J = x.cols();
	double m = -1.0;
	for (unsigned int i = 0; i < I; i++)
	{
		for (unsigned int j = 0; j < J; j++)
		{
			if (x(i, j) > m) m = x(i, j);
		}
	}
	return m;
}

// BROKEN; NEEDS TO BE FIXED
Eigen::MatrixXd softmax(Layer x)
{
	double m = max(x);
	for (int f = 0; f < x.rows(); f++)
		x(f) = exp(x(f) - m);
	x /= x.sum();
	return x;
}

// is expected to return a tensor of shape (10, batch_size, 185)
Tensor conv_1d(Layer input, int filters, int kernel_size)
{
	int batch = input.rows();
	int C = input.cols();
	Tensor conv_1d_output;
	Filter F = get_filter();
	
	for (unsigned int i = 0; i < filters; i++)
	{
		std::vector<std::vector<double>> Matrix;
		for (unsigned int j = 0; j < batch; j++)
		{		
			std::vector<double> row;
			for (unsigned int c = 0; c < C - kernel_size + 1; c++)
			{
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
	for (unsigned int c = 0; c < channels; c++)
	{
		std::vector<std::vector<double>> matrix;
		for (unsigned int b = 0; b < batch; b++)
		{
			std::vector<double> row;
			for (unsigned int i = 0; i < columns - pool_size + 1; i++)
			{
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
Eigen::MatrixXd flatten(Tensor x)
{
	int channels = x.size(); // always 10
	int batch = x[0].size(); // varies
	int columns = x[0][0].size(); // always 184

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix;
	matrix.resize(batch, channels*columns);

	std::vector<std::vector<double>> M;
	for (unsigned int b = 0; b < batch; b++)
	{
		std::vector<double> row;
		for (unsigned int c = 0; c < channels; c++)
		{
			for (unsigned int i = 0; i < columns; i++)
			{
				row.push_back(x[c][b][i]);
			}			
		}
		M.push_back(row);
	}
	// convert two-dimensional vector into Eigen matrix
	for (unsigned int i = 0; i < M.size(); i++)
	{
		for (unsigned int j = 0; j < M[0].size(); j++)
		{
			matrix(i, j) = M[i][j];
		}
	}
	std::cout << "> Flattening complete" << std::endl;
	return matrix;
}

Eigen::MatrixXd dense1(Layer x)
{
	x = x * get_weight1();
	int I = x.rows(); int J = x.cols();
	Eigen::VectorXd b = get_bias1().transpose();
	for (int i = 0; i < I; i++)
	{
		for (int j = 0; j < J; j++)
		{
			x(i, j) += b(j);
		}
	}
	std::cout << "> Dense 1 complete" << std::endl;
	return relu(x);
}

Eigen::MatrixXd dense2(Layer x)
{
	x = x * get_weight2();
	int I = x.rows(); int J = x.cols();
	Eigen::VectorXd b = get_bias2().transpose();
	for (int i = 0; i < I; i++)
	{
		for (int j = 0; j < J; j++)
		{
			x(i, j) += b(j);
		}
	}
	std::cout << "> Dense 2 complete" << std::endl;
	return relu(x);
}

Eigen::MatrixXd dense3(Layer x)
{
	x = x * get_weight3();
	int I = x.rows(); int J = x.cols();
	Eigen::VectorXd b = get_bias3().transpose();
	for (int i = 0; i < I; i++)
	{
		for (int j = 0; j < J; j++)
		{
			x(i, j) += b(j);
		}
	}
	std::cout << "> Dense 3 complete" << std::endl;
	return (x);
}


int main()
{
	Eigen::MatrixXd input = Eigen::MatrixXd::Zero(100, 187); // random input for testing

	Tensor x; Eigen::MatrixXd m;

	x = conv_1d(input, 10, 3);
	x = max_pooling1d(x, 2, 1);
	m = flatten(x);
	m = dense1(m);
	m = dense2(m);
	m = dense3(m);
	std::cout << m << std::endl;
}
