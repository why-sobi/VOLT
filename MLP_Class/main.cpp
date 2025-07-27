#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <chrono>

#include <Eigen/Dense>

#include "MLP.hpp"
#include "store.hpp"

//#include <opencv2/opencv.hpp>

int step_function(float value) { return value < 0.5 ? 0 : 1;  }

int main() {
    // Initialize random seed
    std::srand(time(nullptr));

    /*int a, b;
	std::cout << "Enter two integers: ";
	std::cin >> a >> b;

    Eigen::MatrixXf matrix = Eigen::MatrixXf::Random(a, b);

	std::cout << matrix << std::endl;*/

	std::vector<DataUtil::Sample> data = { 
		{{0, 0}, {0}},
		{{1, 0}, {1}},
		{{0, 1}, {1}},
		{{1, 1}, {0}}
	};

	MultiLayerPerceptron mlp(2, 0.5, Loss::Type::MSE); // Create an MLP with 3 inputs, learning rate of 0.01, and MSE loss function
	mlp.addLayer(5, Activation::ActivationType::Tanh); // Add a hidden layer with 5 neurons and ReLU activation
	mlp.addLayer(1, Activation::ActivationType::Sigmoid); // Add an output layer with 2 neurons and softplus activation

	mlp.train(data, 1000);

	// Fill the matrix with random values


return 0;
}