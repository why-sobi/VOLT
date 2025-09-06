#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <chrono>

#include <Eigen/Dense>

#include "./Model/MLP.hpp"
//#include "./Utility/store.hpp"

int step_function(float value) { return value < 0.5 ? 0 : 1;  }


const int POINTS_PER_CLASS = 100;
const int NUM_CLASSES = 3;

std::vector<DataUtil::Sample> generateSpiralDataset() {
    std::vector<DataUtil::Sample> dataset;

    std::srand((unsigned int)std::time(nullptr));

    for (int classIdx = 0; classIdx < NUM_CLASSES; ++classIdx) {
        for (int i = 0; i < POINTS_PER_CLASS; ++i) {
            float r = static_cast<float>(i) / POINTS_PER_CLASS;
            float theta = classIdx * 4 + 4 * r + ((float)std::rand() / RAND_MAX) * 0.2f;

            float x = r * std::sin(theta);
            float y = r * std::cos(theta);

            std::vector<float> input = { x, y };
            std::vector<float> label(NUM_CLASSES, 0.0f);
            label[classIdx] = 1.0f;

            dataset.emplace_back(
                Eigen::Map<Eigen::VectorX<float>>(input.data(), input.size()),
                Eigen::Map<Eigen::VectorX<float>>(label.data(), label.size())
            );
        }
    }

    return dataset;
}

int main() {
	auto start = std::chrono::high_resolution_clock::now();
	// Initialize random seed
	std::srand(time(nullptr));

	std::vector<DataUtil::Sample> data = generateSpiralDataset();

	MultiLayerPerceptron mlp(2, Loss::Type::CategoricalCrossEntropy, new RMSprop(0.005)); // Create an MLP with 3 inputs, learning rate of 0.01, and MSE loss function
	
    mlp.addLayer(32, Activation::ActivationType::ReLU); // Add a hidden layer with 5 neurons and ReLU activation
	mlp.addLayer(32, Activation::ActivationType::ReLU); // Add an output layer with 2 neurons and softplus activation
    
    mlp.addLayer(3, Activation::ActivationType::Softmax); // Add an output layer with 2 neurons and softplus activation

	mlp.train(data, 1000, 32);

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Training completed in " << elapsed.count() << " seconds." << std::endl;
	// Fill the matrix with random values


    return 0;
}