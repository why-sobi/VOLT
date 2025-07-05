# pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include "Layer.hpp"
#include "Pair.hpp"

class MultiLayerPerceptron {
private:
	int input_size;
	float learning_rate;
	std::vector<Layer> layers; // Vector of layers in the MLP (only has hidden and output layer, no such thing as input layer)
public:
	MultiLayerPerceptron(int input_size) {
		this->input_size = input_size;
	}
	~MultiLayerPerceptron() {
		std::cout << "MLP object destroyed." << std::endl;
	}

	void addLayer(int num_neurons, std::string actFunction = "") {
		if (layers.empty()) { // since first layer it'll num_of_inputs per neuron will be the input_size
			layers.emplace_back(num_neurons, input_size, actFunction);
		}
		else { // subsequent layers will have num_of_inputs per neuron equal to the number of neurons in the previous layer
			int num_inputs_per_neuron = layers[layers.size() - 1].getNumNeurons(); // get number of neurons in the last layer
			layers.emplace_back(num_neurons, num_inputs_per_neuron, actFunction);
		}
	}

	std::vector<float> forwardPass(std::vector<float>& input) {
		if (input.size() != input_size) {
			std::cerr << "Input size (" << input.size() << ") does not match the MLP input size (" << this->input_size << ")" << std::endl;
			std::exit(EXIT_FAILURE);
		}

		std::vector<float> current_input = input; // Initialize current_input with the input vector (will update it in each layer)
		for (auto& layer : layers) {
			current_input = layer.forward(current_input); // Forward pass through each layer 
		}
		std::cout << "Forward pass completed." << std::endl;
		for (const auto& output : current_input) {
			std::cout << output << " "; // Print the output of the last layer
		}

		return current_input; // Return the output of the last layer
	}

	void train(std::vector<Pair<std::vector<float>, std::vector<float>>> &data, int epochs) { 
		// vector of inputs => corresponding to one output vector (depending on the size of output layer) (data) 
		// and vector of Pairs for batch processing
		
		std::cout << "Training the MLP model..." << std::endl;

		for (int epoch = 0; epoch < epochs; ++epoch) {
			for (auto& sample : data) {
				// sample.first is the input vector
				// sample.second is the expected output vector
				std::vector<float> output = forwardPass(sample.first); // Forward pass to get the output
				float error = 0.0f;

				std::vector<float> propagatingVector(sample.second.size()); // first it stores relevant errors (gradient of E1 wrt activated output) (in this case output - expected output) 


				// Calculate error (for simplicity, using mean squared error)
				if (output.size() != sample.second.size()) {
					std::cerr << "Output size (" << output.size() << ") does not match expected output size (" << sample.second.size() << ")" << std::endl;
					std::exit(EXIT_FAILURE);
				}

				for (size_t i = 0; i < output.size(); ++i) {
					propagatingVector[i] = output[i] - sample.second[i]; // storing relevant errors
					error += std::pow(output[i] - sample.second[i], 2); // i think instead of square multiplying is faster but we'll see
				}

				error /= 2; // Mean squared error

				std::cout << "Epoch: " << epoch + 1 << ", Sample Error: " << error << std::endl;
				
				backPropagation(propagatingVector);
			}
		}
	}
	void predict() {
		// Prediction logic here
		std::cout << "Making predictions with the MLP model..." << std::endl;
	}

	void backPropagation(std::vector<float>& errors) {
		for (int i = layers.size() - 1; i > -1; i--) {
			layers[i].backPropagate_Layer(errors, learning_rate);
		}	
	}
};

/*
		*	at layer layers.size() - 1 there are n neurons
		*	we'll have a loop of n * errors.size() (middle and inner most it'll be handled in the Layer class)
		*
*/