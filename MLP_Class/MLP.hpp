# pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <fstream>
//#include <Eigen/Dense>
#include "Layer.hpp"
#include "Pair.hpp"

class MultiLayerPerceptron {
private:
	int input_size;
	float learning_rate;
	std::vector<Layer> layers; // Vector of layers in the MLP (only has hidden and output layer, no such thing as input layer)
public:
	MultiLayerPerceptron(int input_size, float learning_rate) {
		this->input_size = input_size;
		this->learning_rate = learning_rate;
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
		//std::cout << "Forward pass completed." << std::endl;
		//for (const auto& output : current_input) {
		//	std::cout << output << " "; // Print the output of the last layer
		//}

		return current_input; // Return the output of the last layer
	}

	void train(std::vector<Pair<std::vector<float>, std::vector<float>>> &data, int epochs) { 
		// vector of inputs => corresponding to one output vector (depending on the size of output layer) (data) 
		// and vector of Pairs for batch processing
		
		for (int epoch = 0; epoch < epochs; ++epoch) {
			float epoch_error = 0.0f;
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
					//error += std::pow(output[i] - sample.second[i], 2); // i think instead of square multiplying is faster but we'll see
					error += (output[i] - sample.second[i]) * (output[i] - sample.second[i]);
				}

				error /= 2; // Mean squared error
				epoch_error += error;
				
				backPropagation(propagatingVector);
			}
			
			std::cout << "Epoch " << epoch + 1 << ", Avg Error: " << epoch_error / data.size() << " | Avg Accuracy: " << 1 - epoch_error / data.size() << "\n----------------------------------------------------------\n";
		}
	}
	std::vector<float> predict(std::vector<float>& input) {
		// Prediction logic here
		std::cout << "Making predictions with the MLP model..." << std::endl;
		return forwardPass(input);
	}

	void backPropagation(std::vector<float>& errors) {
		for (int i = layers.size() - 1; i > -1; i--) {
			layers[i].backPropagate_Layer(errors, learning_rate);
		}	
	}

	void save(const std::string& filename) const {
		std::cout << "Saving...\n";
		

		std::ofstream out(filename + "bin", std::ios::binary);
		if (!out) throw std::runtime_error("Failed to open file for saving");

		size_t num_layers = layers.size();
		out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

		for (const auto& layer : layers) {
			size_t num_neurons = layer.getNumNeurons();
			out.write(reinterpret_cast<const char*>(&num_neurons), sizeof(num_neurons));

			for (auto& neuron : layer.getNeurons()) {
				size_t num_weights = neuron.getWeights().size();
				out.write(reinterpret_cast<const char*>(&num_weights), sizeof(num_weights));
				out.write(reinterpret_cast<const char*>(neuron.getWeights().data()), num_weights * sizeof(float));
				out.write(reinterpret_cast<const char*>(&neuron.getBias()), sizeof(float));
			}
		}

		out.close();
	}
};

/*
		*	at layer layers.size() - 1 there are n neurons
		*	we'll have a loop of n * errors.size() (middle and inner most it'll be handled in the Layer class)
		*
*/