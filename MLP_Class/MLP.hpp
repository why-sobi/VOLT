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
	std::vector<Layer> layers;																					// Vector of layers in the MLP (only has hidden and output layer, no such thing as input layer)
public:
	MultiLayerPerceptron() {}
	MultiLayerPerceptron(std::string filename) {
		this->load(filename); 
	}
	MultiLayerPerceptron(int input_size, float learning_rate) {
		this->input_size = input_size;
		this->learning_rate = learning_rate;
	}
	~MultiLayerPerceptron() {
		std::cout << "MLP object destroyed." << std::endl;
	}

	void addLayer(int num_neurons, Activation::ActivationType function) {
		if (layers.empty()) {																					// since first layer it'll num_of_inputs per neuron will be the input_size
			layers.emplace_back(num_neurons, input_size, function);
		}
		else {																									// subsequent layers will have num_of_inputs per neuron equal to the number of neurons in the previous layer
			int num_inputs_per_neuron = layers[layers.size() - 1].getNumNeurons();								// get number of neurons in the last layer
			layers.emplace_back(num_neurons, num_inputs_per_neuron, function);
		}
	}

	std::vector<float> forwardPass(std::vector<float>& input) {
		if (input.size() != input_size) {
			std::cerr << "Input size (" << input.size() << ") does not match the MLP input size (" << this->input_size << ")" << std::endl;
			std::exit(EXIT_FAILURE);
		}

		std::vector<float> current_input = input;																// Initialize current_input with the input vector (will update it in each layer)
		for (auto& layer : layers) {
			current_input = layer.forward(current_input);														// Forward pass through each layer 
		}

		return current_input;																					// Return the output of the last layer
	}

	void train(std::vector<Pair<std::vector<float>, std::vector<float>>> &data, int epochs) { 
		// vector of inputs => corresponding to one output vector (depending on the size of output layer) (data) 
		// and vector of Pairs for batch processing
		
		for (int epoch = 0; epoch < epochs; ++epoch) {
			float epoch_error = 0.0f;
			for (auto& sample : data) {
				// sample.first is the input vector
				// sample.second is the expected output vector
				std::vector<float> output = forwardPass(sample.first);											// Forward pass to get the output
				float error = 0.0f;

				std::vector<float> propagatingVector(sample.second.size());										// first it stores relevant errors (gradient of E1 wrt activated output) (in this case output - expected output) 


				// Calculate error (for simplicity, using mean squared error)
				if (output.size() != sample.second.size()) {
					std::cerr << "Output size (" << output.size() << ") does not match expected output size (" << sample.second.size() << ")" << std::endl;
					std::exit(EXIT_FAILURE);
				}

				for (size_t i = 0; i < output.size(); ++i) {
					propagatingVector[i] = output[i] - sample.second[i];										// storing relevant errors
					error += (output[i] - sample.second[i]) * (output[i] - sample.second[i]);
				}

				error /= 2;																						// Mean squared error
				epoch_error += error;
				
				backPropagation(propagatingVector);
			}
			
			std::cout << "Epoch " << epoch + 1 << ", Avg Error: " << epoch_error / data.size() << " | Avg Accuracy: " << 1 - epoch_error / data.size() << "\n----------------------------------------------------------\n";
		}
	}
	std::vector<float> predict(std::vector<float>& input) {
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
		

		std::ofstream out(filename + ".bin", std::ios::binary);
		if (!out) throw std::runtime_error("Failed to open file for saving");

		size_t num_layers = layers.size();

		out.write(reinterpret_cast<const char*>(&this->input_size), sizeof(this->input_size));					// writing the input size
		out.write(reinterpret_cast<const char*>(&this->learning_rate), sizeof(this->learning_rate));			// writing the learning_rate
		out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));								// writing the number of layers

		for (const auto& layer : layers) {
			std::string activationFunc = layer.getActivationFunc();
			size_t length = activationFunc.length();
			size_t totalNeurons = layer.getNumNeurons();
			std::vector<Neuron> neurons = layer.getNeurons();


			out.write(reinterpret_cast<const char*>(&length), sizeof(length));									// storing length of actFunc string
			out.write(reinterpret_cast<const char*>(activationFunc.c_str()), length);							// writing activation func as string
			out.write(reinterpret_cast<const char*>(&totalNeurons), sizeof(totalNeurons));						// writing total number of neurons

			for (const auto& neuron : neurons) {
				std::vector<float> weights = neuron.getWeights();
				size_t total_weights = weights.size();
				float bias = neuron.getBias();

				out.write(reinterpret_cast<const char*>(&bias), sizeof(bias));									// writing bias
				out.write(reinterpret_cast<const char*>(&total_weights), sizeof(total_weights));				// writing size of weights vector
				out.write(reinterpret_cast<const char*>(weights.data()), total_weights * sizeof(float));		// writing activation func as string
			}
		}

		out.close();
	}

	void load(std::string filename) {
		std::cout << "Loading...\n";

		std::ifstream in(filename + ".bin", std::ios::binary);
		if (!in) throw std::runtime_error("Cannot open file or it does not exist!");

		size_t num_layers;
		
		// reading the MLP meta data (input size and learning rate)
		in.read(reinterpret_cast<char*>(&this->input_size), sizeof(this->input_size));							// reading input size
		in.read(reinterpret_cast<char*>(&this->learning_rate), sizeof(this->learning_rate));					// reading learning rate
		in.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));										// reading number of layers (input layer is not a layer)
		layers.clear();
		layers.reserve(num_layers);																				// allocating memory here so no extra reallocation

		for (int i = 0; i < num_layers; i++) {

			size_t len;
			size_t total_neurons;
			std::string activation_func;
			std::vector<char> buffer;
			
			// reading the activation function name
			in.read(reinterpret_cast<char*>(&len), sizeof(len));												// reading activation function string length
			buffer.resize(len);																					// resizing buffer to adequate length to store function name
			in.read(buffer.data(), len);																		// reading function name and storing in buffer
			activation_func.assign(buffer.data(), len);															// storing function name from buffer to a activation_func to the string


			// reading the number of neurons per layer
			in.read(reinterpret_cast<char*>(&total_neurons), sizeof(total_neurons));							// reading neurons per layer
			std::vector<Neuron> neurons;
			neurons.reserve(total_neurons);																		// to avoid unnecessary reallocation


			for (int j = 0; j < total_neurons; j++) {

				float bias;
				size_t total_weights;
				std::vector<float> weights;

				// readings bias, total weights and weights values
				in.read(reinterpret_cast<char*>(&bias), sizeof(bias));											// reading bias
				in.read(reinterpret_cast<char*>(&total_weights), sizeof(total_weights));						// reading number of weights
				weights.resize(total_weights);																	// pre-req to reading in a vector

				in.read(reinterpret_cast<char*>(weights.data()), total_weights * sizeof(float));				// reading actual weights

				// putting in the neurons vector
				neurons.emplace_back(bias, weights);															// re-creating the neuron and storing it

			}

			// putting in the layers vector
			layers.emplace_back(neurons, activation_func);														// re-creating the layer and storing it
		}
	}
};