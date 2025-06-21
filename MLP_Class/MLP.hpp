#include <iostream>
#include <Eigen/Dense>
#include "Layer.hpp"
#include "Pair.hpp"

class MultiLayerPerceptron {
private:
	int input_size;
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

	void train(std::vector<Pair<std::vector<float>, std::vector<float>>> &data, int epochs) { 
		// vector of inputs => corresponding to one output vector (depending on the size of output layer) (data) 
		// and vector of Pairs for batch processing
		
		std::cout << "Training the MLP model..." << std::endl;
	}
	void predict() {
		// Prediction logic here
		std::cout << "Making predictions with the MLP model..." << std::endl;
	}
};