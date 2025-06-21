#pragma once

#include <iostream>
#include <vector>
#include "utils.hpp"

class Neuron {
    std::vector<float> weights;
    float bias;
	std::string activationFunc;

public:
    Neuron(int num_inputs, std::string actFunc = "") { // function null for now
        weights.resize(num_inputs); // total connection a single neuron will have (from left to right)
		for (int i = 0; i < num_inputs; ++i) { // need to setup random weights (-1, 1)
			weights[i] = getRandomFloat(-1.0f, 1.0f); 
        }
		bias = getRandomFloat(-1.0f, 1.0f); // setup random bias (-1, 1)
    }
	~Neuron() {
		// Destructor implementation (not needed yet since no dynamic allocation)
	}

    float activate(const std::vector<float>& inputs) {
		// net output = inputs * weights + bias (dot product)
		if (inputs.size() != weights.size()) {
			std::cerr << "Error: Number of inputs does not match number of weights." << std::endl;
			return 0.0f; // or throw an exception
		}

		float net_out = 0.0f;
		for (size_t i = 0; i < inputs.size(); ++i) {
			net_out += inputs[i] * this->weights[i];
		}

		net_out += this->bias; // add bias
		return sigmoid(net_out); // apply activation function (or actual output)
    }
	
	
	float sigmoid(float x) {
		// Sigmoid activation function
		return 1.0f / (1.0f + std::exp(-x));
	}

	float derivativeSigmoid(float x) {
		// Derivative of the sigmoid function
		return x * (1.0f - x); // assuming x is the output of the sigmoid function
	}
};
