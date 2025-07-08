#pragma once

#include <iostream>
#include <vector>
#include "utils.hpp"

class Neuron {
    std::vector<float> weights;
    float bias;
	float last_output;

public:
    Neuron(int num_inputs) { // function null for now
        weights.resize(num_inputs); // total connection a single neuron will have (from left to right)
		for (int i = 0; i < num_inputs; ++i) { // need to setup random weights (-1, 1)
			weights[i] = getRandomFloat(-1.0f, 1.0f); 
        }
		bias = getRandomFloat(-1.0f, 1.0f); // setup random bias (-1, 1)
		last_output = 0.0f;
    }

	Neuron(float bias, std::vector<float>& weights) {
		this->bias = bias;
		this->weights = weights;

		last_output = 0.0f;
	}

	~Neuron() {
		// Destructor implementation (not needed yet since no dynamic allocation)
	}

    float activate(const std::vector<float>& inputs, std::function<float(float)>& activationFunc) {
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
		last_output = activationFunc(net_out); // apply activation function (or actual output)
		return last_output; 
    }

	std::vector<float> backPropagate_Neuron(float &error, std::vector<float>& last_inputs, float learning_rate, std::function<float(float)>& derActivationFun) {
		std::vector<float> propagatedError(weights.size()); // how much each weight contributed to error

		// derivate of activation function remains same for each neuron of same layer but error (handled in layers class) and last_input changes
		float delta = error * derActivationFun(last_output); // delta tells how much output of the neuron contributed to error

		for (int i = 0; i < weights.size(); i++) {			
			propagatedError[i] = weights[i] * delta;
			// updating weight
			weights[i] -= learning_rate * delta * last_inputs[i];
		}

		bias -= learning_rate * delta;

		return propagatedError;
	}

	const std::vector<float>& getWeights() const {
		return weights;
	}

	const float& getBias() const { return bias;  }
};
