#pragma once

#include <iostream>
#include <vector>
#include "Neuron.hpp"

class Layer {
private:
    std::vector<Neuron> neurons;
    std::vector<float> last_input; // makes it easier to access the activated output of last layer (is used for grad(net/w))
    std::function<float(float)> activationFunction;
    std::function<float(float)> derActivationFunction;

public:
    Layer(int num_neurons, int num_inputs_per_neuron, std::string actFunc = "") { // function null for now
        neurons.reserve(num_neurons);    // no unecessary reallocation 
        
        for (int i = 0; i < num_neurons; ++i) {
            neurons.emplace_back(num_inputs_per_neuron);
        }
        // neurons.emplace_back(num_inputs_per_neuron); same as neurons.push_back(Neuron(num_inputs_per_neuron));
        // emplace_back takes the int, checks for neuron constructor and calls it using the int (this results in no unnecessary allocation or assignment)
        this->activationFunction = setActivationFunction(actFunc);
        this->derActivationFunction = setDerActivationFunction(actFunc);


        if (!this->activationFunction || !this->derActivationFunction) {
            std::cerr << "Invalid activation function: " << actFunc << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    std::vector<float> forward(const std::vector<float>& inputs) {
        last_input = inputs; // storing the last inputs at layer level
        std::vector<float> outputs;
        outputs.reserve(neurons.size());

        for (auto& neuron : neurons) {
            outputs.push_back(neuron.activate(inputs, activationFunction));
        }
        return outputs;
    }

    void backPropagate_Layer(std::vector<float>& errors, float learning_rate) {
        std::vector<float> new_errors(last_input.size(), 0.0f);

        for (int i = 0; i < neurons.size(); i++) {
            std::vector<float> influences = neurons[i].backPropagate_Neuron(errors[i], last_input, learning_rate, derActivationFunction); // update new_errors here

            for (int j = 0; j < influences.size(); j++) {
                new_errors[j] += influences[j];
            }
        }

        errors = new_errors;
    }
        

	size_t getNumNeurons() const {
		return neurons.size();
	}

    const std::vector<Neuron>& getNeurons() const {
        return neurons;
    }
};