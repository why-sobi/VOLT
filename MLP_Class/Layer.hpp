#pragma once

#include <iostream>
#include <vector>
#include "Neuron.hpp"

class Layer {
private:
    std::vector<Neuron> neurons;
    std::string activationFunc;

public:
    Layer(int num_neurons, int num_inputs_per_neuron, std::string actFunc = "") { // function null for now
        neurons.reserve(num_neurons);    // no unecessary reallocation 
        
        for (int i = 0; i < num_neurons; ++i) {
            neurons.emplace_back(num_inputs_per_neuron);
        }
        // neurons.emplace_back(num_inputs_per_neuron); same as neurons.push_back(Neuron(num_inputs_per_neuron));
        // emplace_back takes the int, checks for neuron constructor and calls it using the int (this results in no unnecessary allocation or assignment)

    }

    std::vector<float> forward(const std::vector<float>& inputs) {
        std::vector<float> outputs;
        outputs.reserve(neurons.size());

        for (auto& neuron : neurons) {
            outputs.push_back(neuron.activate(inputs));
        }
        return outputs;
    }

	size_t getNumNeurons() const {
		return neurons.size();
	}
};
