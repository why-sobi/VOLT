#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "Activation.hpp"

class Layer {
private:
	Eigen::MatrixX<float> weights;                                                          // Rows = Number of Neurons, Columns = Number of Inputs per Neuron
	Eigen::VectorX<float> biases;                                                           // Size = Number of Neurons
	Eigen::VectorX<float> last_output;                                                      // Stores the last output of the layer
	Eigen::VectorX<float> last_input;                                                       // Stores the last input to the layer
	Activation::ActivationType functionType;                                                // Type of activation function used in the layer
	std::function<float(float)> activationFunction;                                         // Pointer to the activation function
	std::function<float(float)> derActivationFunction;                                      // Pointer to the derivative of the activation function

public:
    Layer(int num_neurons, int num_inputs_per_neuron, Activation::ActivationType& function) { 
		weights = Eigen::MatrixX<float>::Random(num_neurons, num_inputs_per_neuron);         // Random weights for each neuron (default range [-1, 1])
		biases = Eigen::VectorX<float>::Random(num_neurons);                                 // Random biases for each neuron (default range [-1, 1])

		this->functionType = function;                                                       // Need to store this here so that we can serialize the layer later
        this->activationFunction = Activation::getActivation(function);                      // Don't need to serialize these function necessarily then
        this->derActivationFunction = Activation::getDerivative(function);


		if (!this->activationFunction || !this->derActivationFunction) {                    // if activation function or its derivative is not set
            std::cerr << "Invalid activation function: " << Activation::actTypeToString(function) << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    //Layer(std::vector<Neuron>& neurons, std::string& activationFunc) {
    //    this->neurons = neurons;
    //    //Activation::ActivationType func = Activation::stringToActivationType(activationFunc);
    //    this->functionType = Activation::stringToActivationType(activationFunc);
    //    this->activationFunction = Activation::getActivation(this->functionType);
    //    this->derActivationFunction = Activation::getDerivative(this->functionType);
    //}

    //std::vector<float> forward(const std::vector<float>& inputs) {
    //    last_input = inputs; // storing the last inputs at layer level
    //    std::vector<float> outputs;
    //    outputs.reserve(neurons.size());

    //    for (auto& neuron : neurons) {
    //        outputs.push_back(neuron.activate(inputs, activationFunction));
    //    }
    //    return outputs;
    //}

    Eigen::VectorX<float> forward(const Eigen::VectorX<float>& inputs) {
		last_input = inputs;                                                                // storing the last inputs at layer level
		last_output = (weights * inputs + biases).unaryExpr(activationFunction);            // Apply activation function to the linear combination of inputs and weights
		return last_output;
    }

    //void backPropagate_Layer(std::vector<float>& errors, float learning_rate) {
    //    std::vector<float> new_errors(last_input.size(), 0.0f);

    //    for (int i = 0; i < neurons.size(); i++) {
    //        std::vector<float> influences = neurons[i].backPropagate_Neuron(errors[i], last_input, learning_rate, derActivationFunction); // update new_errors here

    //        for (int j = 0; j < influences.size(); j++) {
    //            new_errors[j] += influences[j];
    //        }
    //    }

    //    errors = new_errors;
    //}

    void backPropagate_Layer(Eigen::VectorX<float>& errors, float learning_rate) {
        // .size returns the number of elements in the last_input (since its a column vector its just number of rows)
		Eigen::VectorX<float> new_errors = Eigen::VectorX<float>::Zero(last_input.size());

        for (int i = 0; i < weights.rows(); i++) {                                          // Iterating over each neuron
			float delta = errors(i) * derActivationFunction(last_output(i));                // Calculate delta for the neuron

            for (int j = 0; j < weights.cols(); j++) {                                      // Iterating over the neuron's weights
				new_errors(j) += weights(i, j) * delta;                                     // Update the new errors vector
				weights(i, j) -= learning_rate * delta * last_input(j);                     // Update the weight
            }

			biases(i) -= learning_rate * delta;                                             // Update the bias
        }
		errors = new_errors;                                                                // Update the errors vector for the next layer
    }
        

	const size_t getNumNeurons() const {
		return weights.rows();
	}
    /*
    const std::vector<Neuron>& getNeurons() const {
        return neurons;
    }*/

    const std::string getActivationFunc() const {
        return Activation::actTypeToString(this->functionType);
    }

};