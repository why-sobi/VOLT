#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "Activation.hpp"
#include "utils.hpp"
#include "Loss.hpp"

class Layer {
private:
	Eigen::MatrixX<float> weights;                                                          // Rows = Number of Neurons, Columns = Number of Inputs per Neuron
	Eigen::VectorX<float> biases;                                                           // Size = Number of Neurons
	Eigen::VectorX<float> last_output;                                                      // Stores the last output of the layer
	Eigen::VectorX<float> last_input;                                                       // Stores the last input to the layer
	Activation::ActivationType functionType;                                                // Type of activation function used in the layer

public:
    Layer(int num_neurons, int num_inputs_per_neuron, Activation::ActivationType& function) { 
		weights = Eigen::MatrixX<float>::Random(num_neurons, num_inputs_per_neuron);         // Random weights for each neuron (default range [-1, 1])
		biases = Eigen::VectorX<float>::Random(num_neurons);                                 // Random biases for each neuron (default range [-1, 1])

		this->functionType = function;                                                       // Need to store this here so that we can serialize the layer later
    }

    Eigen::VectorX<float> forward(const Eigen::VectorX<float>& inputs) {
		last_input = inputs;                                                                // storing the last inputs at layer level
        
        last_output = weights * inputs + biases;                                            // Calculation of forward pass (Ax + b) 
        
        Activate(last_output, functionType);
        return last_output;
    }

    void backPropagate_Layer(Eigen::VectorX<float>& errors, float learning_rate, const Loss::Type lossType) {
        // .size returns the number of elements in the last_input (since its a column vector its just number of rows)
		Eigen::VectorX<float> new_errors = Eigen::VectorX<float>::Zero(last_input.size());
        // Calculate deltas for each neuron (element-wise multiplication of errors and derivative of activation function) (not same as Matrix multiplication)
        Eigen::VectorX<float> deltas;
		
        if (functionType == Activation::ActivationType::Softmax && lossType == Loss::Type::CategoricalCrossEntropy) {
            deltas = errors;                                                                // Softmax derivative is handled differently (prediction - labels)
        } else {
            deltas = errors.cwiseProduct(last_output.unaryExpr(Activation::getDerivative(functionType))); 
		}

        new_errors = weights.transpose() * deltas;

		weights -= learning_rate * deltas * last_input.transpose();                         // Update weights (outer product of deltas and last_input)
        biases -= learning_rate * deltas;                                                   // Update the biases
       
		errors = new_errors;                                                                // Update the errors vector for the next layer
    }
       
     /*void backPropagate_Layer(Eigen::VectorX<float>& errors, float learning_rate) {
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
     */

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