#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "../Functions/Activation.hpp"
#include "../Functions/Loss.hpp"
#include "../Utility/utils.hpp"
#include "../Optimizers/Optimizer.hpp"
#include "../Weights/Initializers.hpp"

class Layer {
public:
	Eigen::MatrixXf weights;                                                                // Rows = Number of Neurons, Columns = Number of Inputs per Neuron
    Eigen::MatrixXf last_batched_input;                                                     // Rows = Number of features, Columns = Numbers of Samples
	Eigen::MatrixXf last_batched_output;                                                    // Rows = Number of labels, Columns = Numbers of Samples
    
    Eigen::VectorX<float> biases;                                                           // Size = Number of Neurons
	Activation::ActivationType functionType;                                                // Type of activation function used in the layer

public:
    Layer(int num_neurons, int num_inputs_per_neuron, Activation::ActivationType& function) { 
		weights = Eigen::MatrixXf(num_neurons, num_inputs_per_neuron);                      // num_neurons x num_inputs_per_neuron
		biases = Eigen::VectorX<float>(num_neurons);                                        // num_neurons x 1

		this->functionType = function;                                                      // Need to store this here so that we can serialize the layer later
    
        Init::InitWeightsAndBias(weights, biases, Init::setupType(function));
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) {
        last_batched_input = input;                                                          // For backprop
        last_batched_output = (weights * input).colwise() + biases;                          // calculation AX+B

        if (functionType == Activation::ActivationType::Linear) {
			return last_batched_output;                                                     // Linear activation function just returns the output as is
        }
        else if (functionType == Activation::ActivationType::Softmax) {                     // special case as softmax is computed over the output vector not output values per neuron
            for (int nCol = 0; nCol < int(last_batched_output.cols()); nCol++) {
                last_batched_output.col(nCol) = Activation::softmax(last_batched_output.col(nCol));
            }
        }
        else {
            last_batched_output = last_batched_output.unaryExpr(Activation::getActivation(functionType));
        }

        return last_batched_output;
    }

    void backPropagate_Layer(Eigen::MatrixXf& errors, const Loss::Type lossType, Optimizer*& optimizer, int idx) {
        // .size returns the number of elements in the last_input (since its a column vector its just number of rows)
		Eigen::MatrixXf new_errors = Eigen::MatrixXf::Zero(last_batched_input.rows(), last_batched_input.cols());
        // Calculate deltas for each neuron (element-wise multiplication of errors and derivative of activation function) (not same as Matrix multiplication)
        Eigen::MatrixXf deltas;
		int batch_size = last_batched_input.cols();                                         // Number of samples in the batch
        if (functionType == Activation::ActivationType::Softmax && lossType == Loss::Type::CategoricalCrossEntropy) {
            deltas = errors;                                                                // Softmax derivative is handled differently (prediction - labels)
        } else {
            deltas = errors.cwiseProduct(last_batched_output.unaryExpr(Activation::getDerivative(functionType))); 
		}

        new_errors = weights.transpose() * deltas;

        auto dW = deltas * last_batched_input.transpose();             // Weights effecting the outcome
        auto dB = deltas.rowwise().sum();                              // Biases affecting the outcome

        optimizer->updateWeightsAndBiases(weights, biases, dW, dB, idx);

		//weights -= (learning_rate / batch_size) * deltas * last_batched_input.transpose();// Update weights (outer product of deltas and last_input)
        //biases -= (learning_rate / batch_size) * deltas.rowwise().sum();                  // Update the biases
		errors = new_errors;                                                                // Update the errors vector for the next layer
    }
       
	const size_t getNumNeurons() const {
		return weights.rows();
	}

    const Eigen::MatrixXf getWeights() const { return weights; }
    const Eigen::VectorXf getBiases() const { return biases; }

    const std::string getActivationFunc() const { return Activation::actTypeToString(this->functionType); }

    const Activation::ActivationType getActivationType() const { return functionType; }

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

};