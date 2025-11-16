#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "../Functions/Activation.hpp"
#include "../Functions/Loss.hpp"
#include "../Utility/utils.hpp"
#include "../Optimizers/Optimizer.hpp"
#include "../Weights/Initializers.hpp"
#include "../Regularization/Regularization.hpp"

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

    void backPropagate_Layer(
        Eigen::MatrixXf& errors, 
        const Loss::Type lossType, 
        Optimizer*& optimizer,
        int idx,
        float lambda,
        Regularization type
    ) {
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

        Eigen::MatrixX<float> dW = deltas * last_batched_input.transpose();             // Weights effecting the outcome
        Eigen::VectorX<float> dB = deltas.rowwise().sum();                              // Biases affecting the outcome

        // Adding regularization added around 10-15 seconds in training time need to check why    (or maybe it doesnt weird things man)     

        regularizeGradient(dW, weights, batch_size, lambda, type);
        optimizer->updateWeightsAndBiases(weights, biases, dW, dB, idx); // Updating in here
		errors = new_errors;                                           // Update the errors vector for the next layer
    }
       
	const size_t getNumNeurons() const {
		return weights.rows();
	}

    const Eigen::MatrixXf getWeights() const { return weights; }
    const Eigen::VectorXf getBiases() const { return biases; }

    const std::string getActivationFunc() const { return Activation::actTypeToString(this->functionType); }

    const Activation::ActivationType getActivationType() const { return functionType; }

};