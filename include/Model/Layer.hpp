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
	Eigen::MatrixX<float> weights;                                                                // Rows = Number of Neurons, Columns = Number of Inputs per Neuron
    Eigen::MatrixX<float> last_batched_input;                                                     // Rows = Number of features, Columns = Numbers of Samples
	Eigen::MatrixX<float> last_batched_output;                                                    // Rows = Number of labels, Columns = Numbers of Samples
    
    Eigen::MatrixX<float> new_errors_buffer;                                                      // Used in backpropagation to store the new errors after backpropagating through the layer (to reduce heap allocs)
    Eigen::MatrixX<float> dW_buffer;                                                              // Used in backpropagation to store the deltas for the layer (to reduce heap allocs)
    Eigen::VectorX<float> dB_buffer;                                                              // Used in backpropagation to store the deltas for the layer (to reduce heap allocs)
    Eigen::MatrixX<float> deltas_buffer;                                                          // Used in backpropagation to store the deltas for the layer (to reduce heap allocs)     

    Eigen::VectorX<float> biases;                                                                 // Size = Number of Neurons
	Activation::ActivationType functionType;                                                      // Type of activation function used in the layer

public:
    Layer() {}                                                                              // default constructor shouldn't be used for general usage
    Layer(int num_neurons, int num_inputs_per_neuron, Activation::ActivationType& function) { 
		weights = Eigen::MatrixX<float>(num_neurons, num_inputs_per_neuron);                      // num_neurons x num_inputs_per_neuron
		biases = Eigen::VectorX<float>(num_neurons);                                        // num_neurons x 1

		this->functionType = function;                                                      // Need to store this here so that we can serialize the layer later
        
        Init::InitWeightsAndBias(weights, biases, Init::setupType(function));
    }

    Eigen::MatrixX<float> forward(const Eigen::MatrixX<float>& input) {
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
        Eigen::MatrixX<float>& errors, 
        const Loss::Type lossType, 
        Optimizer*& optimizer,
        int idx,
        float lambda,
        Regularization reg_type
    ) {
		this->new_errors_buffer.setZero(this->last_batched_input.rows(), this->last_batched_input.cols());
        
        // Calculate deltas for each neuron (element-wise multiplication of errors and derivative of activation function) (not same as Matrix multiplication)
		int batch_size = static_cast<int>(this->last_batched_input.cols());                   // Number of samples in the batch
        
        // Softmax derivative is handled differently (prediction - labels)
        if (this->functionType == Activation::ActivationType::Softmax && lossType == Loss::Type::CategoricalCrossEntropy) { this->deltas_buffer.noalias() = errors; } 
        else { this->deltas_buffer.noalias() = errors.cwiseProduct(last_batched_output.unaryExpr(Activation::getDerivative(functionType))); }

        // Matrix mul are heavy hence using buffers to store the results and reduce heap allocs by reusing them for each layer
        new_errors_buffer.noalias() = weights.transpose() * this->deltas_buffer;
        dW_buffer.noalias() = this->deltas_buffer * last_batched_input.transpose();                         // Weights effecting the outcome
        dB_buffer.noalias() = this->deltas_buffer.rowwise().sum();                                          // Biases affecting the outcome  

        regularizeGradient(this->dW_buffer, this->weights, batch_size, lambda, reg_type);
        optimizer->updateWeightsAndBiases(this->weights, biases, this->dW_buffer, this->dB_buffer, idx);    // Updating in here

        errors.noalias() = this->new_errors_buffer;                                                         // Update the errors vector for the next layer
    }

    void saveLayer(std::fstream& file) const {
        // writes the content of the layer (relevant ones) into a bin file
        io::writeEigenMatrix<float>(file, this->weights);
        io::writeEigenVector<float>(file, this->biases);
        io::writeEnum<Activation::ActivationType>(file, functionType);
    }

    void readLayer(std::fstream& file) { 
        // updates value in-place rather than returning the object
        this->weights = io::readEigenMatrix<float>(file);
        this->biases = io::readEigenVector<float>(file);
        this->functionType = io::readEnum<Activation::ActivationType>(file);
    }
       
	const int getNumNeurons() const { return static_cast<int>(weights.rows()); }
    const Eigen::MatrixX<float> getWeights() const { return weights; }
    const Eigen::VectorXf getBiases() const { return biases; }
    const std::string getActivationFunc() const { return Activation::actTypeToString(this->functionType); }
    const Activation::ActivationType getActivationType() const { return functionType; }
};