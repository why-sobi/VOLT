# pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <fstream>

#include "Layer.hpp"
#include "../Utility/Pair.hpp"
#include "../Data/DataUtil.hpp"
#include "../Normalizer/Normalizer.hpp"

class MultiLayerPerceptron {
private:
	int input_size;
	std::vector<Layer> layers;																					// Vector of layers in the MLP (only has hidden and output layer, no such thing as input layer)
	std::vector<std::string> labels;
	
	Loss::Type lossType;																						// What loss function to use
	Optimizer* optimizer;																						// Optimizer type
	Normalizer normalizer;																						// Normalizer object to handle normalization and denormalization

public:
	MultiLayerPerceptron(std::string filename) {
		//this->load(filename); 
	}
	MultiLayerPerceptron(int input_size, Loss::Type lossFunctionName, Optimizer* optimizer) {
		this->input_size = input_size;
		this->lossType = lossFunctionName;
		this->labels = {};
		this->optimizer = optimizer;
	}
	~MultiLayerPerceptron() {
		if (optimizer) {
			delete optimizer;
		}
		optimizer = nullptr;
	}

	void fit_transform(DataUtility::DataMatrix<float>& dataset, DataUtility::DataMatrix<float>& labels, const NormalizeType normType) {
		normalizer = Normalizer();																			// reset normalizer
		// asEigen gives us a Map object which is like a matrix view of the underlying data
		auto dataset_matrix = dataset.asEigen();
		auto labels_matrix = labels.asEigen();

		// Normalize each feature/column
		for (int i = 0; i < dataset.cols; i++) {
			normalizer.normalize("Data " + std::to_string(i), dataset_matrix.col(i), normType);
		}

		for (int i = 0; i < labels.cols; i++) {
			normalizer.normalize("Label " + std::to_string(i), labels_matrix.col(i), normType);
		}
	}

	void transform(DataUtility::DataMatrix<float>& dataset, const NormalizeType normType) {
		auto dataset_matrix = dataset.asEigen();
		// Normalize each feature/column
		for (int i = 0; i < dataset.cols; i++) {
			normalizer.normalize("Data " + std::to_string(i), dataset_matrix.col(i), normType);
		}
	}

	void setLabels(const std::vector<std::string>& labels) {
		if (labels.size() == 0) {
			std::cerr << "Labels size cannot be zero!\n";
			std::exit(EXIT_FAILURE);
		}

		this->labels = labels;
		std::sort(this->labels.begin(),this->labels.end());
	}

	void addLayer(int num_neurons, Activation::ActivationType function) {
		if (layers.empty()) {																					// since first layer it'll num_of_inputs per neuron will be the input_size
			layers.emplace_back(num_neurons, input_size, function);
		}
		else {																									// subsequent layers will have num_of_inputs per neuron equal to the number of neurons in the previous layer
			int num_inputs_per_neuron = layers[layers.size() - 1].getNumNeurons();								// get number of neurons in the last layer
			layers.emplace_back(num_neurons, num_inputs_per_neuron, function);
		}
		int id = layers.size() - 1;																				// Setting up the optimizer (if using any other than SGD)
		optimizer->registerLayer(id, layers.back().getWeights(), layers.back().getBiases());

	}

	void train(DataUtility::DataMatrix<float>& X_train, DataUtility::DataMatrix<float>& y_train, const int epochs, const int batch_size, const int patience = 0) {
		// vector of inputs => corresponding to one output vector (depending on the size of output layer) (data) 
		// and vector of Pairs for batch processing
		if (layers.size() <= 1) {
			std::cerr << "Layers should be more than 1 to train!\n";
			std::exit(EXIT_FAILURE);
		}
		if (batch_size < 0) {
			std::cerr << "Batch size must be greater than 0!\n";
			std::exit(EXIT_FAILURE);
		}
		if (X_train.data.size() == 0 || y_train.data.size() == 0) {
			std::cerr << "Training data cannot be empty!\n";
			std::exit(EXIT_FAILURE);
		}
		if (epochs <= 0) {
			std::cerr << "Number of epochs must be greater than 0!\n";
			std::exit(EXIT_FAILURE);
		}
		if ((labels.size()) && (labels.size() != layers[layers.size() - 1].getNumNeurons())) {
			std::cerr << "Number of Labels(" << labels.size() << ") do not match the output layer's number of neurons(" << layers[layers.size() - 1].getNumNeurons() << ")\n";
			std::exit(EXIT_FAILURE);
		}
		if (patience < 0) {
			std::cerr << "Patience cannot be a non-negative number\n";
			std::exit(EXIT_FAILURE);
		}

		// asEigen gives us a Map object which is like a matrix view of the underlying data
		auto X_train_matrix = X_train.asEigen();
		auto y_train_matrix = y_train.asEigen();

		// early convergence check stuff
		int stale_loss = 0;
		float prev_loss = std::numeric_limits<float>::max();
		float min_delta = 1e-3f;

		std::vector<int> indexes(X_train.rows);																// total samples
		std::iota(indexes.begin(), indexes.end(), 0);														// filling the vector with range [0, X_train.rows) to use for shuffling

		int output_size = layers[layers.size() - 1].getNumNeurons();
		Activation::ActivationType output_activation = layers[layers.size() - 1].getActivationType();		// Used for calculating gradient and loss

		if (this->input_size != X_train.cols) {
			std::cerr << "Input size (" << this->input_size << ") does not match the dimensions of training features(" << X_train.cols << ")!\n";
			exit(EXIT_FAILURE);
		}

		if (output_size != y_train.cols) {
			std::cerr << "Ouput size (" << output_size << ") does not match the dimensions of Labels(" << y_train.cols << ")!\n";
			exit(EXIT_FAILURE);
		}

		for (int epoch = 0; epoch < epochs; ++epoch) {
			float epoch_error = 0.0f;
			shuffle(indexes);																				// shuffling indexes before each epoch

			for (int start = 0; start < X_train.rows; start += batch_size) {
				int end = std::min(start + batch_size, int(X_train.rows));									// checking if the remaining will sum to the batch size or fall short (last ones could fall short)
				int current_batch_size = end - start;														// either batch_size or < batch_size

				Eigen::MatrixX<float> batched_features(this->input_size, current_batch_size);				// input size is the number of features each sample would have
				Eigen::MatrixX<float> batched_labels(output_size, current_batch_size);						// output size would be the number of labels per sample

				// constructing our matrices
				for (int b = 0; b < current_batch_size; b++) {
					const auto& features = X_train_matrix.row(b);
					const auto& labels = y_train_matrix.row(b);
					// Each Column will represent a single Sample
					batched_features.col(b) = features;														
					batched_labels.col(b) = labels;
				}

				Eigen::MatrixXf batched_output = forwardPass(batched_features);

				if (batched_output.rows() != output_size) {
					std::cerr << "Batched Output size (" << batched_output.rows() << ") does not match expected output size (" << output_size << ")\n";
					std::exit(EXIT_FAILURE);
				}

				float error = Loss::CalculateLoss(batched_output, batched_labels, lossType);
				Eigen::MatrixXf propagatingErrors = Loss::CalculateGradient(batched_output, batched_labels, output_activation, lossType);

				epoch_error += error;

				backPropagation(propagatingErrors);															// Backpropagation to update weights and biases
			}
			if (patience != 0) {
				if (prev_loss - epoch_error / X_train.rows <= min_delta) {
					++stale_loss;
					if (stale_loss >= patience) {
						std::cout << "Early Stopping at epoch: " << epoch + 1 << " since model converged!\n";
						break;
					}
				}
				else {
					stale_loss = 0;
				}
				prev_loss = epoch_error / X_train.rows;
			}
			
			std::cout << "Epoch " << epoch + 1 << ", Avg Error: " << epoch_error / X_train.rows << " | Avg Accuracy: " << 1 - epoch_error / X_train.rows << "\n----------------------------------------------------------\n";
		}
	}

	Eigen::MatrixXf forwardPass(const Eigen::MatrixXf& input) {
		if (int(input.rows()) != this->input_size) {
			std::cerr << "The Batched Input features size: " << int(input.rows()) << " does not match the input size : " << this->input_size << '\n';
			std::exit(EXIT_FAILURE);
		}

		Eigen::MatrixXf current_input = input;
		for (auto& layer : layers) {
			current_input = layer.forward(current_input);
		}

		return current_input;
	}

	void backPropagation(Eigen::MatrixXf& errors) {

		for (int i = layers.size() - 1; i > -1; i--) {
			layers[i].backPropagate_Layer(errors, lossType, optimizer, i);
		}
	}

	Eigen::VectorX<float> forwardPass(const Eigen::VectorX<float>& input) {
		if (input.size() != input_size) {
			std::cerr << "Input size (" << input.size() << ") does not match the MLP input size (" << this->input_size << ")" << std::endl;
			std::exit(EXIT_FAILURE);
		}

		Eigen::VectorX<float> current_input = input;															// Initialize current_input with the input vector (will update it in each layer)
		for (auto& layer : layers) {
			current_input = layer.forward(current_input);														// Forward pass through each layer 
		}

		return current_input;																					// Return the output of the last layer
	}

	Eigen::VectorX<float> predict(std::vector<float>& input) {
		std::cout << "Making predictions with the MLP model..." << std::endl;
		Eigen::VectorX<float> input_vector = Eigen::Map<Eigen::VectorX<float>>(input.data(), input.size());
		return forwardPass(input_vector);
	}

	std::string LabelMaxPredict(const std::vector<float>& prediction) {
		if (labels.size() == 0) {
			std::cerr << "No Labels are set!\n";
			std::exit(EXIT_FAILURE);
		}
		if (prediction.size() != labels.size()) {
			std::cerr << "Prediction size (" << prediction.size() << ") does not match the number of labels (" << labels.size() << ")" << std::endl;
			std::exit(EXIT_FAILURE);
		}
		auto max_it = std::max_element(prediction.begin(), prediction.end());
		int index = std::distance(prediction.begin(), max_it);
		return labels[index];
	}

	std::unordered_map<std::string, float> getLabeledPrediction(const std::vector<float>& prediction) {
		if (labels.size() == 0) {
			std::cerr << "No Labels are set!\n";
			std::exit(EXIT_FAILURE);
		}
		if (prediction.size() != labels.size()) {
			std::cerr << "Prediction size (" << prediction.size() << ") does not match the number of labels (" << labels.size() << ")" << std::endl;
			std::exit(EXIT_FAILURE);
		}
		std::unordered_map<std::string, float> labeled_prediction;
		for (size_t i = 0; i < labels.size(); ++i) {
			labeled_prediction[labels[i]] = prediction[i];
		}
		return labeled_prediction;
	}

	const std::vector<std::string> getLabels() {
		if (this->labels.size() == 0) {
			std::cerr << "No Labels are set!\n";
			std::exit(EXIT_FAILURE);
		}

		return this->labels;
	}

	// Save And Load Function
	/*
	void save(const std::string& filename) const {
		std::cout << "Saving...\n";
		

		std::ofstream out(filename + ".bin", std::ios::binary);
		if (!out) throw std::runtime_error("Failed to open file for saving");

		size_t num_layers = layers.size();

		out.write(reinterpret_cast<const char*>(&this->input_size), sizeof(this->input_size));					// writing the input size
		out.write(reinterpret_cast<const char*>(&this->learning_rate), sizeof(this->learning_rate));			// writing the learning_rate
		out.write(reinterpret_cast<const char*>(&this->lossType), sizeof(this->lossType));						// writing the loss function type

		size_t num_labels = labels.size();
		out.write(reinterpret_cast<const char*>(&num_labels), sizeof(num_labels));								// writing the number of labels
		for (const auto& label : labels) {
			size_t length = label.length();
			out.write(reinterpret_cast<const char*>(&length), sizeof(length));									// storing length of label string
			out.write(reinterpret_cast<const char*>(label.c_str()), length);									// writing label as string
		}

		out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));								// writing the number of layers

		for (const auto& layer : layers) {
			std::string activationFunc = layer.getActivationFunc();
			size_t length = activationFunc.length();
			size_t totalNeurons = layer.getNumNeurons();
			std::vector<Neuron> neurons = layer.getNeurons();


			out.write(reinterpret_cast<const char*>(&length), sizeof(length));									// storing length of actFunc string
			out.write(reinterpret_cast<const char*>(activationFunc.c_str()), length);							// writing activation func as string
			out.write(reinterpret_cast<const char*>(&totalNeurons), sizeof(totalNeurons));						// writing total number of neurons

			for (const auto& neuron : neurons) {
				std::vector<float> weights = neuron.getWeights();
				size_t total_weights = weights.size();
				float bias = neuron.getBias();

				out.write(reinterpret_cast<const char*>(&bias), sizeof(bias));									// writing bias
				out.write(reinterpret_cast<const char*>(&total_weights), sizeof(total_weights));				// writing size of weights vector
				out.write(reinterpret_cast<const char*>(weights.data()), total_weights * sizeof(float));		// writing activation func as string
			}
		}

		out.close();
	}

	void load(std::string filename) {
		std::cout << "Loading...\n";

		std::ifstream in(filename + ".bin", std::ios::binary);
		if (!in) throw std::runtime_error("Cannot open file or it does not exist!");

		size_t num_layers, total_labels;
		
		// reading the MLP meta data (input size and learning rate)
		in.read(reinterpret_cast<char*>(&this->input_size), sizeof(this->input_size));							// reading input size
		in.read(reinterpret_cast<char*>(&this->learning_rate), sizeof(this->learning_rate));					// reading learning rate
		int type;
		in.read(reinterpret_cast<char*>(&type), sizeof(type));													// reading loss function type
		this->lossType = static_cast<Loss::Type>(type);															// converting int to Loss::Type enum

		in.read(reinterpret_cast<char*>(&total_labels), sizeof(total_labels));									// reading number of labels	
		labels.clear();
		labels.reserve(total_labels);																			// allocating memory here so no extra reallocation
		for (size_t i = 0; i < total_labels; i++) {
			size_t len;
			std::vector<char> buffer;
			in.read(reinterpret_cast<char*>(&len), sizeof(len));												// reading label string length
			buffer.resize(len);																					// resizing buffer to adequate length to store label
			in.read(buffer.data(), len);																		// reading label and storing in buffer
			labels.emplace_back(buffer.data(), len);															// storing label from buffer to labels vector
		}

		in.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));										// reading number of layers (input layer is not a layer)
		layers.clear();
		layers.reserve(num_layers);																				// allocating memory here so no extra reallocation

		for (int i = 0; i < num_layers; i++) {

			size_t len;
			size_t total_neurons;
			std::string activation_func;
			std::vector<char> buffer;
			
			// reading the activation function name
			in.read(reinterpret_cast<char*>(&len), sizeof(len));												// reading activation function string length
			buffer.resize(len);																					// resizing buffer to adequate length to store function name
			in.read(buffer.data(), len);																		// reading function name and storing in buffer
			activation_func.assign(buffer.data(), len);															// storing function name from buffer to a activation_func to the string


			// reading the number of neurons per layer
			in.read(reinterpret_cast<char*>(&total_neurons), sizeof(total_neurons));							// reading neurons per layer
			std::vector<Neuron> neurons;
			neurons.reserve(total_neurons);																		// to avoid unnecessary reallocation


			for (int j = 0; j < total_neurons; j++) {

				float bias;
				size_t total_weights;
				std::vector<float> weights;

				// readings bias, total weights and weights values
				in.read(reinterpret_cast<char*>(&bias), sizeof(bias));											// reading bias
				in.read(reinterpret_cast<char*>(&total_weights), sizeof(total_weights));						// reading number of weights
				weights.resize(total_weights);																	// pre-req to reading in a vector

				in.read(reinterpret_cast<char*>(weights.data()), total_weights * sizeof(float));				// reading actual weights

				// putting in the neurons vector
				neurons.emplace_back(bias, weights);															// re-creating the neuron and storing it

			}

			// putting in the layers vector
			layers.emplace_back(neurons, activation_func);														// re-creating the layer and storing it
		}
	}
	*/
};