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

#define EIGEN_USE_THREADS

using namespace DataUtility;

class MultiLayerPerceptron {
private:
	int input_size;
	float lambda;
	std::vector<Layer> layers;																					// Vector of layers in the MLP (only has hidden and output layer, no such thing as input layer)
	Optimizer* optimizer;																						// Optimizer type
	Loss::Type lossType;																						// What loss function to use
	Regularization type;


	float calculateAccuracy(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& labels) const  {
		if (lossType == Loss::Type::CategoricalCrossEntropy) {
			// Multi-class classification: argmax comparison
			int correct = 0;
			for (int i = 0; i < predictions.cols(); i++) {
				int pred_class, true_class;
				predictions.col(i).maxCoeff(&pred_class);
				labels.col(i).maxCoeff(&true_class);
				if (pred_class == true_class) correct++;
			}
			return float(correct) / predictions.cols();

		}
		else if (lossType == Loss::Type::BinaryCrossEntropy) {
			// Binary classification: threshold at 0.5
			int correct = 0;
			for (int i = 0; i < predictions.cols(); i++) {
				for (int j = 0; j < predictions.rows(); j++) {
					int pred = predictions(j, i) >= 0.5 ? 1 : 0;
					int actual = labels(j, i) >= 0.5 ? 1 : 0;
					if (pred == actual) correct++;
				}
			}
			return float(correct) / (predictions.rows() * predictions.cols());

		}
		else {
			float tolerance = 0.05f;  
			return ((labels - predictions).array().abs() <= tolerance).cast<float>().mean();
		}
	}

	std::pair<float, float> validation_helper
	(
		DataMatrix<float>& X_val,
		DataMatrix<float>& y_val,
		const int batch_size,
		const int output_size,
		const Activation::ActivationType output_activation
	) {
		auto X_val_matrix = X_val.asEigen();
		auto y_val_matrix = y_val.asEigen();

		float val_error = 0.0f, val_accuracy = 0.0f;


		for (int start = 0; start < X_val.rows; start += batch_size) {
			int end = std::min(start + batch_size, int(X_val.rows));									// checking if the remaining will sum to the batch size or fall short (last ones could fall short)
			int current_batch_size = end - start;														// either batch_size or < batch_size

			Eigen::MatrixX<float> batched_features(this->input_size, current_batch_size);				// input size is the number of features each sample would have
			Eigen::MatrixX<float> batched_labels(output_size, current_batch_size);						// output size would be the number of labels per sample

			// constructing our matrices
			for (int b = 0; b < current_batch_size; b++) {
				const auto& features = X_val_matrix.row(start + b);
				const auto& labels = y_val_matrix.row(start + b);
				// Each Column will represent a single Sample
				batched_features.col(b) = features.transpose();
				batched_labels.col(b) = labels.transpose();
			}

			forwardPass(batched_features);

			if (batched_features.rows() != output_size) {
				std::cerr << "Batched Output size (" << batched_features.rows() << ") does not match expected output size (" << output_size << ")\n";
				std::exit(EXIT_FAILURE);
			}

			float error = Loss::CalculateLoss(batched_features, batched_labels, lossType);
			Eigen::MatrixXf propagatingErrors = Loss::CalculateGradient(batched_features, batched_labels, output_activation, lossType);

			val_error += error;
			val_accuracy += calculateAccuracy(batched_labels, batched_features);
		}

		return { val_error, val_accuracy };
	}
	

	std::pair<float, float> train_helper
	(
		DataMatrix<float>& X_train,
		DataMatrix<float>& y_train,
		const int batch_size,
		const int output_size,
		const std::vector<int>& indices,
		const Activation::ActivationType output_activation
	) {
		auto X_train_matrix = X_train.asEigen();
		auto y_train_matrix = y_train.asEigen();

		float train_error = 0.0f, train_accuracy = 0.0f;

		long long f_time = 0, e_time = 0, grad_time = 0, prop_time = 0;

		for (int start = 0; start < X_train.rows; start += batch_size) {
			int end = std::min(start + batch_size, int(X_train.rows));									// checking if the remaining will sum to the batch size or fall short (last ones could fall short)
			int current_batch_size = end - start;														// either batch_size or < batch_size

			Eigen::MatrixX<float> batched_features(this->input_size, current_batch_size);				// input size is the number of features each sample would have
			Eigen::MatrixX<float> batched_labels(output_size, current_batch_size);						// output size would be the number of labels per sample

			// constructing our matrices
			for (int b = 0; b < current_batch_size; b++) {
				// Each Column will represent a single Sample
				batched_features.col(b) = X_train_matrix.row(indices[start + b]);
				batched_labels.col(b) = y_train_matrix.row(indices[start + b]);
			}

			// batched_featuers are now transformed batched_output (it is updated in place)
			auto f_start = std::chrono::high_resolution_clock::now();
			forwardPass(batched_features);
			auto f_end = std::chrono::high_resolution_clock::now();
			f_time += std::chrono::duration_cast<std::chrono::milliseconds>(f_end - f_start).count();


			if (batched_features.rows() != output_size) {
				std::cerr << "Batched Output size (" << batched_features.rows() << ") does not match expected output size (" << output_size << ")\n";
				std::exit(EXIT_FAILURE);
			}

			auto e_start = std::chrono::high_resolution_clock::now();
			float error = Loss::CalculateLoss(batched_features, batched_labels, lossType);
			auto e_end = std::chrono::high_resolution_clock::now();
			e_time += std::chrono::duration_cast<std::chrono::milliseconds>(e_end - e_start).count();


			Eigen::MatrixXf propagatingErrors = Loss::CalculateGradient(batched_features, batched_labels, output_activation, lossType);
			auto CG_end = std::chrono::high_resolution_clock::now();
			grad_time += std::chrono::duration_cast<std::chrono::milliseconds>(CG_end - e_end).count();


			train_error += error;
			train_accuracy += calculateAccuracy(batched_features, batched_labels);
			
			auto bp_start = std::chrono::high_resolution_clock::now();
			backPropagation(propagatingErrors);															// Backpropagation to update weights and biases
			auto bp_end = std::chrono::high_resolution_clock::now();
			prop_time += std::chrono::duration_cast<std::chrono::milliseconds>(bp_end - bp_start).count();

		}
		std::cout << "Forward Pass: " << f_time <<
			"(ms) | Error Calc: " << e_time <<
			"(ms) | Grad Calc: " << grad_time <<
			"(ms) | Back Prop: " << prop_time << "(ms)\n";
		return {train_error, train_accuracy};
	}

public:
	Normalizer normalizer;																						// Normalizer object to handle normalization and denormalization

	MultiLayerPerceptron(std::string filename) {
		this->load(filename); 
	}
	MultiLayerPerceptron(int input_size, Regularization type, float lambda, Loss::Type lossFunctionName, Optimizer* optimizer) {
		if (!optimizer) {
			std::cerr << "Optimizer cannot be a nullptr!\n";
			std::exit(EXIT_FAILURE);
		}
		this->input_size = input_size;
		this->lossType = lossFunctionName;
		this->type = type;
		this->lambda = lambda;
		this->optimizer = optimizer;
		normalizer = Normalizer();

	}
	~MultiLayerPerceptron() {
		if (optimizer) {
			delete optimizer;
		}
		optimizer = nullptr;
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

	void train(
		DataMatrix<float>& X_train, 
		DataMatrix<float>& y_train, 
		const int epochs, 
		const int batch_size, 
		const int patience = 0
	) {
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
		if (patience < 0) {
			std::cerr << "Patience cannot be a non-negative number\n";
			std::exit(EXIT_FAILURE);
		}

		// early convergence check stuff
		int stale_loss = 0;
		float prev_loss = std::numeric_limits<float>::max();
		constexpr float min_delta = 1.5e-3f;

		std::vector<int> indexes(X_train.rows);																// total samples
		std::iota(indexes.begin(), indexes.end(), 0);														// filling the vector with range [0, X_train.rows) to use for shuffling

		int output_size = layers[layers.size() - 1].getNumNeurons();
		int num_batches = (X_train.rows + batch_size - 1) / batch_size;
		Activation::ActivationType output_activation = layers[layers.size() - 1].getActivationType();		// Used for calculating gradient and loss

		if (this->input_size != X_train.cols) {
			std::cerr << "Input size (" << this->input_size << ") does not match the dimensions of training features(" << X_train.cols << ")!\n";
			exit(EXIT_FAILURE);
		}

		if (output_size != y_train.cols) {
			std::cerr << "Output size (" << output_size << ") does not match the dimensions of Labels(" << y_train.cols << ")!\n";
			exit(EXIT_FAILURE);
		}

		for (int epoch = 0; epoch < epochs; ++epoch) {
			shuffle(indexes);																				// shuffling indexes before each epoch
			auto [train_error, train_accuracy] = train_helper(X_train, y_train, batch_size, output_size, indexes, output_activation);
			
			train_error /= num_batches;
			train_accuracy /= num_batches;

			std::cout << "Epoch " << epoch + 1 << 
				" | Train Loss: " << train_error << 
				" | Train Accuracy: " << train_accuracy <<
				"\n----------------------------------------------------------\n";
			if (patience != 0) {
				if ((prev_loss - train_error) <= min_delta) {
					++stale_loss;
					if (stale_loss >= patience) {
						std::cout << "Early Stopping at epoch: " << epoch + 1 << " since model converged!\n";
						break;
					}
				}
				else {
					stale_loss = 0;
				}
				prev_loss = train_error;
			}
		}
	}

	void train(
		DataMatrix<float>& X_train, 
		DataMatrix<float>& y_train, 
		DataMatrix<float>& X_val, 
		DataMatrix<float>& y_val, 
		const int epochs, 
		const int batch_size, 
		const int patience = 0
	) {
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
		if (X_train.data.size() == 0 || y_train.data.size() == 0 || X_val.data.size() == 0 || y_val.data.size() == 0) {
			std::cerr << "Any Data Matrix cannot be empty!\n";
			std::exit(EXIT_FAILURE);
		}
		if (X_train.cols != X_val.cols || y_train.cols != y_val.cols) {
			std::cerr << "Validation Dimensions do not match train dimensions!\n";
			std::exit(EXIT_FAILURE);
		}
		if (epochs <= 0) {
			std::cerr << "Number of epochs must be greater than 0!\n";
			std::exit(EXIT_FAILURE);
		}
		if (patience < 0) {
			std::cerr << "Patience cannot be a non-negative number\n";
			std::exit(EXIT_FAILURE);
		}

		// asEigen gives us a Map object which is like a matrix view of the underlying data
		auto X_val_matrix = X_train.asEigen();
		auto y_val_matrix = y_val.asEigen();

		// early convergence check stuff
		int stale_loss = 0;
		float prev_loss = std::numeric_limits<float>::max(), test_accuracy, val_accuracy;
		float min_delta = 1e-3f, tolerance = 1e-6f;

		std::vector<int> indexes(X_train.rows);																// total samples
		std::iota(indexes.begin(), indexes.end(), 0);														// filling the vector with range [0, X_train.rows) to use for shuffling

		int output_size = layers[layers.size() - 1].getNumNeurons();
		int num_batches = (X_train.rows + batch_size - 1) / batch_size;
		Activation::ActivationType output_activation = layers[layers.size() - 1].getActivationType();		// Used for calculating gradient and loss

		if (this->input_size != X_train.cols) {
			std::cerr << "Input size (" << this->input_size << ") does not match the dimensions of training features(" << X_train.cols << ")!\n";
			exit(EXIT_FAILURE);
		}

		if (output_size != y_train.cols) {
			std::cerr << "Output size (" << output_size << ") does not match the dimensions of Labels(" << y_train.cols << ")!\n";
			exit(EXIT_FAILURE);
		}

		for (int epoch = 0; epoch < epochs; ++epoch) {
			shuffle(indexes);																				// shuffling indexes before each epoch
			auto [train_error, train_accuracy] = train_helper(X_train, y_train, batch_size, output_size, indexes, output_activation);
			train_error /= num_batches;
			train_accuracy /= num_batches;

			auto [val_error, val_accuracy] = validation_helper(X_val, y_val, batch_size, output_size, output_activation);
			val_error /= X_val.rows;
			val_accuracy /= X_val.rows;


			std::cout << "Epoch " << epoch + 1 << 
				", Train Loss: " << train_error << " | Train Accuracy: " << train_accuracy <<
				", Validation Loss: " << val_error << " | Validation Accuracy: " << val_accuracy <<
				"\n----------------------------------------------------------\n";

			if (patience != 0) {
				if ((prev_loss - val_error) <= min_delta) {
					++stale_loss;
					if (stale_loss >= patience) {
						std::cout << "Early Stopping at epoch: " << epoch + 1 << " since model converged!\n";
						break;
					}
				}
				else {
					stale_loss = 0;
				}
				prev_loss = val_error;
			}
		}
	}

	void backPropagation(Eigen::MatrixXf& errors) {
		for (int i = layers.size() - 1; i > -1; i--) {
			layers[i].backPropagate_Layer(errors, lossType, optimizer, i, this->lambda, this->type);
		}
	}

	void forwardPass(Eigen::MatrixXf& input) {
		if (int(input.rows()) != this->input_size) {
			std::cerr << "The Batched Input features size: " << int(input.rows()) << " does not match the input size : " << this->input_size << '\n';
			std::exit(EXIT_FAILURE);
		}

		for (auto& layer : layers) {
			input = layer.forward(input);
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

	Eigen::VectorX<float> predict(const std::span<float>& input) {
		Eigen::VectorX<float> input_vector = Eigen::Map<Eigen::VectorX<float>>(input.data(), input.size());
		return forwardPass(input_vector);
	}

	// batching and forward passing
	/*Eigen::MatrixX<float> predict(DataUtility::DataMatrix<float>& X_test, int batch_size = 32) {
		Eigen::MatrixX<float> predictions(X_test.rows, layers[layers.size() - 1].getNumNeurons());

	}*/

	float evaluate(DataMatrix<float>& X_test, DataMatrix<float>& y_test) {
		Eigen::MatrixX<float> predictions(X_test.rows, y_test.cols); // not single col (if softmax or multilabel and so on ...)
		auto y_test_matrix = y_test.asEigen();

		for (int i = 0; i < X_test.rows; i++) {
			predictions.row(i) = this->predict(X_test(i));
		}

		if (lossType == Loss::Type::CategoricalCrossEntropy ||
			lossType == Loss::Type::BinaryCrossEntropy ||
			lossType == Loss::Type::HingeLoss) {

			int correct = 0;

			if (lossType == Loss::Type::BinaryCrossEntropy) {
				// Binary classification: threshold at 0.5
				for (int i = 0; i < X_test.rows; i++) {
					int pred_class = predictions(i, 0) >= 0.5 ? 1 : 0;
					int true_class = y_test(i, 0) >= 0.5 ? 1 : 0;
					if (pred_class == true_class) correct++;
				}
			}
			else {
				// Multi-class: argmax
				for (int i = 0; i < X_test.rows; i++) {
					int pred_class, true_class;
					predictions.row(i).maxCoeff(&pred_class);
					y_test_matrix.row(i).maxCoeff(&true_class);
					if (pred_class == true_class) correct++;
				}
			}

			return static_cast<float>(correct) / X_test.rows;  // Accuracy
		}

		// Regression metrics (for MSE)
		else if (lossType == Loss::Type::MSE) {
			// Return R² score (coefficient of determination)
			// R² = 1 - (SS_res / SS_tot)

			// Mean of actual values
			float y_mean = y_test_matrix.mean();
			float ss_tot = (y_test_matrix.array() - y_mean).square().sum();
			float ss_res = (y_test_matrix - predictions).array().square().sum();

			// R² score
			float r2 = 1.0f - (ss_res / ss_tot);

			return r2;  // Returns value typically between -inf and 1.0
			// 1.0 = perfect, 0.0 = baseline, <0 = worse than baseline
		}

		// Unknown loss type
		else {
			throw std::runtime_error("Unsupported loss type for evaluation");
		}
	}

	// Save And Load Function
	void save(const std::string& filename) const {
		std::fstream file(filename + ".bin", std::ios::out | std::ios::binary);

		io::writeNumeric<int>(file, this->input_size);
		io::writeNumeric<float>(file, this->lambda);

		io::writeEnum<Loss::Type>(file, this->lossType);
		io::writeEnum<Regularization>(file, this->type);

		// vector of layers
		io::writeNumeric<size_t>(file, this->layers.size());
		for (const Layer& layer : layers) {
			layer.saveLayer(file);
		}
		
		this->optimizer->saveOptimizer(file);
		this->normalizer.saveNormalizer(file);

		file.close();
	}

	void load(std::string filename) {
		std::fstream file(filename + ".bin", std::ios::in | std::ios::binary);

		this->input_size = io::readNumeric<int>(file);
		this->lambda = io::readNumeric<float>(file);

		this->lossType = io::readEnum<Loss::Type>(file);
		this->type = io::readEnum<Regularization>(file);

		// vector of layers
		size_t size = io::readNumeric<size_t>(file);
		this->layers = std::vector<Layer>(size);

		for (size_t s = 0; s < size; s++) {
			this->layers[s].readLayer(file);
		}
		if (this->optimizer) {
			delete optimizer;
			optimizer = nullptr;
		}
		this->optimizer = Optimizer::loadFromFile(file);
		this->normalizer.readNormalizer(file);

		file.close();
	}
};