#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <chrono>

#include <Eigen/Dense>

#include "./Model/MLP.hpp"



int step_function(float value) { return value < 0.5 ? 0 : 1;  }

int main() {
	auto start = std::chrono::high_resolution_clock::now();

    auto [X, y] = DataUtility::readCSV<float>("../datasets/mnist.csv", { "label" });
    y = DataUtility::one_hot_encode(y);

    MultiLayerPerceptron model(X.cols, Loss::Type::CategoricalCrossEntropy, new Adam(0.01));
    model.normalizer.fit_transform(X, NormalizeType::MinMax);

    auto [X_train, y_train, X_test, y_test] = DataUtility::train_test_split(X, y);

    model.addLayer(128, Activation::ActivationType::ReLU);
    model.addLayer(64, Activation::ActivationType::ReLU);
    model.addLayer(y.cols, Activation::ActivationType::Softmax);

    model.train(X_train, y_train, 30, 64, 5);

    float correct = 0;
    for (int i = 0; i < X_test.rows; i++) {
        auto input = X_test(i);
        auto prediction = model.predict(input);
        int pred_class;
        prediction.maxCoeff(&pred_class);
        int true_class;
        y_test.asEigen().row(i).maxCoeff(&true_class);
        
		if (pred_class == true_class) correct++;
	}
	std::cout << "Accuracy: " << correct / X_test.rows << "\n";
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Data loading, preprocessing and model training took: " << duration << " ms\n";
    
    return 0;
}

// TEST SPLITS AND DIFFERENT MODEL TRAINING 


// TODO:

    // Fix normalizer usage (should be inside model class)
    // Split dataset into train and test (both stratified and random) [testing left]
    // Change Dataset and Label into the same class (same class name can be used for different purposes and also add a nested vector initializer) (named DataMatrix<T>)
    // Change model.train signature to accept Eigen matrices
    // Implement validation during training

    // Add different weight initializers
    // Make save and load work
    // CMAKE setup (if wanna)


/*
* EXAMPLE

// XOR dataset
    DataUtility::DataMatrix<float> X_xor({
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    });

    DataUtility::DataMatrix<float> y_xor({
        {0},
        {1},
        {1},
        {0}
    });

    MultiLayerPerceptron xor_model(X_xor.cols, Loss::Type::BinaryCrossEntropy, new Adam(0.01));
    xor_model.addLayer(4, Activation::ActivationType::ReLU);    // Small hidden layer
    xor_model.addLayer(y_xor.cols, Activation::ActivationType::Sigmoid);  // Binary output

    // No normalization needed for XOR (already 0s and 1s)
    xor_model.train(X_xor, y_xor, 5000, 4, 5);  // Batch size = 4 (all samples)

    // Test all 4 inputs
    for (int i = 0; i < 4; i++) {
        auto input = X_xor(i);
        auto prediction = xor_model.predict(input);
        std::cout << "Input: [" << input[0] << ", " << input[1]
            << "] -> Predicted: " << step_function(prediction[0])
            << ", Actual: " << y_xor(i, 0) << std::endl;
    }

*/