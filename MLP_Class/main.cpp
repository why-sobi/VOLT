#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <chrono>

#include <Eigen/Dense>

#include "./Model/MLP.hpp"
//#include "./Utility/store.hpp"

int step_function(float value) { return value < 0.5 ? 0 : 1;  }

int main() {
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

    MultiLayerPerceptron xor_model(X_xor.cols, Loss::Type::MSE, new Adam(0.01));
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
}




// TODO:

    // Fix normalizer usage (should be inside model class)
    // Split dataset into train and test (both stratified and random) [testing left]
    // Change Dataset and Label into the same class (same class name can be used for different purposes and also add a nested vector initializer) (named DataMatrix<T>)
    // Change model.train signature to accept Eigen matrices
    // Implement validation during training

    // Make save and load work
    // CMAKE setup (if wanna)


