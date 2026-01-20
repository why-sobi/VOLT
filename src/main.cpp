#include <iostream>
#include <chrono>
#include <omp.h>

#include <Model/MLP.hpp>


int main() {
    auto [X, y] = DataUtility::readCSV<float>("../datasets/mnist.csv", { "label" });
    y = DataUtility::one_hot_encode(y);
    auto [X_train, y_train, X_test, y_test] = DataUtility::train_test_split(X, y, 0.3f);


    MultiLayerPerceptron model(
        static_cast<int>(X_train.cols),
        Regularization::L2,
        0.005f,
        Loss::Type::CategoricalCrossEntropy,
        new Adam(0.01f)
    );


    model.normalizer.fit_transform(X_train, NormalizeType::MinMax);
    model.normalizer.transform(X_test);

    model.addLayer(128, Activation::ActivationType::ReLU);
    model.addLayer(64, Activation::ActivationType::ReLU);
    model.addLayer(static_cast<int>(y.cols), Activation::ActivationType::Softmax);

    auto start = std::chrono::high_resolution_clock::now(); // Start the clock

    model.train(X_train, y_train, 30, 64); // The training happens here

    auto end = std::chrono::high_resolution_clock::now(); // Stop the clock

    // Calculate the difference
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "TOTAL TRAINING TIME: " << elapsed.count() << " seconds" << std::endl;

    std::cout << "Accuracy: " << model.evaluate(X_test, y_test) * 100 << '\n';
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
    // Add Regularization (L2)

    // Make save and load work
    // Write extensive documentation (will give a deep dive and theory and also revision)
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