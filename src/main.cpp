#include <iostream>
#include <chrono>
#include <omp.h>

#include <Model/MLP.hpp>


int main() {
    auto [X_train, y_train] = DataUtility::readCSV<float>("../datasets/mnist_train.csv", { "label" });
    auto [X_test, y_test]   = DataUtility::readCSV<float>("../datasets/mnist_test.csv", { "label" });
    y_train = DataUtility::one_hot_encode(y_train);
    y_test  = DataUtility::one_hot_encode(y_test);

    MultiLayerPerceptron model(
        static_cast<int>(X_train.cols),         // input size
        Regularization::L2,                     // regularization type
        0.0001f,                                // lambda (regularization strength)      
        Loss::Type::CategoricalCrossEntropy,    // loss function
        new Adam(0.01f)                         // optimizer (learning rate = 0.01f)
    );
    

    model.normalizer.fit_transform(X_train, NormalizeType::MinMax);
    model.normalizer.transform(X_test);


    model.addLayer(128, Activation::ActivationType::ReLU);
    model.addLayer(64, Activation::ActivationType::ReLU);
    model.addLayer(static_cast<int>(y_train.cols), Activation::ActivationType::Softmax);


    auto start = std::chrono::high_resolution_clock::now(); // Start the clock

    model.train(X_train, y_train, 30, 64, 3);               // The training happens here

    auto end   = std::chrono::high_resolution_clock::now(); // Stop the clock

    // Calculate the difference
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "TOTAL TRAINING TIME: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Accuracy: " << model.evaluate(X_test, y_test) * 100 << '\n';
        
    return 0;
}

// TEST SPLITS AND DIFFERENT MODEL TRAINING 


// TODO:

// 1. Autograd system
// 2. Polymorhpic Layer setup
// 3. No 'new' for Layer use alias for unique ptr for ease of use
// 4. Batch Norm
// 5. Pooling
// 6. Convolution Network


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