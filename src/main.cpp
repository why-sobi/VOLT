#include <iostream>
#include <chrono>

#include <Model/MLP.hpp>

int main() {
    std::cout << "Training MLP on MNIST dataset..." << std::endl;

    auto [X_train, y_train] = DataUtility::readCSV<float>("../datasets/mnist_train.csv", { "label" });
    auto [X_test, y_test]   = DataUtility::readCSV<float>("../datasets/mnist_test.csv", { "label" });
    y_train = DataUtility::one_hot_encode(y_train);
    y_test  = DataUtility::one_hot_encode(y_test);

    std::cout << "Training samples: " << X_train.rows << ", Test samples: " << X_test.rows << std::endl;

    MultiLayerPerceptron model(
        static_cast<int>(X_train.cols),         // input size
        Regularization::L2,                     // regularization type
        0.0001f,                                // lambda (regularization strength)      
        Loss::Type::CategoricalCrossEntropy,    // loss function
        new Adam(0.01f)                         // optimizer (learning rate = 0.01f)
    );
    

    model.normalizer.fit(X_train, NormalizeType::MinMax);
    model.normalizer.transform(X_train);
    model.normalizer.transform(X_test);


    model.addLayer(128, Activation::ActivationType::ReLU);
    model.addLayer(64, Activation::ActivationType::ReLU);
    model.addLayer(static_cast<int>(y_train.cols), Activation::ActivationType::Softmax);


    auto start = std::chrono::high_resolution_clock::now(); // Start the clock

    model.train(X_train, y_train, 30, 64, 2);               // The training happens here

    auto end   = std::chrono::high_resolution_clock::now(); // Stop the clock

    // Calculate the difference
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "TOTAL TRAINING TIME: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Accuracy: " << model.evaluate(X_test, y_test) * 100 << '\n';
        
    return 0;
}

