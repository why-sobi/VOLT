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
	MultiLayerPerceptron model(10, Loss::Type::MSE, new Adam(0.05));
    
    auto [X, y] = DataUtility::readCSV<float>("../datasets/ProcessedHousing.csv", {"price", "area"}, {"bedrooms"});
    model.fit_transform(X, y, NormalizeType::ZScore);

	auto [X_train, y_train, X_test, y_test] = DataUtility::train_test_split(X, y, 0.2, true);

    model.addLayer(32, Activation::ActivationType::Sigmoid);
    model.addLayer(2, Activation::ActivationType::Sigmoid);

    // new expected should be
     model.train(X_train, y_train, 1000, 32);

    // TODO:
	
    // Fix normalizer usage (should be inside model class)
	// Split dataset into train and test (both stratified and random) [testing left]
    // Change Dataset and Label into the same class (same class name can be used for different purposes and also add a nested vector initializer) (named Matrix<T>)
    // Change model.train signature to accept Eigen matrices
	
    // Implement validation during training
    // Make save and load work
    // CMAKE setup (if wanna)

    return 0;
}