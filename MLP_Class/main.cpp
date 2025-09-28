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
	MultiLayerPerceptron model;
    auto [X, y] = DataUtility::readCSV<float>("../datasets/ProcessedHousing.csv", {"price", "area"}, {"bedrooms"});
    model.fit_transform(X, y, NormalizeType::ZScore);
    //std::cout << y.asEigen() << '\n';


	auto [X_train, y_train, X_test, y_test] = DataUtility::train_test_split(X, y, 0.2, false);
    //std::cout << X_train.asEigen();
	std::cout << X.rows << " " << X_train.rows << " " << X_test.rows << '\n';


    // new expected should be
    // model.train(X_train.asEigen(), y_train.asEigen(), 1000, 32, 10);

    // TODO:
	
    // Fix normalizer usage (should be inside model class)
	
    // Split dataset into train and test (both stratified and random)
	// Change model.train signature to accept Eigen matrices
	// Implement validation during training
    // Make save and load work

    return 0;
}