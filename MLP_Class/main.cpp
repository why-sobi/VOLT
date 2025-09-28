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
    auto [X_train, y_train] = DataUtility::readCSV<float>("../datasets/ProcessedHousing.csv", {"price", "area"}, {"bedrooms"});
    //std::cout << y_train << '\n';

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