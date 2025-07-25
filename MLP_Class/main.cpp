#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include "MLP.hpp"
#include "store.hpp"

//#include <opencv2/opencv.hpp>

int step_function(float value) { return value < 0.5 ? 0 : 1;  }
//using Normalizer = DataUtil::Normalize::Type;

int main() {
    // Initialize random seed
    std::srand(time(nullptr));

    Normalizer normalizer;
    std::vector<DataUtil::Sample> training_data = DataUtil::PreprocessDataset(
    "../datasets/ProcessedHousing.csv",
        {"price", "area"},
		normalizer,
        NormalizeType::MinMax
    );

    MultiLayerPerceptron mlp(11, 0.5);
	mlp.addLayer(10, Activation::ActivationType::Tanh);
	mlp.addLayer(2, Activation::ActivationType::Sigmoid);
	
    mlp.train(training_data, 1000);

    //normalizer.test();

    //SaveModel(mlp, normalizer, "abc");

    return 0;
}