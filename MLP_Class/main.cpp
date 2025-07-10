#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include "Pair.hpp" 
#include "MLP.hpp"
#include "Layer.hpp"
#include "DataUtil.hpp"


//#include <opencv2/opencv.hpp>

int step_function(float value) { return value < 0.5 ? 0 : 1;  }
//using Normalizer = DataUtil::Normalize::Type;

int main() {
    // Initialize random seed
    std::srand(time(nullptr));

    /*std::vector<Pair<std::vector<float>, std::vector<float>>> training_data =
    {
        { {0.0f, 0.0f}, {0.0f} },
        { {0.0f, 1.0f}, {1.0f} },
        { {1.0f, 0.0f}, {1.0f} },
        { {1.0f, 1.0f}, {0.0f} }
    };
    MultiLayerPerceptron model("XOR_MODEL");

    for (auto& input : training_data) {
        std::cout << input.first << " => " << step_function(model.predict(input.first)[0]) << '\n';
    }*/

    std::vector<DataUtil::Sample> dataset = DataUtil::PreprocessDataset("../datasets/ProcessedHousing.csv", {"price"}, DataUtil::Normalize::Type::ZScore);

    for (const DataUtil::Sample& sample : dataset) {
        std::cout << sample.first << " : " << sample.second << '\n';
    }
    return 0;
}