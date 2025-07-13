#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include "MLP.hpp"

//#include <opencv2/opencv.hpp>

int step_function(float value) { return value < 0.5 ? 0 : 1;  }
//using Normalizer = DataUtil::Normalize::Type;

int main() {
    // Initialize random seed
    std::srand(time(nullptr));

	Normalizer normalizer;

    auto dataset = DataUtil::PreprocessDataset(
        "../datasets/ProcessedHousing.csv",
        {"price"},
        normalizer,
		NormalizeType::MinMax
    );

	for (const auto& sample : dataset) {
        std::cout << sample.first << " : " << sample.second << '\n';
	}


    return 0;
}