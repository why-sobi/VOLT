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

    /*
    std::unordered_map<std::string, DataUtil::Normalize::Type> norm_map = {{"price" , DataUtil::Normalize::Type::ZScore}};

    // Reading ProcessedHousing csv, whose labels are {"price"} and applying ZScore normalization {other options are None & MinMax} 
    std::vector<DataUtil::Sample> dataset = DataUtil::PreprocessDataset("../datasets/ProcessedHousing.csv", // path of file
                                                                        {"price"},                          // labels
                                                                        DataUtil::Normalize::Type::None,    // fallback normalization technique
                                                                        norm_map);                          // assigned norm techniques to columns

    for (const DataUtil::Sample& sample : dataset) {
        std::cout << sample.first << " : " << sample.second << '\n';
    }
    */

    /* Load and preprocess dataset
    //std::vector<DataUtil::Sample> dataset = DataUtil::PreprocessDataset(
    //    "../datasets/ProcessedHousing.csv",   // path to dataset
    //    { "price" },                          // output/label
    //    DataUtil::Normalize::Type::ZScore     // fallback normalization (no need for norm map since every column is getting ZScore-d)
    //    );

    //// Create MLP
    //int inputSize = dataset[0].first.size();
    //MultiLayerPerceptron model(inputSize, 0.01f); // inputSize, learningRate

    //// Add layers: hidden layer (say 8 neurons), then output layer (1 neuron for regression)
    //model.addLayer(8, Activation::ActivationType::ReLU);
    //model.addLayer(1, Activation::ActivationType::Linear); // linear output for regression

    //// Train
    //model.train(dataset, 1000);

    //// Save model
    //model.save("housing_model"); */

    std::vector<float> input = {
    7420,   // area
    4,      // bedrooms
    2,      // bathrooms
    3,      // stories
    1,      // mainroad (1 = yes, 0 = no)
    1,      // guestroom
    1,      // basement
    0,      // hotwaterheating
    1,      // airconditioning
    3,      // parking
    1,      // prefarea
    1       // furnishingstatus (maybe: 0 = unfurnished, 1 = semi-furnished, 2 = furnished)
    };

    
    DataUtil::Normalize::apply(input, DataUtil::Normalize::Type::ZScore);

    MultiLayerPerceptron model("housing_model");


    return 0;
}