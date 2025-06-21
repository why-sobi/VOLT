#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "Pair.hpp" // or whatever your actual path is
#include "MLP.hpp"
#include "Layer.hpp"

//#include <opencv2/opencv.hpp>

int main() {
	// Initialize random seed
    std::srand(time(nullptr));
    

    std::vector<Pair<std::vector<float>, std::vector<float>>> training_data = {
    { {0.0f, 0.0f}, {0.0f} },
    { {0.0f, 1.0f}, {1.0f} },
    { {1.0f, 0.0f}, {1.0f} },
    { {1.0f, 1.0f}, {0.0f} }
    };

	MultiLayerPerceptron model(2); // Input size is 2 for XOR problem
	model.addLayer(2, "sigmoid"); // Hidden layer with 2 neurons
	model.addLayer(1, "sigmoid"); // Output layer with 1 neuron
    model.train(training_data, 10);

    return 0;
}