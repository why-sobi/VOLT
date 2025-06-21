#include <iostream>
#include <cstdlib>
#include <ctime>
#include "MLP.hpp"
#include "Layer.hpp"
//#include <opencv2/opencv.hpp>

int main() {
	// Initialize random seed
    std::srand(time(nullptr));
    
    // Layer with 3 neurons, each neuron takes 2 inputs
    Layer test_layer(3, 2);

    // Sample input (e.g. like AND gate)
    std::vector<float> input = { 1.0f, 0.5f };

    // Forward pass through the layer
    std::vector<float> output = test_layer.forward(input);

    // Print output
    std::cout << "Layer output: ";
    for (float val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}