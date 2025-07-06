#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "Pair.hpp" 
#include "MLP.hpp"
#include "Layer.hpp"

//#include <opencv2/opencv.hpp>

template <typename T>
std::ostream& operator << (std::ostream& out, const std::vector<T>& input) {
    out << '{';
    for (int i = 0; i < input.size(); i++) {
        out << input[i];
        if (i != input.size() - 1) { out << ", "; }
    }
    out << '}';
    return out;
}

int step_function(float value) { return value < 0.5 ? 0 : 1;  }

int main() {
    // Initialize random seed
    std::srand(time(nullptr));


    std::vector<Pair<std::vector<float>, std::vector<float>>> training_data =
    {
        { {0.0f, 0.0f}, {0.0f} },
        { {0.0f, 1.0f}, {1.0f} },
        { {1.0f, 0.0f}, {1.0f} },
        { {1.0f, 1.0f}, {0.0f} }
    };

    MultiLayerPerceptron model(2, 0.5); // Input size is 2 for XOR problem
    model.addLayer(2, "tanh"); // Hidden layer with 2 neurons
    model.addLayer(1, "sigmoid"); // Output layer with 1 neuron
    model.train(training_data, 6500);

    for (int i = 0; i < training_data.size(); i++) {
        std::cout << training_data[i].first << " => " << step_function(model.predict(training_data[i].first)[0]) << '\n';
    }

    return 0;
}