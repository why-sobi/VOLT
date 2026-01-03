#pragma once

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <string>
#include <set> // ordered set
#include <unordered_set>
#include <unordered_map>
#include <Eigen/Dense>

#include "../Functions/Activation.hpp"
#include "store.hpp"

float getRandomFloat(float min, float max) {
	// Generate a random float between min and max

	// static_cast<float>(rand()) / static_cast<float>(RAND_MAX)  gives range (0, 1)
	// multiplying by (max - min) gives range (0, (max - min))
	// adding min gives range (min, max)
	
	// dry run if needed
	return (static_cast<float>(rand()) / (static_cast<float>(RAND_MAX))) * (max - min) + min;
}


void Activate(Eigen::VectorX<float>& input, const Activation::ActivationType function) {
	switch (function) {
		case Activation::ActivationType::Linear:
			break;
		case Activation::ActivationType::Sigmoid:
			for (int i = 0; i < input.size(); i++) {
				input(i) = Activation::sigmoid(input(i));
			}
			break;
		case Activation::ActivationType::Tanh:
			for (int i = 0; i < input.size(); i++) {
				input(i) = Activation::tanh_act(input(i));
			}
			break;
		case Activation::ActivationType::ReLU:
			for (int i = 0; i < input.size(); i++) {
				input(i) = Activation::relu(input(i));
			}
			break;
		case Activation::ActivationType::LeakyReLU:
			for (int i = 0; i < input.size(); i++) {
				input(i) = Activation::leaky_relu(input(i));
			}
			break;
		case Activation::ActivationType::ELU:
			for (int i = 0; i < input.size(); i++) {
				input(i) = Activation::elu(input(i));
			}
			break;
		case Activation::ActivationType::Softplus:
			for (int i = 0; i < input.size(); i++) {
				input(i) = Activation::softplus(input(i));
			}
			break;
		case Activation::ActivationType::Swish:
			for (int i = 0; i < input.size(); i++) {
				input(i) = Activation::swish(input(i));
			}
			break;
		case Activation::ActivationType::Softmax:
			input = Activation::softmax(input);
			break;
		default:
			std::cerr << "Unknown activation function type." << std::endl;
			exit(EXIT_FAILURE);
	}
}

template <typename T>
std::vector<T> set_diff(std::vector<T>& vec1, std::vector<T>& vec2) { // use this if you do simply want set A-B without sorting (maintains order of vec1)
	std::set<T> set_vec2(vec2.begin(), vec2.end());
	std::vector<T> result;
	for (const T& item : vec1) {
		if (set_vec2.find(item) == set_vec2.end()) {
			result.push_back(item);
		}
	}
	return result;
}

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

template <typename T>
void shuffle(std::vector<T>& arr) { // Using Fischer Yates Algorithm 
	for (size_t i = arr.size() - 1; i > 0; i--) {
		int j = rand() % (i + 1); // Generate a random index from 0 to i
		std::swap(arr[i], arr[j]); // Swap the elements at indices i and j
	}
}

template <typename T>
std::unordered_map<T, size_t> uniqueCount(const std::vector<T>& vec) {
	std::unordered_map<T, size_t> count;
	for (const T& val : vec) {
		count[val]++;
	}
	return count;
}

template <typename T>
size_t total_unique_values(const std::vector<T>& vec) {
	return std::unordered_set<T>(vec.begin(), vec.end()).size();
}