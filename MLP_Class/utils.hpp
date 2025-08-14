#pragma once

#include <iostream>
#include <cstdlib>
#include <ctime>

#include <Eigen/Dense>

#include "Activation.hpp"

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

//void DerActivation(Eigen::VectorX<float>& input, const Activation::ActivationType function) {
//	switch (function) {
//	case Activation::ActivationType::Sigmoid:
//		input = input.unaryExpr(Activation::d_sigmoid);
//		break;
//	case Activation::ActivationType::Tanh:
//		input = input.unaryExpr(Activation::d_tanh);
//		break;
//	case Activation::ActivationType::ReLU:
//		input = input.unaryExpr(Activation::d_relu);
//		break;
//	case Activation::ActivationType::LeakyReLU:
//		input = input.unaryExpr(Activation::d_leaky_relu);
//		break;
//	case Activation::ActivationType::Linear:
//		input = input.unaryExpr(Activation::d_linear);
//		break;
//	case Activation::ActivationType::ELU:
//		input = input.unaryExpr(Activation::d_elu);
//		break;
//	case Activation::ActivationType::Softplus:
//		input = input.unaryExpr(Activation::d_softplus);
//		break;
//	case Activation::ActivationType::Swish:
//		input = input.unaryExpr(Activation::d_swish);
//		break;
//	case Activation::ActivationType::Softmax:
//		input = Activation::d_softmax(input);
//		break;
//	default:
//		std::cerr << "Unknown activation function type." << std::endl;
//		exit(EXIT_FAILURE);
//	}
//}

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