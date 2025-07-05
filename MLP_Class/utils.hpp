#pragma once

#include <iostream>
#include <cstdlib>
#include <ctime>
#include "Activation.hpp"

float getRandomFloat(float min, float max) {
	// Generate a random float between min and max

	// static_cast<float>(rand()) / static_cast<float>(RAND_MAX)  gives range (0, 1)
	// multiplying by (max - min) gives range (0, (max - min))
	// adding min gives range (min, max)
	
	// dry run if needed
	return (static_cast<float>(rand()) / (static_cast<float>(RAND_MAX))) * (max - min) + min;
}


std::function<float(float)> setActivationFunction(std::string& funcName) {
	auto it = Activation::activations.find(funcName);
	return it != Activation::activations.end() ? it->second : nullptr;
}

std::function<float(float)> setDerActivationFunction(std::string& funcName) {
	auto it = Activation::derivatives.find(funcName);
	return it != Activation::activations.end() ? it->second : nullptr;
}