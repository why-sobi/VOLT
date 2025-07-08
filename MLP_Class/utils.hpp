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


std::function<float(float)> setActivationFunction(Activation::ActivationType& function) {
	return Activation::getActivation(function);
}

std::function<float(float)> setDerActivationFunction(Activation::ActivationType& function) {
	return Activation::getDerivative(function);
}