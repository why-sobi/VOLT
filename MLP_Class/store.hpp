#pragma once

#include "MLP.hpp"
#include "Normalizer.hpp"

void SaveModel(const MultiLayerPerceptron& model, const Normalizer& norm, const std::string& filename) {
	// Save the model
	model.save(filename);

	// Save the normalizer
	norm.save(filename);
}


void LoadModel(MultiLayerPerceptron& model, Normalizer& norm, const std::string& filename) {
	// Save the model
	model.load(filename);

	// Save the normalizer
	norm.load(filename);
}

void SaveModel(const MultiLayerPerceptron& model, const std::string& filename) {
	// Save the model
	model.save(filename);

}


void LoadModel(MultiLayerPerceptron& model, const std::string& filename) {
	// Save the model
	model.load(filename);

}