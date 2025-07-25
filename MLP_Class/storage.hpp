#include "MLP.hpp"
#include "Normalizer.hpp"

void SaveModel(const MultiLayerPerceptron& model, const Normalizer& norm, const std::string& filename) {
	// Saving model 
	model.save(filename);

	norm.save(filename);
}

void LoadModel(MultiLayerPerceptron& model, Normalizer& norm, const std::string& filename) {
	model.load(filename);

	norm.load(filename);
}

void SaveModel(const MultiLayerPerceptron& model, const std::string& filename) {
	// Saving model 
	model.save(filename);
}

void LoadModel(MultiLayerPerceptron& model, const std::string& filename) {
	model.load(filename);
}