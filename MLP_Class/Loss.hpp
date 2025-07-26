# pragma once

#include <iostream>
#include <vector>


namespace Loss {
	enum class Type {
		MSE,
		HingeLoss,
		BinaryCrossEntropy,
		CategoricalCrossEntropy
	};

	float MeanSquaredError(const std::vector<float>& predictions, const std::vector<float>& targets) {
		if (predictions.size() != targets.size()) {
			throw std::invalid_argument("Predictions and targets must have the same size.");
		}
		float sum = 0.0f;
		for (size_t i = 0; i < predictions.size(); ++i) {
			float diff = predictions[i] - targets[i];
			sum += diff * diff;
		}
		return sum / predictions.size();
	}

	float HingeLoss(const std::vector<float>& predictions, const std::vector<float>& targets) {
		if (predictions.size() != targets.size()) {
			throw std::invalid_argument("Predictions and targets must have the same size.");
		}
		float loss = 0.0f;
		for (size_t i = 0; i < predictions.size(); ++i) {
			loss += std::max(0.0f, 1 - targets[i] * predictions[i]);
		}
		return loss / predictions.size();
	}

	float BinaryCrossEntropy(const std::vector<float>& predictions, const std::vector<float>& targets) {
		if (predictions.size() != targets.size()) {
			throw std::invalid_argument("Predictions and targets must have the same size.");
		}
		float loss = 0.0f;
		for (size_t i = 0; i < predictions.size(); ++i) {
			if (targets[i] == 1) {
				loss -= std::log(std::max(predictions[i], 1e-7f)); // to avoid log(0<=)
			}
			else {
				loss -= std::log(std::max(1 - predictions[i], 1e-7f));
			}
		}
		return loss / predictions.size();
	}

	float CategoricalCrossEntropy(const std::vector<float>& predictions, const std::vector<float>& targets) {
		if (predictions.size() != targets.size()) {
			throw std::invalid_argument("Predictions and targets must have the same size.");
		}
		float loss = 0.0f;
		for (size_t i = 0; i < predictions.size(); ++i) {
			loss -= targets[i] * std::log(predictions[i]);
		}
		return loss / predictions.size();
	}

	float CalculateLoss(Type lossType, const std::vector<float>& predictions, const std::vector<float>& targets) {
		switch (lossType) {
			case Type::MSE:
				return MeanSquaredError(predictions, targets);
			case Type::HingeLoss:
				return HingeLoss(predictions, targets);
			case Type::BinaryCrossEntropy:
				return BinaryCrossEntropy(predictions, targets);
			case Type::CategoricalCrossEntropy:
				return CategoricalCrossEntropy(predictions, targets);
			default:
				throw std::invalid_argument("Unknown loss type.");
		}
	}

	static std::vector<float> CalculateGradient_MSE(
		const std::vector<float>& prediction,
		const std::vector<float>& target
	) {
		std::vector<float> grad(prediction.size());
		for (size_t i = 0; i < prediction.size(); ++i)
			grad[i] = prediction[i] - target[i];  // dL/dŷ = ŷ - y
		return grad;
	}

	static std::vector<float> CalculateGradient_Hinge(
		const std::vector<float>& prediction,
		const std::vector<float>& target
	) {
		std::vector<float> grad(prediction.size());
		for (size_t i = 0; i < prediction.size(); ++i) {
			float y = target[i];         // should be -1 or +1
			float y_hat = prediction[i];
			grad[i] = (1 - y * y_hat > 0) ? -y : 0.0f;
		}
		return grad;
	}

	static std::vector<float> CalculateGradient_BCE(
		const std::vector<float>& prediction,
		const std::vector<float>& target
	) {
		std::vector<float> grad(prediction.size());
		for (size_t i = 0; i < prediction.size(); ++i)
			grad[i] = prediction[i] - target[i];  // same trick as MSE
		return grad;
	}


	static std::vector<float> CalculateGradient_CCE(
		const std::vector<float>& prediction,
		const std::vector<float>& target
	) {
		std::vector<float> grad(prediction.size());
		for (size_t i = 0; i < prediction.size(); ++i)
			grad[i] = prediction[i] - target[i];  // ŷ - y
		return grad;
	}

	static std::vector<float> CalculateGradient(
		Type lossType,
		const std::vector<float>& prediction,
		const std::vector<float>& target
	) {
		switch (lossType) {
		case Type::MSE:
			return CalculateGradient_MSE(prediction, target);
		case Type::HingeLoss:
			return CalculateGradient_Hinge(prediction, target);
		case Type::BinaryCrossEntropy:
			return CalculateGradient_BCE(prediction, target);
		case Type::CategoricalCrossEntropy:
			return CalculateGradient_CCE(prediction, target);
		default:
			throw std::invalid_argument("Unknown loss type.");
		}
	}
};

