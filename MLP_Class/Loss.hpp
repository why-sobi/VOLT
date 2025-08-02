# pragma once

#include <iostream>
#include <vector>

#include <Eigen/Dense>


namespace Loss {
	enum class Type {
		MSE,
		HingeLoss,
		BinaryCrossEntropy,
		CategoricalCrossEntropy
	};

	float MeanSquaredError(const Eigen::VectorX<float>& predictions, const Eigen::VectorX<float>& targets) {
		if (predictions.size() != targets.size()) {
			throw std::invalid_argument("Predictions and targets must have the same size.");
		}
		Eigen::VectorX<float> diff = predictions - targets;
		float loss = diff.squaredNorm(); // sum of squares

		return loss / predictions.size(); // average
	}

	float HingeLoss(const Eigen::VectorX<float>& predictions, const Eigen::VectorX<float>& targets) {
		if (predictions.size() != targets.size()) {
			throw std::invalid_argument("Predictions and targets must have the same size.");
		}
		// Calculate the loss using the hinge loss formula
		Eigen::VectorXf lossVector = (1 - targets.array() * predictions.array()).cwiseMax(0.0f);							// CWiseMax checks which one is Max element-wise

		// Calculate the mean loss
		float loss = lossVector.sum() / predictions.size();

		return loss;
	}

	float BinaryCrossEntropy(const Eigen::VectorX<float>& predictions, const Eigen::VectorX<float>& targets) {
		if (predictions.size() != targets.size()) {
			throw std::invalid_argument("Predictions and targets must have the same size.");
		}
		// Calculate the loss using the binary cross-entropy formula
		Eigen::VectorXf lossVector = -(targets.array() * predictions.array().log() +
			(1 - targets.array()) * (1 - predictions.array()).log());

		// To avoid log(0), we clamp the predictions
		lossVector = lossVector.unaryExpr([](float val) { return std::max(val, 1e-7f); });

		// Calculate the mean loss
		float loss = lossVector.sum() / predictions.size();

		return loss;
	}

	//float CategoricalCrossEntropy(const Eigen::VectorX<float>& predictions, const Eigen::VectorX<float>& targets) {
	//	if (predictions.size() != targets.size()) {
	//		throw std::invalid_argument("Predictions and targets must have the same size.");
	//	}
	//	//std::cout << predictions << " : " << targets << std::endl;
	//	float loss = 0.0f;
	//	for (size_t i = 0; i < predictions.size(); ++i) {
	//		loss -= targets[i] * std::log(predictions[i] + 1e-7);
	//	}
	//	return loss / predictions.size();
	//}

	float CategoricalCrossEntropy(const Eigen::VectorX<float>& predictions, const Eigen::VectorX<float>& targets) {
		if (predictions.size() != targets.size()) {
			throw std::invalid_argument("Predictions and targets must have the same size.");
		}
		// Clamp predictions to avoid log(0)
		Eigen::VectorX<float> clipped = predictions.cwiseMax(1e-7f).cwiseMin(1.0f - 1e-7f);
		// Compute element-wise: -target * log(prediction)
		Eigen::VectorX<float> lossVec = -targets.array() * clipped.array().log();
		// Sum and average
		float loss = lossVec.mean();
		return loss; 
	}

	float CalculateLoss(Type lossType, const Eigen::VectorX<float>& predictions, const Eigen::VectorX<float>& targets) {
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

	static Eigen::VectorX<float> CalculateGradient_MSE(
		const Eigen::VectorX<float>& prediction,
		const Eigen::VectorX<float>& target
	) {
		Eigen::VectorX<float> grad = prediction - target;
		return grad;
	}

	static Eigen::VectorX<float> CalculateGradient_Hinge(
		const Eigen::VectorX<float>& prediction,
		const Eigen::VectorX<float>& target
	) {
		// Compute mask: 1 if (1 - y * y_hat) > 0, else 0
		Eigen::ArrayXf mask = ((1.0f - target.array() * prediction.array()) > 0.0f).cast<float>();
		// Gradient: -y where mask is 1, else 0
		Eigen::VectorXf grad = -target.array() * mask;
		return grad;
	}

	static Eigen::VectorX<float> CalculateGradient_BCE(
		const Eigen::VectorX<float>& prediction,
		const Eigen::VectorX<float>& target
	) {
		Eigen::VectorX<float> grad = prediction - target;
		return grad;
	}


	static Eigen::VectorX<float> CalculateGradient_CCE(
		const Eigen::VectorX<float>& prediction,
		const Eigen::VectorX<float>& target
	) {
		Eigen::VectorX<float> grad = prediction - target;
		return grad;
	}

	static Eigen::VectorX<float> CalculateGradient(
		Type lossType,
		const Eigen::VectorX<float>& prediction,
		const Eigen::VectorX<float>& target
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

