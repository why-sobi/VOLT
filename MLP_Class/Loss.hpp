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

	float MeanSquaredError(const Eigen::MatrixX<float>& predictions, const Eigen::MatrixX<float>& targets) {
		
		Eigen::MatrixX<float> diff = predictions - targets;
		float loss = diff.array().square().sum();																			// Summing the square of each value in the error matrix
		return loss / (predictions.rows() * predictions.cols());															// averaging over the whole batch 
	}

	float HingeLoss(const Eigen::MatrixX<float>& predictions, const Eigen::MatrixX<float>& targets) {
		Eigen::MatrixXf lossMatrix = (1.0f - targets.array() * predictions.array()).cwiseMax(0.0f);							// CWiseMax checks which one is Max element-wise
		return lossMatrix.sum() / (predictions.rows() * predictions.cols());												// avg over all elements
	}

	float BinaryCrossEntropy(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
		Eigen::MatrixXf clipped = predictions.cwiseMax(1e-7f).cwiseMin(1.0f - 1e-7f);										// to avoid log(0)
		Eigen::MatrixXf lossMatrix = -(targets.array() * clipped.array().log() +	
			(1.0f - targets.array()) * (1.0f - clipped.array()).log());														// Main formula
		return lossMatrix.sum() / (predictions.rows() * predictions.cols());												// averaging
	}

	float CategoricalCrossEntropy(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
		Eigen::MatrixXf clipped = predictions.cwiseMax(1e-7f).cwiseMin(1.0f - 1e-7f);										// to avoid log(0)
		Eigen::MatrixXf lossMatrix = -targets.array() * clipped.array().log();												// Compute element-wise: -target * log(prediction)
		return lossMatrix.sum() / predictions.cols();																		// average over batch, not over all outputs
	}

	float CalculateLoss(const Eigen::MatrixX<float>& predictions, const Eigen::MatrixX<float>& targets, Type lossType) {
		if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
			std::cerr << "Predictions and Targets do not match dimensions!\n";
			std::exit(EXIT_FAILURE);
		}

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

	static Eigen::MatrixX<float> CalculateGradient_MSE(
		const Eigen::MatrixX<float>& prediction,
		const Eigen::MatrixX<float>& target
	) {
		return prediction - target;
	}

	static Eigen::MatrixX<float> CalculateGradient_Hinge(
		const Eigen::MatrixX<float>& prediction,
		const Eigen::MatrixX<float>& target
	) {
		// Create a mask for where hinge loss is active: (1 - y * y_hat) > 0
		Eigen::ArrayXXf mask = ((1.0f - target.array() * prediction.array()) > 0.0f).cast<float>();

		// Gradient = -y where mask is active
		Eigen::ArrayXXf grad = -target.array() * mask;

		return grad.matrix(); // Convert back to MatrixXf
	}

	static Eigen::MatrixX<float> CalculateGradient_BCE(
		const Eigen::MatrixX<float>& prediction,
		const Eigen::MatrixX<float>& target
	) {
		return prediction - target;
	}


	static Eigen::MatrixX<float> CalculateGradient_CCE(
		const Eigen::MatrixX<float>& prediction,
		const Eigen::MatrixX<float>& target
	) {
		return prediction - target;
	}

	static Eigen::MatrixX<float> CalculateGradient(
		const Eigen::MatrixX<float>& prediction,
		const Eigen::MatrixX<float>& target,
		Type lossType
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

