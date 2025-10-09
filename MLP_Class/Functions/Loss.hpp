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

	float MeanSquaredError(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
		
		Eigen::MatrixXf diff = predictions - targets;
		float loss = diff.array().square().sum();																			// Summing the square of each value in the error matrix
		return loss / predictions.cols();																					// averaging over the batch size 
	}

	float HingeLoss(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
		Eigen::MatrixXf lossMatrix = (1.0f - targets.array() * predictions.array()).cwiseMax(0.0f);							// CWiseMax checks which one is Max element-wise
		return lossMatrix.sum() / predictions.cols();																		// averaging over the batch size 
	}

	float BinaryCrossEntropy(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
		Eigen::MatrixXf clipped = predictions.cwiseMax(1e-7f).cwiseMin(1.0f - 1e-7f);										// to avoid log(0)
		Eigen::MatrixXf lossMatrix = -(targets.array() * clipped.array().log() +	
			(1.0f - targets.array()) * (1.0f - clipped.array()).log());														// Main formula
		return lossMatrix.sum() / predictions.cols();																		// averaging over the batch size
	}

	float CategoricalCrossEntropy(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
		Eigen::MatrixXf clipped = predictions.cwiseMax(1e-7f).cwiseMin(1.0f - 1e-7f);										// to avoid log(0)
		Eigen::MatrixXf lossMatrix = -targets.array() * clipped.array().log();												// Compute element-wise: -target * log(prediction)
		return lossMatrix.sum() / predictions.cols();																		// averaging over the batch size
	}

	float CalculateLoss(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets, Type lossType) {
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

	static Eigen::MatrixXf CalculateGradient_MSE(
		const Eigen::MatrixXf& prediction,
		const Eigen::MatrixXf& target
	) {
		return 2.0f * (prediction - target);																				// Gradient of MSE
	}

	static Eigen::MatrixXf CalculateGradient_Hinge(
		const Eigen::MatrixXf& prediction,
		const Eigen::MatrixXf& target
	) {
		// Create a mask for where hinge loss is active: (1 - y * y_hat) > 0
		Eigen::ArrayXXf mask = ((1.0f - target.array() * prediction.array()) > 0.0f).cast<float>();

		// Gradient = -y where mask is active
		Eigen::ArrayXXf grad = -target.array() * mask;

		return grad.matrix(); // Convert back to MatrixXf
	}

	static Eigen::MatrixXf CalculateGradient_BCE(
		const Eigen::MatrixXf& prediction,
		const Eigen::MatrixXf& target
	) {
		return -(target.array() / prediction.array() + (1.0f - target.array()) / (1.0f - prediction.array())).matrix();
	}


	static Eigen::MatrixXf CalculateGradient_CCE(
		const Eigen::MatrixXf& prediction, 
		const Eigen::MatrixXf& target
	) {
		return prediction - target;
	}

	static Eigen::MatrixXf CalculateGradient(
		const Eigen::MatrixXf& prediction,																			// assuming already activated output
		const Eigen::MatrixXf& target,
		Activation::ActivationType last_activation,																			// used to determine the derivative of the activation function
		Type lossType
	) {
		if (prediction.rows() != target.rows() || prediction.cols() != target.cols()) {
			std::cerr << "Predictions and Targets do not match dimensions!\n";
			std::exit(EXIT_FAILURE);
		}

		if (last_activation == Activation::ActivationType::Sigmoid) {
			if (lossType == Type::BinaryCrossEntropy) {
				return prediction - target;
			}
		}

		if (last_activation == Activation::ActivationType::Linear) {
			if (lossType == Type::BinaryCrossEntropy) {
				return prediction.unaryExpr(Activation::getActivation(Activation::ActivationType::Sigmoid)) - target;
			}
			else if (lossType == Type::CategoricalCrossEntropy) {
				Eigen::MatrixXf softmax_pred = prediction;
				for (int i = 0; i < prediction.cols(); ++i) {
					softmax_pred.col(i) = Activation::softmax(prediction.col(i)); // Apply softmax to each column
				}
				return softmax_pred - target; // Gradient for CCE with softmax
			}
		}

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

