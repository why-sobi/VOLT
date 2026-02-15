#pragma once

#include <Eigen/Dense>

enum class Regularization {
	L1,
	L2,
	ElasticNet,
	None
};

float regularizeLoss(float loss, int batch_size, float lambda, Regularization type) {
	switch (type) {
	case Regularization::L1:
		loss += (lambda / batch_size);
		return loss;
	case Regularization::L2:
		loss += (lambda / batch_size);
		return loss;
	case Regularization::ElasticNet:
		loss += (lambda / batch_size);
		return loss;
	case Regularization::None:
		return loss;
	default:
		throw std::invalid_argument("Invalid Regularization Type!\n");
	}
}

inline void regularizeGradient(Eigen::MatrixX<float>& dW, Eigen::MatrixX<float>& weights, int batch_size, float lambda, Regularization type) {
	switch (type) {
	case Regularization::L1:
		dW += (lambda / batch_size) * weights.array().sign().matrix();
		break;
	case Regularization::L2:
		dW += (lambda / batch_size) * weights;
		break;

	case Regularization::ElasticNet: {
			dW.array() += (lambda / batch_size) * (weights.array().sign() + weights.array());
			break;
	}
	case Regularization::None:
		break;
	default:
		throw std::invalid_argument("Invalid Regularization Type!\n");
	}
}

