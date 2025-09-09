#pragma once

#include <Eigen/Dense>

class Optimizer {
protected:
	
public:
	Optimizer() {}

	virtual void registerLayer(int Idx, const Eigen::MatrixXf& W, const Eigen::VectorXf& B) {}
	virtual void updateWeightsAndBiases(Eigen::MatrixXf& W, Eigen::VectorXf& B,
		const Eigen::MatrixXf& dW, const Eigen::MatrixXf& dB, int idx) = 0;
};

class SGD : public Optimizer {
	float lr;
public:
	SGD(float learning_rate): lr(learning_rate) {}

	void updateWeightsAndBiases(Eigen::MatrixXf& W, Eigen::VectorXf& B, const Eigen::MatrixXf& dW, const Eigen::MatrixXf& dB, int idx) override {
		W -= lr * dW;
		B -= lr * dB;
	}
};

class Momentum : public Optimizer {
	float lr, momentum;
	std::unordered_map<int, Eigen::MatrixXf> vW, vB;
public:
	Momentum(float learning_rate, float momentum) : lr(learning_rate), momentum(momentum) {}

	void registerLayer(int idx, const Eigen::MatrixXf& W, const Eigen::VectorXf& B) override {
		vW[idx] = Eigen::MatrixXf::Zero(W.rows(), W.cols());
		vB[idx] = Eigen::MatrixXf::Zero(B.rows(), B.cols());
	}

	void updateWeightsAndBiases(Eigen::MatrixXf& W, Eigen::VectorXf& B, const Eigen::MatrixXf& dW, const Eigen::MatrixXf& dB, int idx) override {
		vW[idx] = momentum * vW[idx] - lr * dW;
		vB[idx] = momentum * vB[idx] - lr * dB;
		W += vW[idx];
		B += vB[idx];
	}
};

class Adam : public Optimizer {
	float lr, beta1, beta2, eps;
	int t;
	std::unordered_map<int, Eigen::MatrixXf> mW, vW, mB, vB;

public:
	Adam(float lr, float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8)
		: lr(lr), beta1(beta1), beta2(beta2), eps(eps), t(0) {}


	void registerLayer(int idx, const Eigen::MatrixXf& W, const Eigen::VectorXf& B) override {
		mW[idx] = Eigen::MatrixXf::Zero(W.rows(), W.cols());
		vW[idx] = Eigen::MatrixXf::Zero(W.rows(), W.cols());
		mB[idx] = Eigen::MatrixXf::Zero(B.rows(), B.cols());
		vB[idx] = Eigen::MatrixXf::Zero(B.rows(), B.cols());
	}
	void updateWeightsAndBiases(Eigen::MatrixXf& W, Eigen::VectorXf& B, const Eigen::MatrixXf& dW, const Eigen::MatrixXf& dB, int idx) override {
		t++;

		mW[idx] = beta1 * mW[idx] + (1 - beta1) * dW;
		vW[idx] = beta2 * vW[idx] + (1 - beta2) * dW.cwiseProduct(dW);
		mB[idx] = beta1 * mB[idx] + (1 - beta1) * dB;
		vB[idx] = beta2 * vB[idx] + (1 - beta2) * dB.cwiseProduct(dB);

		Eigen::MatrixXf mW_hat = mW[idx] / (1 - pow(beta1, t));
		Eigen::MatrixXf vW_hat = vW[idx] / (1 - pow(beta2, t));
		Eigen::MatrixXf mB_hat = mB[idx] / (1 - pow(beta1, t));
		Eigen::MatrixXf vB_hat = vB[idx] / (1 - pow(beta2, t));

		W -= lr * mW_hat.cwiseQuotient((vW_hat.array().sqrt() + eps).matrix());
		B -= lr * mB_hat.cwiseQuotient((vB_hat.array().sqrt() + eps).matrix());
	}
};

class RMSprop : public Optimizer {
	float lr, beta, eps;
	std::unordered_map<int, Eigen::MatrixXf> sW, sB;

public:
	RMSprop(float lr, float beta = 0.9, float eps = 1e-8) : lr(lr), beta(beta), eps(eps) {}

	void registerLayer(int idx, const Eigen::MatrixXf& W, const Eigen::VectorXf& B) override {
		sW[idx] = Eigen::MatrixXf::Zero(W.rows(), W.cols());
		sB[idx] = Eigen::MatrixXf::Zero(B.rows(), B.cols());
	}
	void updateWeightsAndBiases(Eigen::MatrixXf& W, Eigen::VectorXf& B, const Eigen::MatrixXf& dW, const Eigen::MatrixXf& dB, int idx) override {
		sW[idx] = beta * sW[idx] + (1 - beta) * dW.cwiseProduct(dW);
		sB[idx] = beta * sB[idx] + (1 - beta) * dB.cwiseProduct(dB);

		W -= lr * dW.cwiseQuotient((sW[idx].array().sqrt() + eps).matrix());
		B -= lr * dB.cwiseQuotient((sB[idx].array().sqrt() + eps).matrix());
	}
};