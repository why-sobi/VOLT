#pragma once

#include <Eigen/Dense>

class Optimizer {
protected:
	
public:
	Optimizer() {}

	virtual void registerLayer(int Idx, const Eigen::MatrixXf& W, const Eigen::VectorXf& B) {}
	virtual void updateWeightsAndBiases(Eigen::MatrixXf& W, Eigen::VectorXf& B,
		const Eigen::MatrixXf& dW, const Eigen::VectorXf& dB, int idx) = 0;
};

class SGD : public Optimizer {
	float lr;
public:
	SGD(float learning_rate): lr(learning_rate) {}

	void updateWeightsAndBiases(Eigen::MatrixXf& W, Eigen::VectorXf& B, const Eigen::MatrixXf& dW, const Eigen::VectorXf& dB, int idx) override {
		W -= lr * dW;
		B -= lr * dB;
	}
};

class Momentum : public Optimizer {
	float lr, momentum;
	std::vector<Eigen::MatrixXf> vW;
    std::vector<Eigen::VectorXf> vB;
public:
	Momentum(float learning_rate, float momentum=0.9f) : lr(learning_rate), momentum(momentum) {}

	void registerLayer(int idx, const Eigen::MatrixXf& W, const Eigen::VectorXf& B) override {
        if (idx >= vW.size()) {
            vW.resize(idx + 1);
            vB.resize(idx + 1);
        }
		vW[idx] = Eigen::MatrixXf::Zero(W.rows(), W.cols());
		vB[idx] = Eigen::VectorXf::Zero(B.rows(), B.cols());
	}

	void updateWeightsAndBiases(Eigen::MatrixXf& W, Eigen::VectorXf& B, const Eigen::MatrixXf& dW, const Eigen::VectorXf& dB, int idx) override {
		vW[idx] = momentum * vW[idx] - lr * dW;
		vB[idx] = momentum * vB[idx] - lr * dB;
		W += vW[idx];
		B += vB[idx];
	}
};

class Adam : public Optimizer {
    float lr, beta1, beta2, eps;
    std::vector<int> t;
    std::vector<Eigen::MatrixXf> mW, vW;
    std::vector<Eigen::VectorXf> mB, vB;

public:
    Adam(float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
        : lr(lr), beta1(beta1), beta2(beta2), eps(eps) {
    }

    void registerLayer(int idx, const Eigen::MatrixXf& W, const Eigen::VectorXf& B) override {
        if (idx >= t.size()) {
            t.resize(idx + 1);
            mW.resize(idx + 1); vW.resize(idx + 1);
            mB.resize(idx + 1); vB.resize(idx + 1);
        }
        t[idx] = 0;
        mW[idx] = Eigen::MatrixXf::Zero(W.rows(), W.cols());
        vW[idx] = Eigen::MatrixXf::Zero(W.rows(), W.cols());
        mB[idx] = Eigen::VectorXf::Zero(B.size());
        vB[idx] = Eigen::VectorXf::Zero(B.size());
    }

    void updateWeightsAndBiases(Eigen::MatrixXf& W, Eigen::VectorXf& B,
        const Eigen::MatrixXf& dW, const Eigen::VectorXf& dB, int idx) override {
        int tt = ++t[idx];

        // Moment updates
        mW[idx].array() = beta1 * mW[idx].array() + (1 - beta1) * dW.array();
        vW[idx].array() = beta2 * vW[idx].array() + (1 - beta2) * dW.array().square();

        mB[idx].array() = beta1 * mB[idx].array() + (1 - beta1) * dB.array();
        vB[idx].array() = beta2 * vB[idx].array() + (1 - beta2) * dB.array().square();

        // Precompute bias-corrected scalars
        float bc1 = 1.0f / (1.0f - std::pow(beta1, tt));
        float bc2 = 1.0f / (1.0f - std::pow(beta2, tt));

        // Update weights in-place
        W.array() -= lr * (mW[idx].array() * bc1) / (vW[idx].array() * bc2).sqrt() + eps;
        B.array() -= lr * (mB[idx].array() * bc1) / (vB[idx].array() * bc2).sqrt() + eps;
    }
};


class RMSprop : public Optimizer {
    float lr, beta, eps;
    std::vector<Eigen::MatrixXf> sW;
    std::vector<Eigen::VectorXf> sB;

public:
    RMSprop(float lr, float beta = 0.9f, float eps = 1e-8f)
        : lr(lr), beta(beta), eps(eps) {
    }

    void registerLayer(int idx, const Eigen::MatrixXf& W, const Eigen::VectorXf& B) override {
        if (idx >= sW.size()) {
            sW.resize(idx + 1);
            sB.resize(idx + 1);
        }
        sW[idx] = Eigen::MatrixXf::Zero(W.rows(), W.cols());
        sB[idx] = Eigen::VectorXf::Zero(B.size());
    }

    void updateWeightsAndBiases(Eigen::MatrixXf& W, Eigen::VectorXf& B,
        const Eigen::MatrixXf& dW, const Eigen::VectorXf& dB, int idx) override {
        sW[idx].array() = beta * sW[idx].array() + (1 - beta) * dW.array().square();
        sB[idx].array() = beta * sB[idx].array() + (1 - beta) * dB.array().square();

        W.array() -= lr * dW.array() / (sW[idx].array().sqrt() + eps);
        B.array() -= lr * dB.array() / (sB[idx].array().sqrt() + eps);
    }
};
