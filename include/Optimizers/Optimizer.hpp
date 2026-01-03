#pragma once

#include <Eigen/Dense>
#include "../Utility/utils.hpp"

enum class OptimizerType : uint8_t {
    SGD,
    Momentum,
    Adam,
    RMSProp
};

class SGD;
class Momentum;
class Adam;
class RMSprop;


class Optimizer {
protected:
    OptimizerType opType;

    
    // protected constructor because no need for just optimizer object
    // can't allow Optimizer* = new Optimizer either
    Optimizer(OptimizerType OpType): opType(OpType) {} 
    
public:
	virtual void registerLayer(int Idx, const Eigen::MatrixXf& W, const Eigen::VectorXf& B) {}
	virtual void updateWeightsAndBiases(Eigen::MatrixXf& W, Eigen::VectorXf& B,
		const Eigen::MatrixXf& dW, const Eigen::VectorXf& dB, int idx) = 0;
    virtual void saveOptimizer(std::fstream& file) = 0;
    virtual void loadOptimizer(std::fstream& file) = 0; 

    static Optimizer* loadFromFile(std::fstream& file);
};

class SGD : public Optimizer {
	float lr;
public:
	SGD(float learning_rate): Optimizer(OptimizerType::SGD), lr(learning_rate) {}

	void updateWeightsAndBiases(Eigen::MatrixXf& W, Eigen::VectorXf& B, const Eigen::MatrixXf& dW, const Eigen::VectorXf& dB, int idx) override {
		W -= lr * dW;
		B -= lr * dB;
	}

    void saveOptimizer(std::fstream& file) override {
        io::writeEnum<OptimizerType>(file, this->opType);
        io::writeNumeric<float>(file, this->lr);
    }

    void loadOptimizer(std::fstream& file) override {
        this->lr = io::readNumeric<float>(file);
    }
};

class Momentum : public Optimizer {
	float lr, momentum;
	std::vector<Eigen::MatrixXf> vW;
    std::vector<Eigen::VectorXf> vB;
public:
	Momentum(float learning_rate, float momentum=0.9f) : Optimizer(OptimizerType::Momentum), lr(learning_rate), momentum(momentum) {}

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

    void saveOptimizer(std::fstream& file) override {
        io::writeEnum<OptimizerType>(file, this->opType);

        io::writeNumeric<float>(file, this->lr);
        io::writeNumeric<float>(file, this->momentum);
        io::writeEigenMatVector<float>(file, this->vW);
        io::writeEigenVecVector<float>(file, this->vB);
    }

    void loadOptimizer(std::fstream& file) override {
        this->lr = io::readNumeric<float>(file);
        this->momentum = io::readNumeric<float>(file);
        this->vW = io::readEigenMatVector<float>(file);
        this->vB = io::readEigenVecVector<float>(file);
    }
};

class Adam : public Optimizer {
    float lr, beta1, beta2, eps;
    std::vector<int> t;
    std::vector<Eigen::MatrixXf> mW, vW;
    std::vector<Eigen::VectorXf> mB, vB;

public:
    Adam(float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
        : Optimizer(OptimizerType::Adam), lr(lr), beta1(beta1), beta2(beta2), eps(eps) {}

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
        float bc1 = 1.0f / (1.0f - static_cast<float>(std::pow(beta1, tt)));
        float bc2 = 1.0f / (1.0f - static_cast<float>(std::pow(beta2, tt)));

        // Update weights in-place
        W.array() -= lr * (mW[idx].array() * bc1) / (vW[idx].array() * bc2).sqrt() + eps;
        B.array() -= lr * (mB[idx].array() * bc1) / (vB[idx].array() * bc2).sqrt() + eps;
    }

    void saveOptimizer(std::fstream& file) override {
        io::writeEnum<OptimizerType>(file, this->opType);

        io::writeNumeric<float>(file, this->lr);
        io::writeNumeric<float>(file, this->beta1);
        io::writeNumeric<float>(file, this->beta2);
        io::writeNumeric<float>(file, this->eps);
        
        io::writeEigenMatVector<float>(file, this->mW);
        io::writeEigenMatVector<float>(file, this->vW);

        io::writeEigenVecVector<float>(file, this->mB);
        io::writeEigenVecVector<float>(file, this->vB);
    }

    void loadOptimizer(std::fstream& file) override {
        this->lr = io::readNumeric<float>(file);
        this->beta1 = io::readNumeric<float>(file);
        this->beta2 = io::readNumeric<float>(file);
        this->eps = io::readNumeric<float>(file);

        this->mW = io::readEigenMatVector<float>(file);
        this->vW = io::readEigenMatVector<float>(file);
        
        this->mB = io::readEigenVecVector<float>(file);
        this->vB = io::readEigenVecVector<float>(file);
    }
};


class RMSprop : public Optimizer {
    float lr, beta, eps;
    std::vector<Eigen::MatrixXf> sW;
    std::vector<Eigen::VectorXf> sB;

public:
    RMSprop(float lr, float beta = 0.9f, float eps = 1e-8f)
        : Optimizer(OptimizerType::RMSProp), lr(lr), beta(beta), eps(eps) {}

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

    void saveOptimizer(std::fstream& file) override {
        io::writeEnum<OptimizerType>(file, this->opType);

        io::writeNumeric<float>(file, this->lr);
        io::writeNumeric<float>(file, this->beta);
        io::writeNumeric<float>(file, this->eps);

        io::writeEigenMatVector<float>(file, sW);
        io::writeEigenVecVector<float>(file, sB);
    }

    void loadOptimizer(std::fstream& file) override {
        this->lr = io::readNumeric<float>(file);
        this->beta = io::readNumeric<float>(file);
        this->eps = io::readNumeric<float>(file);

        this->sW = io::readEigenMatVector<float>(file);
        this->sB = io::readEigenVecVector<float>(file);
    }
};


Optimizer* Optimizer::loadFromFile(std::fstream& file) {
    Optimizer* optimizer = nullptr;
    OptimizerType opType = io::readEnum<OptimizerType>(file);
    switch (opType) {
    case OptimizerType::SGD:
        optimizer = new SGD(0.001f); // temp value will be overwritten anyway
        break;
    case OptimizerType::Momentum:
        optimizer = new Momentum(0.001f);
        break;
    case OptimizerType::Adam:
        optimizer = new Adam(0.001f);
        break;
    case OptimizerType::RMSProp:
        optimizer = new RMSprop(0.001f);
        break;
    default:
        throw std::runtime_error("Could not determine type of optimizer!\n");
    }
    optimizer->loadOptimizer(file);
    optimizer->opType = opType;
    return optimizer;
}