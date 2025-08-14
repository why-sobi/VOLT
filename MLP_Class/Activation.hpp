#pragma once

#include <cmath>
#include <iostream>
#include <unordered_map>
#include <functional>
#include <Eigen/Dense>

float clip(float x, float lower = -50.0f, float upper = 50.0f) {
    return std::max(lower, std::min(x, upper));
}

namespace Activation {
    enum class ActivationType : uint8_t {
        Sigmoid,
        Tanh,
        ReLU,
        LeakyReLU,
        Linear,
        ELU,
        Softplus,
        Swish,
        Softmax
    };

    // Activation Functions
    inline float sigmoid(float x) {
        x = clip(x);
        return 1.0f / (1.0f + std::exp(-x));
    }

    inline float tanh_act(float x) {
        return std::tanh(x);
    }

    inline float relu(float x) {
        return x > 0 ? x : 0;
    }

    inline float leaky_relu(float x, float alpha = 0.01f) {
        return x > 0 ? x : alpha * x;
    }

    inline float linear(float x) {
        return x;
    }

    inline float elu(float x, float alpha = 1.0f) {
        return x >= 0 ? x : alpha * (std::exp(x) - 1);
    }

    inline float softplus(float x) {
        x = clip(x);
        return std::log1p(std::exp(-std::fabs(x))) + std::max(x, 0.0f);
    }

    inline float swish(float x) {
        return x * sigmoid(x);
    }

    inline Eigen::VectorX<float> softmax(const Eigen::VectorX<float>& input) {
		float max_val = input.maxCoeff();
		Eigen::VectorX<float> stabilized = input.array() - max_val; // for numerical stability
        Eigen::VectorX<float> exp_values = stabilized.array().exp();
        return exp_values / exp_values.sum();
    }

    
    // Derivatives
    inline float d_sigmoid(float x) {
        float s = sigmoid(x);
        return s * (1 - s);
    }

    inline float d_tanh(float x) {
        float t = tanh_act(x);
        return 1 - t * t;
    }

    inline float d_relu(float x) {
        return x > 0.0 ? 1.0f : 0.0f;
    }

    inline float d_leaky_relu(float x, float alpha = 0.01f) {
        return x > 0 ? 1 : alpha;
    }

    inline float d_linear(float x) {
        return 1;
    }

    inline float d_elu(float x, float alpha = 1.0f) {
        return x >= 0 ? 1 : alpha * std::exp(x);
    }

    inline float d_softplus(float x) {
        return sigmoid(x);
    }

    inline float d_swish(float x) {
        float s = sigmoid(x);
        return s + x * s * (1 - s);
    }

	inline Eigen::MatrixXf d_softmax(const Eigen::VectorX<float>& input) { // calculation of the jacobian matrix (aka derivative of softmax)
        size_t size = input.size();
        Eigen::MatrixXf jacobian(size, size);

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if (i == j) {
                    jacobian(i, j) = input(i) * (1 - input(i));
                }
                else {
                    jacobian(i, j) = -input(i) * input(j);
                }
            }
        }
        return jacobian;
	}
    
    inline std::string actTypeToString(ActivationType type) {
        switch (type) {
        case ActivationType::Sigmoid:   return "sigmoid";
        case ActivationType::Tanh:      return "tanh";
        case ActivationType::ReLU:      return "relu";
        case ActivationType::LeakyReLU: return "leaky_relu";
        case ActivationType::Linear:    return "linear";
        case ActivationType::ELU:       return "elu";
        case ActivationType::Softplus:  return "softplus";
        case ActivationType::Swish:     return "swish";
        }
        return "unknown";
    }

    inline ActivationType stringToActivationType(const std::string& name) {
        if (name == "sigmoid")     return ActivationType::Sigmoid;
        if (name == "tanh")        return ActivationType::Tanh;
        if (name == "relu")        return ActivationType::ReLU;
        if (name == "leaky_relu")  return ActivationType::LeakyReLU;
        if (name == "linear")      return ActivationType::Linear;
        if (name == "elu")         return ActivationType::ELU;
        if (name == "softplus")    return ActivationType::Softplus;
        if (name == "swish")       return ActivationType::Swish;
        throw std::invalid_argument("Unknown activation name: " + name);
    }


    inline std::function<float(float)> getActivation(ActivationType type) {
        switch (type) {
        case ActivationType::Sigmoid:   return sigmoid;
        case ActivationType::Tanh:      return tanh_act;
        case ActivationType::ReLU:      return relu;
        case ActivationType::LeakyReLU: return [](float x) { return leaky_relu(x); };
        case ActivationType::Linear:    return linear;
        case ActivationType::ELU:       return [](float x) { return elu(x); };
        case ActivationType::Softplus:  return softplus;
        case ActivationType::Swish:     return swish;
        }
        throw std::invalid_argument("Unknown ActivationType");
    }

    inline std::function<float(float)> getDerivative(ActivationType type) {
        switch (type) {
        case ActivationType::Sigmoid:   return d_sigmoid;
        case ActivationType::Tanh:      return d_tanh;
        case ActivationType::ReLU:      return d_relu;
        case ActivationType::LeakyReLU: return [](float x) { return d_leaky_relu(x); };
        case ActivationType::Linear:    return d_linear;
        case ActivationType::ELU:       return [](float x) { return d_elu(x); };
        case ActivationType::Softplus:  return d_softplus;
        case ActivationType::Swish:     return d_swish;
        }
        throw std::invalid_argument("Unknown ActivationType");
    }

}
