#pragma once

#include <cmath>
#include <iostream>
#include <unordered_map>
#include <functional>

namespace Activation {

    // Activation Functions
    inline float sigmoid(float x) {
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
        return std::log(1 + std::exp(x));
    }

    inline float swish(float x) {
        return x * sigmoid(x);
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
        return x > 0 ? 1 : 0;
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

    // Maps (Optional)
    const std::unordered_map<std::string, std::function<float(float)>> activations = {
    {"sigmoid", sigmoid},
    {"tanh", tanh_act},
    {"relu", relu},
    {"leaky_relu", [](float x) { return leaky_relu(x); }},
    {"linear", linear},
    {"elu", [](float x) { return elu(x); }},
    {"softplus", softplus},
    {"swish", swish},
    };

    const std::unordered_map<std::string, std::function<float(float)>> derivatives = {
        {"sigmoid", d_sigmoid},
        {"tanh", d_tanh},
        {"relu", d_relu},
        {"leaky_relu", [](float x) { return d_leaky_relu(x); }},
        {"linear", d_linear},
        {"elu", [](float x) { return d_elu(x); }},
        {"softplus", d_softplus},
        {"swish", d_swish},
    };


}
