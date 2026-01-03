#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <random>

#include "../Functions/Activation.hpp"


namespace Init {
	enum class type {
		He,
		Xavier,
        LeCun,
		Orthogonal,
	};

    type setupType(Activation::ActivationType act) {
        switch (act) {
        case Activation::ActivationType::ReLU:
        case Activation::ActivationType::LeakyReLU:
        case Activation::ActivationType::ELU:
            return type::He;

        case Activation::ActivationType::Sigmoid:
        case Activation::ActivationType::Tanh:
        case Activation::ActivationType::Softmax:
            return type::Xavier;

        case Activation::ActivationType::SeLU:
            return type::LeCun;

        case Activation::ActivationType::Linear:
            return type::Orthogonal;

        default:
            std::cerr << "Warning: Unknown activation type � defaulting to Orthogonal.\n";
            return type::Orthogonal;
        }
    }

    void heInit(Eigen::MatrixX<float>& W, int fan_in) {
        float stddev = std::sqrt(2.0f / fan_in);
        std::normal_distribution<float> dist(0.0f, stddev);
        std::mt19937 gen(std::random_device{}());
        for (int i = 0; i < W.rows(); ++i)
            for (int j = 0; j < W.cols(); ++j)
                W(i, j) = dist(gen);
    }

    void xavierInit(Eigen::MatrixX<float>& W, int fan_in, int fan_out) {
        float limit = std::sqrt(6.0f / (fan_in + fan_out));
        std::uniform_real_distribution<float> dist(-limit, limit);
        std::mt19937 gen(std::random_device{}());
        for (int i = 0; i < W.rows(); ++i)
            for (int j = 0; j < W.cols(); ++j)
                W(i, j) = dist(gen);
    }

    void lecunInit(Eigen::MatrixX<float>& W, int fan_in) {
        float stddev = std::sqrt(1.0f / fan_in);
        std::normal_distribution<float> dist(0.0f, stddev);
        std::mt19937 gen(std::random_device{}());
        for (int i = 0; i < W.rows(); ++i)
            for (int j = 0; j < W.cols(); ++j)
                W(i, j) = dist(gen);
    }


    void orthogonalInit(Eigen::MatrixX<float>& W) {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::mt19937 gen(std::random_device{}());

        Eigen::MatrixX<float> A = W.unaryExpr([&](float) { return dist(gen); });

        // QR decomposition
        Eigen::HouseholderQR<Eigen::MatrixX<float>> qr(A);
        Eigen::MatrixX<float> Q = qr.householderQ();
        Eigen::MatrixX<float> R = qr.matrixQR().triangularView<Eigen::Upper>();

        // Normalize sign to ensure deterministic orientation
        Eigen::VectorXf d = R.diagonal().array().sign();
        Q = Q * d.asDiagonal();

        W = Q.leftCols(W.cols());
    }


    void InitWeightsAndBias(Eigen::MatrixX<float>& W, Eigen::VectorX<float>& b, type init_type) {
        int fan_in = static_cast<int>(W.cols()), fan_out = static_cast<int>(W.rows());
        switch (init_type) {
        case type::He: 
            heInit(W, fan_in);
            break;
        case type::Xavier: 
            xavierInit(W, fan_in, fan_out);
            break;
        case type::LeCun:
            lecunInit(W, fan_in);
            break;
        case type::Orthogonal: 
            orthogonalInit(W);
            break;
        default:
            std::cerr << "Unknown Weight Initializer!\n";
        }

        if (init_type == type::He) b.setConstant(0.01f);
        else                       b.setZero();
    }
};
