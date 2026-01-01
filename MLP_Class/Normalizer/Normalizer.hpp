#pragma once

#include <iostream>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <fstream>
#include <Eigen/Dense>
#include "../Data/DataUtil.hpp"
#include "../Utility/utils.hpp"

enum class NormalizeType { None, MinMax, ZScore };

class Normalizer {
private:
    struct Stats {
        float min = 0.0f;
        float max = 0.0f;
        float mean = 0.0f;
        float std_dev = 0.0f;
        NormalizeType type = NormalizeType::None;

        Stats() = default;
        Stats(float min, float max, float mean, float std_dev, NormalizeType type)
            : min(min), max(max), mean(mean), std_dev(std_dev), type(type) {
        }
    };

    std::unordered_map<std::string, Stats> stats;   // column index → stats

    // === helpers ===
    static float safeStdDev(const Eigen::Ref<const Eigen::VectorXf, 0, Eigen::InnerStride<>>& col, float mean) {
        if (col.size() == 0) return 1e-8f;
        float sq_sum = (col.array() - mean).square().sum();
        float std_dev = std::sqrt(sq_sum / col.size());
        return (std_dev == 0) ? 1e-8f : std_dev;
    }

    void minmax(const std::string& colRef, Eigen::Ref<Eigen::VectorXf, 0, Eigen::InnerStride<>> col) {
        const Stats& s = stats[colRef];
        
        if ((s.max - s.min) == 0) { // this is because this col was ignored in training hence should be ignored in test
            col.setZero();
            return;
        }

        float range = (s.max - s.min);
        col = (col.array() - s.min) / range;

        col = col.array().min(1.0f).max(0.0f); // another check j in case so everything [0, 1]

    }

    void zscore(const std::string& colRef, Eigen::Ref<Eigen::VectorXf, 0, Eigen::InnerStride<>> col) {
        const Stats& s = stats[colRef];
        float denom = (s.std_dev == 0) ? 1e-8f : s.std_dev;
        col = (col.array() - s.mean) / denom;
    }

    void reverse_minmax(const std::string& colRef, Eigen::Ref<Eigen::VectorXf, 0, Eigen::InnerStride<>> col) {
        const Stats& s = stats[colRef];
        float range = (s.max - s.min == 0) ? 1e-8f : (s.max - s.min);
        col = col.array() * range + s.min;
    }

    void reverse_zscore(const std::string& colRef, Eigen::Ref<Eigen::VectorXf, 0, Eigen::InnerStride<>> col) {
        const Stats& s = stats[colRef];
        col = col.array() * s.std_dev + s.mean;
    }

public:
    void normalize(
        const std::string& colRef,
        Eigen::Ref<Eigen::VectorXf, 0, Eigen::InnerStride<>> col,
        NormalizeType type = NormalizeType::None
    ) {
        if (col.size() == 0) return;

        if (stats.find(colRef) == stats.end()) {
            float min_val = col.minCoeff();
            float max_val = col.maxCoeff();
            float mean_val = col.mean();
            float std_dev_val = safeStdDev(col, mean_val);

            stats[colRef] = Stats(min_val, max_val, mean_val, std_dev_val, type);
        }
        else if (type == NormalizeType::None) {
            type = stats[colRef].type;
        }

        switch (type) {
        case NormalizeType::None: return;
        case NormalizeType::MinMax: minmax(colRef, col); return;
        case NormalizeType::ZScore: zscore(colRef, col); return;
        default: throw std::runtime_error("Unknown normalizer type!");
        }
    }

    void denormalize(const std::string& colRef, Eigen::Ref<Eigen::VectorXf, 0, Eigen::InnerStride<>> col) {
        if (col.size() == 0) return;

        if (stats.find(colRef) == stats.end()) {
            throw std::runtime_error("No stats found for column " + colRef);
        }

        switch (stats[colRef].type) {
        case NormalizeType::None: return;
        case NormalizeType::MinMax: reverse_minmax(colRef, col); return;
        case NormalizeType::ZScore: reverse_zscore(colRef, col); return;
        default: throw std::runtime_error("Unknown normalizer type!");
        }
    }

    void fit_transform(DataUtility::DataMatrix<float>& dataset, DataUtility::DataMatrix<float>& labels, const NormalizeType normType) {
        *this = Normalizer();																			// reset normalizer
        // asEigen gives us a Map object which is like a matrix view of the underlying data
        auto dataset_matrix = dataset.asEigen();
        auto labels_matrix = labels.asEigen();

        // Normalize each feature/column
        for (int i = 0; i < dataset.cols; i++) {
            this->normalize("Data " + std::to_string(i), dataset_matrix.col(i), normType);
        }

        for (int i = 0; i < labels.cols; i++) {
            this->normalize("Label " + std::to_string(i), labels_matrix.col(i), normType);
        }
    }

    void fit_transform(DataUtility::DataMatrix<float>& dataset, const NormalizeType normType) {
        *this = Normalizer();																			// reset normalizer
        // asEigen gives us a Map object which is like a matrix view of the underlying data
        auto dataset_matrix = dataset.asEigen();

        // Normalize each feature/column
        for (int i = 0; i < dataset.cols; i++) {
            this->normalize("Data " + std::to_string(i), dataset_matrix.col(i), normType);
        }
    }

    void transform(DataUtility::DataMatrix<float>& dataset) {
        auto dataset_matrix = dataset.asEigen();
        // Normalize each feature/column
        for (int i = 0; i < dataset.cols; i++) {
            this->normalize("Data " + std::to_string(i), dataset_matrix.col(i));
        }
    }


    float normalizeSingle(const std::string& colRef, float val) const {
        const auto& s = stats.at(colRef);
        switch (s.type) {
        case NormalizeType::None: return val;
        case NormalizeType::MinMax: {
            float range = (s.max - s.min == 0) ? 1e-8f : (s.max - s.min);
            return (val - s.min) / range;
        }
        case NormalizeType::ZScore:
            return (val - s.mean) / ((s.std_dev == 0) ? 1e-8f : s.std_dev);
        default: throw std::runtime_error("Unknown normalizer type!");
        }
    }

    float denormalizeSingle(const std::string& colRef, float val) const {
        const auto& s = stats.at(colRef);
        switch (s.type) {
        case NormalizeType::None: return val;
        case NormalizeType::MinMax: {
            float range = (s.max - s.min == 0) ? 1e-8f : (s.max - s.min);
            return val * range + s.min;
        }
        case NormalizeType::ZScore:
            return val * s.std_dev + s.mean;
        default: throw std::runtime_error("Unknown normalizer type!");
        }
    }

    bool hasStats(const std::string& colRef) const {
        return stats.find(colRef) != stats.end();
    }

    void test() {
        for (const auto& [feature, s] : stats) {
            std::cout << "Feature: " << feature << " => "
                << "Min: " << s.min << ", Max: " << s.max
                << ", Mean: " << s.mean << ", StdDev: " << s.std_dev
                << ", Type: " << static_cast<int>(s.type) << "\n";
        }
    }

    void saveNormalizer(std::fstream& file) const {
        size_t size = this->stats.size();

        io::writeNumeric<size_t>(file, size);
        for (const auto& [key, value] : this->stats) {
            // key: string
            // value: Stats object

            io::writeString(file, key);
            io::writePODStruct<Stats>(file, value);
        }
    }

    void readNormalizer(std::fstream& file) {
        size_t size = io::readNumeric<size_t>(file);

        for (size_t s = 0; s < size; s++) {
            std::string key = io::readString(file);
            Stats statistics = io::readPODStruct<Stats>(file);

            this->stats[key] = statistics;
        }
    }
};
