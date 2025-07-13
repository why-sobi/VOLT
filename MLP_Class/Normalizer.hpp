# pragma once

#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <numeric>

enum class NormalizeType { None, MinMax, ZScore };


float getMin(const std::vector<float>& column) { return *std::min_element(column.begin(), column.end()); }
float getMax(const std::vector<float>& column) { return *std::max_element(column.begin(), column.end()); }
float getSum(const std::vector<float>& column) { return std::accumulate(column.begin(), column.end(), 0.0f); }
float getMean(const std::vector<float>& column) { return getSum(column) / column.size(); }
float getMedian(const std::vector<float>& column) { return 0.0f; }
float getMod(const std::vector<float>& column) { return 0.0f; }
float getVariance(const std::vector<float>& column) { return 0.0f; }
float getStdDev(const std::vector<float>& column, float mean) {
    float sq_sum = 0.0f;
    for (float val : column) {
        sq_sum += (val - mean) * (val - mean);
    }

    float std_dev = std::sqrt(sq_sum / column.size());
    return (std_dev == 0) ? 1e-8f : std_dev;
}



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


        std::unordered_map<std::string, Stats> stats;

        void minmax(const std::string& column_name, std::vector<float>& column) {
            const auto& s = stats[column_name];
            float range = s.max - s.min;
            if (range == 0) range = 1e-8f;

            for (float& val : column) {
                val = (val - s.min) / range;
            }
        }

        void zscore(const std::string& column_name, std::vector<float>& column) {
            const auto& s = stats[column_name];
            float std_dev = s.std_dev == 0 ? 1e-8f : s.std_dev;

            for (float& val : column) {
                val = (val - s.mean) / std_dev;
            }
        }

        void reverse_minmax(const std::string& column_name, std::vector<float>& column) {
            const auto& s = stats[column_name];
            float range = s.max - s.min;
            if (range == 0) range = 1e-8f;

            for (float& val : column) {
                val = val * range + s.min;
            }
        }

        void reverse_zscore(const std::string& column_name, std::vector<float>& column) {
            const auto& s = stats[column_name];

            for (float& val : column) {
                val = val * s.std_dev + s.mean;
            }
        }

    public:

        void normalize(const std::string& column_name, std::vector<float>& column, NormalizeType type) {
            if (column.empty()) return;

            if (stats.find(column_name) == stats.end()) {
                float min_val = getMin(column);
                float max_val = getMax(column);
                float mean_val = getMean(column);
                float std_dev_val = getStdDev(column, mean_val);

                stats[column_name] = Stats(min_val, max_val, mean_val, std_dev_val, type);
            }

            switch (type) {
            case NormalizeType::None:
                return;
            case NormalizeType::MinMax:
                minmax(column_name, column);
                return;
            case NormalizeType::ZScore:
                zscore(column_name, column);
                return;
            default:
                throw std::runtime_error("No such normalizer!\n");
            }
        }

        void denormalize(const std::string& column_name, std::vector<float>& column) {
            if (column.empty()) return;

            if (stats.find(column_name) == stats.end()) {
                throw std::runtime_error("No stats found for column: " + column_name);
            }

            switch (stats[column_name].type) {
            case NormalizeType::None:
                return;
            case NormalizeType::MinMax:
                reverse_minmax(column_name, column);
                return;
            case NormalizeType::ZScore:
                reverse_zscore(column_name, column);
                return;
            default:
                throw std::runtime_error("No such normalizer!\n");
            }
        }

        float normalizeSingle(const std::string& column_name, float val) const {
            const auto& s = stats.at(column_name);

            switch (s.type) {
            case NormalizeType::None:
                return val;
            case NormalizeType::MinMax:
                return (val - s.min) / (s.max - s.min == 0 ? 1e-8f : s.max - s.min);
            case NormalizeType::ZScore:
                return (val - s.mean) / (s.std_dev == 0 ? 1e-8f : s.std_dev);
            default:
                throw std::runtime_error("No such normalizer!\n");
            }
        }

        float denormalizeSingle(const std::string& column_name, float val) const {
            const auto& s = stats.at(column_name);

            switch (s.type) {
            case NormalizeType::None:
                return val;
            case NormalizeType::MinMax:
                return val * (s.max - s.min == 0 ? 1e-8f : s.max - s.min) + s.min;
            case NormalizeType::ZScore:
                return val * s.std_dev + s.mean;
            default:
                throw std::runtime_error("No such normalizer!\n");
            }
        }

        bool hasStats(const std::string& column_name) const {
            return stats.find(column_name) != stats.end();
        }
    };
