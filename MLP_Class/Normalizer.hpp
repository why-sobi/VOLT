# pragma once

#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <numeric>
#include <fstream>

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
        std::vector<std::string> featureOrder;

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
				featureOrder.push_back(column_name); // maintain order of features
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

		const std::vector<std::string> getFeatureOrder() const {
			return featureOrder;
		}

		void save(const std::string& filename) const {
			// Implement saving logic here
			std::cout << "Saving normalizer stats to " << filename << std::endl;

            std::ofstream out(filename + "_norm.bin", std::ios::binary);

            if (!out) {
				throw std::runtime_error("Failed to open file for saving normalizer stats.");
                std::exit(EXIT_FAILURE);
			}

			size_t feature_count = featureOrder.size();

            out.write(reinterpret_cast<const char*>(&feature_count), sizeof(feature_count));
			for (const auto& feature : featureOrder) {
				const auto& s = stats.at(feature);
                size_t feature_length = feature.length(); 
				out.write(reinterpret_cast<const char*>(&feature_length), sizeof(feature_length)); // write length of feature name
                out.write(reinterpret_cast<const char*>(feature.c_str()), feature_length);
				out.write(reinterpret_cast<const char*>(&s.min), sizeof(s.min));
				out.write(reinterpret_cast<const char*>(&s.max), sizeof(s.max));
				out.write(reinterpret_cast<const char*>(&s.mean), sizeof(s.mean));
				out.write(reinterpret_cast<const char*>(&s.std_dev), sizeof(s.std_dev));
				out.write(reinterpret_cast<const char*>(&s.type), sizeof(s.type));
			}

            out.close();
		}

        void load(const std::string& filename) {
            std::cout << "Loading normalizer stats from " << filename << std::endl;
            std::ifstream in(filename + "_norm.bin", std::ios::binary);

            if (!in) {
                throw std::runtime_error("Failed to open file for loading normalizer stats.");
                std::exit(EXIT_FAILURE);
            }

            size_t feature_count;
            in.read(reinterpret_cast<char*>(&feature_count), sizeof(feature_count));
            stats.clear();
            featureOrder.clear();

            for (size_t i = 0; i < feature_count; ++i) {
                size_t feature_length;
				std::string feature;
				std::vector<char> buffer;
                
                in.read(reinterpret_cast<char*>(&feature_length), sizeof(feature_length));
				buffer.resize(feature_length);
				in.read(buffer.data(), feature_length);
                feature.assign(buffer.data(), feature_length);


                Stats s;
                in.read(reinterpret_cast<char*>(&s.min), sizeof(s.min));
                in.read(reinterpret_cast<char*>(&s.max), sizeof(s.max));
                in.read(reinterpret_cast<char*>(&s.mean), sizeof(s.mean));
                in.read(reinterpret_cast<char*>(&s.std_dev), sizeof(s.std_dev));
                int type_int;
                in.read(reinterpret_cast<char*>(&type_int), sizeof(type_int));
                s.type = static_cast<NormalizeType>(type_int);
                stats[feature] = s;
            }

			in.close();
        }

        void test() {
			for (const auto& [feature, s] : stats) {
				std::cout << "Feature: " << feature << " => "
					<< "Min: " << s.min << ", Max: " << s.max
					<< ", Mean: " << s.mean << ", StdDev: " << s.std_dev
					<< ", Type: " << static_cast<int>(s.type) << "\n";
			}
        }
    };
