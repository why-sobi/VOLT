#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <utility>
#include <unordered_set>
#include <tuple>
#include <random>
#include <cstdint>
#include <Eigen/Dense>

// Defined headers
#include "../Normalizer/Normalizer.hpp"
#include "../Utility/utils.hpp"

// 3rd party
#include "../../libs/rapidcsv/rapidcsv.h"


namespace DataUtility {
    using Sample = std::pair<Eigen::VectorX<float>, Eigen::VectorX<float>>;

    template <typename T>
    struct Dataset {
        std::vector<T> data;
        size_t rows, cols;

        Dataset(size_t r, size_t c) : rows(r), cols(c) { data.reserve(rows * cols); }
        Dataset(std::initializer_list<std::initializer_list<T>> init) {
            rows = init.size();
            cols = init.begin()->size();
            data.reserve(rows * cols);

            for (const auto& row : init) {
                if (row.size() != cols)
                    throw std::runtime_error("Inconsistent row sizes in Dataset initializer");
                data.insert(data.end(), row.begin(), row.end());
            }
        }

        float& operator()(size_t i, size_t j) {
            return data[i * cols + j]; // row-major
        }

        const float& operator()(size_t i, size_t j) const {
            return data[i * cols + j];
        }

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            asEigen() {
            return Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                data.data(), rows, cols
            );
        }

        friend std::ostream& operator<<(std::ostream& out, const Dataset& obj) {
            for (size_t i = 0; i < obj.rows; ++i) {
                for (size_t j = 0; j < obj.cols; ++j) {
                    out << obj(i, j) << " ";
                }
                out << "\n";
            }
            return out;
        }
    };

    template <typename T>
    struct Labels {
        std::vector<T> label_values;
        size_t rows, cols;

        Labels(size_t r, size_t c) : rows(r), cols(c) { label_values.reserve(rows * cols); }
        Labels(std::initializer_list<std::initializer_list<T>> init) {
            rows = init.size();
            cols = init.begin()->size();
            label_values.reserve(rows * cols);

            for (const auto& row : init) {
                if (row.size() != cols)
                    throw std::runtime_error("Inconsistent row sizes in Labels initializer");
                label_values.insert(label_values.end(), row.begin(), row.end());
            }
        }

        float& operator()(size_t i, size_t j) {
            return label_values[i * cols + j]; // row-major
        }

        const float& operator()(size_t i, size_t j) const {
            return label_values[i * cols + j];
        }

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            asEigen() {
            return Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                label_values.data(), rows, cols
            );
        }

        friend std::ostream& operator<<(std::ostream& out, const Labels& obj) {
            for (size_t i = 0; i < obj.rows; ++i) {
                for (size_t j = 0; j < obj.cols; ++j) {
                    out << obj(i, j) << " ";
                }
                out << "\n";
            }
            return out;
        }

    };

    template <typename T>
    std::pair<Dataset<T>, Labels<T>> readCSV(const std::string& filename, std::vector<std::string> labels, std::vector<std::string> dropFeatures = {}) {
        if (filename.empty()) { 
            std::cerr << "Please Provide a valid file path\n";
            exit(EXIT_FAILURE);
        }

        if (labels.empty()) {
            std::cerr << "Please Provide Labels\n";
            exit(EXIT_FAILURE);
        }

        rapidcsv::Document doc(filename);                                                                               // reading the dataset
        std::vector<std::string> relevant_cols = doc.GetColumnNames();                                                  // All columns
        std::unordered_set<std::string> colsNames(relevant_cols.begin(), relevant_cols.end());

        if (!dropFeatures.empty()) {                                                                                    // if columns to drop
            for (const std::string& col : dropFeatures) {
                if (colsNames.count(col)) {
                    doc.RemoveColumn(col);                                                                              // remove columns to drop from relevant columns
                } else {
                    std::cerr << "Column: " << col << " does not exist!\n";
                    exit(EXIT_FAILURE);
                }
            }
        }

        relevant_cols = doc.GetColumnNames();                                                                           // All columns
        colsNames = std::unordered_set<std::string>(relevant_cols.begin(), relevant_cols.end());

        for (const std::string& label : labels) {
            if (!colsNames.count(label)) {
                std::cerr << "Label: " << label << " does not exist!\n";
                exit(EXIT_FAILURE);
            }
		}


        std::vector<std::string> features = set_diff(relevant_cols, labels);

        size_t rows = doc.GetRowCount();
        size_t datasetCols = features.size();
        size_t labelsCols = labels.size();


        Dataset<T> dataset(rows, datasetCols);
        Labels<T> labelsSet(rows, labelsCols);

        std::vector<int> fIdx, lIdx;
        for (const auto& f : features) fIdx.push_back(doc.GetColumnIdx(f));
        for (const auto& l : labels) lIdx.push_back(doc.GetColumnIdx(l));


        // right now dataset and labels RESERVE memory hence we use push_back or emplace_back (if it were objects) 
        // if you note slower performance you might consider shifting to RESIZE memory and using [] operator
        
        // Fill data row by row
        for (size_t i = 0; i < rows; i++) {
            for (int j : fIdx)
                dataset.data.push_back(doc.GetCell<T>(i, j));

            for (int j : lIdx)
                labelsSet.label_values.push_back(doc.GetCell<T>(i, j));
        }

        return { dataset, labelsSet };
    }


    template <typename T>
    std::tuple<Dataset<T>, Labels<T>, Dataset<T>, Labels<T>> stratified_train_test_split
    (
        Dataset<T>& dataset,
        Labels<T>& labels,
        float split_ratio,
        std::mt19937& generator
    ) {
		std::uniform_real_distribution<float> distribution(0.0, 1.0);                                                                       // for occasional tie breaker
		
        size_t test_size = static_cast<size_t>(dataset.rows * split_ratio);                                                                 // number of samples in test set
		size_t train_size = dataset.rows - test_size;                                                                                       // number of samples in train set

		Dataset<T> X_train(train_size, dataset.cols), X_test(test_size, dataset.cols);                                                      // train and test datasets
		Labels<T> y_train(train_size, labels.cols), y_test(test_size, labels.cols);                                                         // train and test labels

        int train_sample_count = 0;
        int test_sample_count = 0;

		float train_count_percentage = static_cast<float>(train_size) / static_cast<float>(dataset.rows);                                   // percentage of samples in train set
		float test_count_percentage = static_cast<float>(test_size) / static_cast<float>(dataset.rows);                                     // percentage of samples in test set

		std::unordered_map<T, int> test_label_count, train_label_count;                                                                     // label counts in test and train sets

		auto labels_matrix = labels.asEigen();
		auto dataset_matrix = dataset.asEigen();

        for (size_t i = 0; i < labels.rows; i++) {
			float train_deviance = 0.0f, test_deviance = 0.0f;
            
            for (size_t j = 0; j < labels.cols; j++) {
                if (train_label_count.find(j) == train_label_count.end()) {                                                                 // if label not in train set
                    train_label_count[j] = 0;
				}
                if (test_label_count.find(j) == test_label_count.end()) {                                                                   // if label not in test set
                    test_label_count[j] = 0;
                }

                size_t train_label_increment = labels_matrix(i, j) == 0 ? 0 : 1;
                size_t test_label_increment = labels_matrix(i, j) == 0 ? 0 : 1;

                float new_train_label_percent = static_cast<float>(train_label_count[j] + train_label_increment) / (train_sample_count + 1);// new count if added to train set
                float new_test_label_percent = static_cast<float>(test_label_count[j] + test_label_increment) / (test_sample_count + 1);    // new count if added to test set
            
                train_deviance += std::abs(new_train_label_percent - train_count_percentage);                                               // Calculating train deviance per label from desired count
                test_deviance += std::abs(new_test_label_percent - test_count_percentage);                                                  // Calculating test deviance per label from desired count
            }

            size_t start_idx = (i * dataset.cols), end_idx = start_idx + dataset.cols;                                                      // start and end indices for dataset
            size_t label_start_idx = (i * labels.cols), label_end_idx = label_start_idx + labels.cols;                                      // start and end indices for labels

            bool add_to_train = true;

            if (train_deviance < test_deviance) {                                                                                           // Add to train set
                add_to_train = true;
            } else if (test_deviance < train_deviance) {                                                                                     // Add to test set
                add_to_train = false;
            } else {                                                                                                                         // Tie breaker
                if (distribution(generator) < 0.5f) {                                                                                        // Add to train set
                    add_to_train = true;
                } else {                                                                                                                     // Add to test set
                    add_to_train = false;
                }
            }

            if (add_to_train) {
                X_train.data.insert(X_train.data.end(), dataset.data.begin() + start_idx, dataset.data.begin() + end_idx);
                y_train.label_values.insert(y_train.label_values.end(), labels.label_values.begin() + label_start_idx, labels.label_values.begin() + label_end_idx);
                train_sample_count++;

                for (size_t j = 0; j < labels.cols; j++) {
                        if (labels_matrix(i, j) != 0)
                            train_label_count[j]++;                                                                                         // increasing the label count
                }

            }
            else {
                X_test.data.insert(X_test.data.end(), dataset.data.begin() + start_idx, dataset.data.begin() + end_idx);
                y_test.label_values.insert(y_test.label_values.end(), labels.label_values.begin() + label_start_idx, labels.label_values.begin() + label_end_idx);
                test_sample_count++;

                for (size_t j = 0; j < labels.cols; j++) {
                    if (labels_matrix(i, j) != 0)
                        test_label_count[j]++;                                                                                              // increasing the label count
                }
            }
        }

        return { X_train, y_train, X_test, y_test };
    }

    template <typename T>
    std::tuple<Dataset<T>, Labels<T>, Dataset<T>, Labels<T>> single_label_stratified_train_test_split
    (
        Dataset<T>& dataset,
        Labels<T>& labels,
        float split_ratio,
        std::mt19937& generator
    ) {
        if (labels.cols > 1) {
            std::cerr << "Labels are more than one per sample!\n";
            exit(EXIT_FAILURE);
        }

        size_t test_size = static_cast<size_t>(dataset.rows * split_ratio);                                                                 // number of samples in test set
        size_t train_size = dataset.rows - test_size;                                                                                       // number of samples in train set

        Dataset<T> X_train(train_size, dataset.cols), X_test(test_size, dataset.cols);                                                      // train and test datasets
        Labels<T> y_train(train_size, labels.cols), y_test(test_size, labels.cols);                                                         // train and test labels

        std::unordered_map<T, std::vector<int>> label_idxs;                                                                                 // stores all indices where a certain label exists

        for (int i = 0; i < dataset.rows; i++) {
            T label_value = labels.label_values[i];
            label_idxs[label_value].push_back(i);
        }

        for (auto& [label, idxs] : label_idxs) {
            std::shuffle(idxs.begin(), idxs.end(), generator);
            size_t test_label_size = idxs.size() * split_ratio;
            size_t train_label_size = idxs.size() - test_label_size;

            for (size_t i = 0; i < train_label_size; i++) {
                int idx = idxs[i];
                size_t start_idx = (idx * dataset.cols), end_idx = start_idx + dataset.cols;                                                   // start and end indices for dataset
                size_t label_start_idx = (idx * labels.cols), label_end_idx = label_start_idx + labels.cols;                                   // start and end indices for labels

                X_train.data.insert(X_train.data.end(), dataset.data.begin() + start_idx, dataset.data.begin() + end_idx);                     // insert features
                y_train.label_values.insert(y_train.label_values.end(), labels.label_values.begin() + label_start_idx, labels.label_values.begin() + label_end_idx); // insert labels
            }

            for (size_t i = train_label_size; i < idxs.size(); i++) {
                size_t idx = idxs[i];
                size_t start_idx = (idx * dataset.cols), end_idx = start_idx + dataset.cols;                                                   // start and end indices for dataset
                size_t label_start_idx = (idx * labels.cols), label_end_idx = label_start_idx + labels.cols;                                   // start and end indices for labels

                X_test.data.insert(X_test.data.end(), dataset.data.begin() + start_idx, dataset.data.begin() + end_idx);                       // insert features
                y_test.label_values.insert(y_test.label_values.end(), labels.label_values.begin() + label_start_idx, labels.label_values.begin() + label_end_idx); // insert labels
            }
        }

        return { X_train, y_train, X_test, y_test };
    }


    template <typename T>
    std::tuple<Dataset<T>, Labels<T>, Dataset<T>, Labels<T>> unstratified_train_test_split
    (
        Dataset<T>& dataset,
        Labels<T>& labels,
        float split_ratio,
        std::mt19937& generator
    ) {
		size_t test_size = static_cast<size_t>(dataset.rows * split_ratio);                                                                 // number of samples in test set
		size_t train_size = dataset.rows - test_size;                                                                                       // number of samples in train set

		Dataset<T> X_train(train_size, dataset.cols), X_test(test_size, dataset.cols);                                                      // train and test datasets
		Labels<T> y_train(train_size, labels.cols), y_test(test_size, labels.cols);                                                         // train and test labels

		std::vector<size_t> indices(dataset.rows);                                                                                          // indices for shuffling
		std::iota(indices.begin(), indices.end(), 0);                                                                                       // fill with 0, 1, ..., dataset.rows - 1
		std::shuffle(indices.begin(), indices.end(), generator);                                                                            // shuffle indices

		// Fill train set
        for (size_t i = 0; i < train_size; ++i) {
			size_t idx = indices[i];
            size_t start_idx = (idx * dataset.cols), end_idx = start_idx + dataset.cols;                                                   // start and end indices for dataset
            size_t label_start_idx = (idx * labels.cols), label_end_idx = label_start_idx + labels.cols;                                   // start and end indices for labels

			X_train.data.insert(X_train.data.end(), dataset.data.begin() + start_idx, dataset.data.begin() + end_idx);                     // insert features
			y_train.label_values.insert(y_train.label_values.end(), labels.label_values.begin() + label_start_idx, labels.label_values.begin() + label_end_idx); // insert labels
        }

		// Fill test set
        for (size_t i = train_size; i < indices.size(); i++) {
            size_t idx = indices[i];
            size_t start_idx = (idx * dataset.cols), end_idx = start_idx + dataset.cols;                                                   // start and end indices for dataset
            size_t label_start_idx = (idx * labels.cols), label_end_idx = label_start_idx + labels.cols;                                   // start and end indices for labels

			X_test.data.insert(X_test.data.end(), dataset.data.begin() + start_idx, dataset.data.begin() + end_idx);                       // insert features
			y_test.label_values.insert(y_test.label_values.end(), labels.label_values.begin() + label_start_idx, labels.label_values.begin() + label_end_idx); // insert labels
        }

		return { X_train, y_train, X_test, y_test };
    }

	template <typename T>
    std::tuple<Dataset<T>, Labels<T>, Dataset<T>, Labels<T>> train_test_split
    (
        Dataset<T>& dataset, 
        Labels<T>& labels, 
        float split_ratio = 0.2, 
        bool stratified = false,
		uint16_t random_seed = 42
    ) {
		std::mt19937 generator(random_seed); // Fixed seed for reproducibility
        stratified
            ? return stratified_train_test_split(dataset, labels, split_ratio, generator)
            : return unstratified_train_test_split(dataset, labels, split_ratio, generator);
    }
}