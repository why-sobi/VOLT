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

		//std::sort(features.begin(), features.end());                                                                    // So that everything is in consistent order
		//std::sort(labels.begin(), labels.end());                                                                        // So that everything is in consistent order

        size_t rows = doc.GetRowCount();
        size_t datasetCols = features.size();
        size_t labelsCols = labels.size();


        Dataset<T> dataset(rows, datasetCols);
        Labels<T> labelsSet(rows, labelsCols);

        std::vector<int> fIdx, lIdx;
        for (const auto& f : features) fIdx.push_back(doc.GetColumnIdx(f));
        for (const auto& l : labels) lIdx.push_back(doc.GetColumnIdx(l));

        // Fill data row by row
        for (size_t i = 0; i < rows; i++) {
            auto row = doc.GetRow<T>(i);

            for (int j : fIdx)
                dataset.data.push_back(row[j]);

            for (int j : lIdx)
                labelsSet.label_values.push_back(row[j]);
        }

        return { dataset, labelsSet };
    }


    template <typename T>
    std::tuple<Dataset<T>, Labels<T>, Dataset<T>, Labels<T>> stratified_train_test_split
    (
        Dataset<T>& dataset,
        Labels<T>& labels,
        float split_ratio,
        std::mt19937& generator,
        std::uniform_int_distribution<int>& distribution
    ) {
        std::cout << distribution(generator) << std::endl;

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
        bool stratfied = false,
		uint16_t random_seed = 42
    ) {
		std::mt19937 generator(random_seed); // Fixed seed for reproducibility
		std::uniform_int_distribution<int> distribution(0, dataset.rows - 1);
        /*stratfied
            ? return stratified_train_test_split(dataset, labels, split_ratio, generator, distribution)
            : return unstratified_train_test_split(dataset, labels, split_ratio, generator);*/
		return unstratified_train_test_split(dataset, labels, split_ratio, generator);
    }
}