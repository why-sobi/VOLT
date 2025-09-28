#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <utility>
#include <unordered_set>
#include <Eigen/Dense>

// Defined headers
#include "../Normalizer/Normalizer.hpp"
#include "../Utility/utils.hpp"

// 3rd party
#include "../../libs/rapidcsv/rapidcsv.h"


namespace DataUtility {
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

    using Sample = std::pair<Eigen::VectorX<float>, Eigen::VectorX<float>>;
    
    
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
}