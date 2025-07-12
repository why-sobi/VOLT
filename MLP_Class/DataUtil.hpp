#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

// Defined headers
#include "Pair.hpp"

// 3rd party
#include "rapidcsv.h"


namespace DataUtil {
    using Sample = Pair<std::vector<float>, std::vector<float>>;

    namespace Normalize {
        enum class Type { None, MinMax, ZScore };

        void minmax(std::vector<float>& column) {
            if (column.empty()) return;

            float min_val = *std::min_element(column.begin(), column.end());
            float max_val = *std::max_element(column.begin(), column.end());

            float range = max_val - min_val;
            if (range == 0) range = 1e-8f; // Prevent divide-by-zero

            for (float& val : column) {
                val = (val - min_val) / range;
            }
        }

        void zscore(std::vector<float>& column) {
            if (column.empty()) return;

            float sum = std::accumulate(column.begin(), column.end(), 0.0f);
            float mean = sum / column.size();

            float sq_sum = 0.0f;
            for (float val : column) {
                sq_sum += (val - mean) * (val - mean);
            }

            float std_dev = std::sqrt(sq_sum / column.size());
            if (std_dev == 0) std_dev = 1e-8f;

            for (float& val : column) {
                val = (val - mean) / std_dev;
            }
        }

        void apply(std::vector<float>& column, Type type) {
            switch (type) {
            case Type::None:
                return;
            case Type::MinMax:
                minmax(column);
                return;
            case Type::ZScore:
                zscore(column);
                return;
            default:
                throw std::runtime_error("No such normalizer!\n");
            }
        }
    }
    // -----------------------------------------------------------------------------------------------------------------------------------------
    // -------------------------------------------------------------- helper  functions --------------------------------------------------------
    // -----------------------------------------------------------------------------------------------------------------------------------------

    // handlers core logic for preprocessing
    std::vector<Sample> PreprocessDatasetHelper(const rapidcsv::Document& doc,
                                size_t rows,
                                const std::vector<std::string>& relevant_cols,
                                const std::vector<std::string>& labels,
		                        const std::vector<std::string>& features,   
                                Normalize::Type defaultType = Normalize::Type::None,
                                const std::unordered_map<std::string, Normalize::Type>& norm_map = {}) 
    { 
        std::vector<Sample> training_dataset;                                                                           // dataset to return
        std::unordered_map<std::string, std::vector<float>> raw_dataset;                                                // temporary form of dataset (allows faster normalization)

        training_dataset.reserve(rows);                                                                                 // allocating in advance so no unnecessary reallocation

        for (const std::string& col : relevant_cols) { // {column name : column data}
            raw_dataset[col] = doc.GetColumn<float>(col);

            auto it = norm_map.find(col);                                                                               // if column is mentioned in norm_map
            // if it is use the mentioned norm function else use the fallback function
            Normalize::apply(raw_dataset[col], it != norm_map.end() ? it->second : defaultType);
        }


        for (size_t i = 0; i < rows; i++) {
            std::vector<float> input;
            std::vector<float> output;

            input.reserve(features.size());                                                                            // allocating in advance so no unnecessary reallocation
            output.reserve(labels.size());                                                                             // allocating in advance so no unnecessary reallocation

            for (const std::string& feature : features) {
                input.push_back(raw_dataset[feature][i]);
            }

            for (const std::string& label : labels) {
                output.push_back(raw_dataset[label][i]);
            }

            training_dataset.emplace_back(input, output);
        }

        return training_dataset;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------------
    // ----------------------------------------------------------------- functions -------------------------------------------------------------
    // -----------------------------------------------------------------------------------------------------------------------------------------

    std::vector<Sample> PreprocessDataset(const std::string& filename, std::vector<std::string> labels, Normalize::Type defaultType = Normalize::Type::None) {
		rapidcsv::Document doc(filename);                                                                               // reading the dataset
        size_t rows = doc.GetRowCount();

		std::vector<std::string> relevant_cols = doc.GetColumnNames();                                                  // All columns
		std::vector<std::string> features;

		// Necessary for set_difference to work
		std::sort(relevant_cols.begin(), relevant_cols.end());
		std::sort(labels.begin(), labels.end());
		std::set_difference(relevant_cols.begin(), relevant_cols.end(),
			labels.begin(), labels.end(),
			std::inserter(features, features.begin()));

		return PreprocessDatasetHelper(doc, rows, relevant_cols, labels, features, defaultType);
    }

    std::vector<Sample> PreprocessDataset(const std::string& filename, std::vector<std::string> labels, Normalize::Type defaultType, const std::unordered_map<std::string, Normalize::Type>& norm_map) {
		rapidcsv::Document doc(filename);                                                                               // reading the dataset
		size_t rows = doc.GetRowCount();

		std::vector<std::string> relevant_cols = doc.GetColumnNames();                                                  // All columns
		std::vector<std::string> features;
		
        // Necessary for set_difference to work
		std::sort(relevant_cols.begin(), relevant_cols.end());
		std::sort(labels.begin(), labels.end());

		std::set_difference(relevant_cols.begin(), relevant_cols.end(),
			labels.begin(), labels.end(),
			std::inserter(features, features.begin()));
		
        return PreprocessDatasetHelper(doc, rows, relevant_cols, labels, features, defaultType, norm_map);
    }
	// overload for dropping columns
    std::vector<Sample> PreprocessDataset(const std::string& filename, std::vector<std::string> labels, std::vector<std::string> dropCols, const Normalize::Type& type = Normalize::Type::None, const std::unordered_map<std::string, Normalize::Type>& norm_map = {}) {
        rapidcsv::Document doc(filename);                                                                               // reading the dataset
        size_t rows = doc.GetRowCount();

        std::vector<std::string> relevant_cols = doc.GetColumnNames();                                                  // All columns
        std::vector<std::string> features;

        // Necessary for set_difference to work
        std::sort(relevant_cols.begin(), relevant_cols.end());
        std::sort(labels.begin(), labels.end());
		std::sort(dropCols.begin(), dropCols.end());

		if (!dropCols.empty()) {                                                                                        // if columns to drop
			std::vector<std::string> new_cols;
			std::set_difference(relevant_cols.begin(), relevant_cols.end(),
				dropCols.begin(), dropCols.end(),
				std::inserter(new_cols, new_cols.begin()));                                                             // remove columns to drop from relevant columns)

			relevant_cols = std::move(new_cols);                                                                        // update relevant columns
		}

        std::set_difference(relevant_cols.begin(), relevant_cols.end(),
            labels.begin(), labels.end(),
            std::inserter(features, features.begin()));

		return PreprocessDatasetHelper(doc, rows, relevant_cols, labels, features, type, norm_map);
    }



    /* std::vector<Sample> PreprocessDataset(
        const std::string& filename,                                                                                    // csv path
        std::vector<std::string> labels,                                                                                // labels names
		std::vector<std::string> dropCols = {},                                                                         // columns to drop
        Normalize::Type defaultType = Normalize::Type::None,                                                            // fallback normalization type
        const std::unordered_map<std::string, Normalize::Type>& norm_map = {})                                          // {column name : normalization type}
    {
		if (filename.empty()) {
			throw std::runtime_error("Filename cannot be empty!\n");
		}

		if (labels.empty()) {
			throw std::runtime_error("Labels cannot be empty!\n");
		}
        
        rapidcsv::Document doc(filename);                                                                               // reading the dataset
        size_t rows = doc.GetRowCount();

        std::vector<std::string> relevant_cols = doc.GetColumnNames();                                                  // All columns
        std::vector<std::string> features;

        // Necessary for set_difference to work
        std::sort(relevant_cols.begin(), relevant_cols.end());
		if (!dropCols.empty()) {                                                                                        // if columns to drop
            std::vector<std::string> new_cols;
            std::sort(dropCols.begin(), dropCols.end());
			std::set_difference(relevant_cols.begin(), relevant_cols.end(),
				dropCols.begin(), dropCols.end(),
				std::inserter(new_cols, new_cols.begin()));                                                             // remove columns to drop from relevant columns)
		
			relevant_cols = std::move(new_cols);                                                                        // update relevant columns
        }
        std::sort(labels.begin(), labels.end());

        std::set_difference(relevant_cols.begin(), relevant_cols.end(),
            labels.begin(), labels.end(),
            std::inserter(features, features.begin()));

        std::vector<Sample> training_dataset;                                                                           // dataset to return
        std::unordered_map<std::string, std::vector<float>> raw_dataset;                                                // temporary form of dataset (allows faster normalization)

        training_dataset.reserve(rows);                                                                                 // allocating in advance so no unnecessary reallocation

        for (const std::string& col : relevant_cols) { // {column name : column data}
            raw_dataset[col] = doc.GetColumn<float>(col);
            
            auto it = norm_map.find(col);                                                                               // if column is mentioned in norm_map
            // if it is use the mentioned norm function else use the fallback function
            Normalize::apply(raw_dataset[col], it != norm_map.end() ? it->second : defaultType);
        }


        for (size_t i = 0; i < rows; i++) {
            std::vector<float> input;
            std::vector<float> output;

            input.reserve(features.size());                                                                            // allocating in advance so no unnecessary reallocation
            output.reserve(labels.size());                                                                             // allocating in advance so no unnecessary reallocation

            for (const std::string& feature : features) {
                input.push_back(raw_dataset[feature][i]);
            }

            for (const std::string& label : labels) {
                output.push_back(raw_dataset[label][i]);
            }

            training_dataset.emplace_back(input, output);
        }

        return training_dataset;
    }*/
}

