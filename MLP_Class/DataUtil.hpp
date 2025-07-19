#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_set>

// Defined headers
#include "Normalizer.hpp"
#include "Pair.hpp"

// 3rd party
#include "rapidcsv.h"



namespace DataUtil {
    using Sample = Pair<std::vector<float>, std::vector<float>>;

    
    // -----------------------------------------------------------------------------------------------------------------------------------------
    // -------------------------------------------------------------- helper  functions --------------------------------------------------------
    // -----------------------------------------------------------------------------------------------------------------------------------------

    // handlers core logic for preprocessing
    std::vector<Sample> PreprocessDatasetHelper(const rapidcsv::Document& doc,
                                size_t rows,
                                const std::vector<std::string>& relevant_cols,
                                const std::vector<std::string>& labels,
		                        const std::vector<std::string>& features, 
                                Normalizer& normalizer,
                                NormalizeType defaultType = NormalizeType::None,
                                const std::unordered_map<std::string, NormalizeType>& norm_map = {}) 
    { 
        std::vector<Sample> training_dataset;                                                                           // dataset to return
        std::unordered_map<std::string, std::vector<float>> raw_dataset;                                                // temporary form of dataset (allows faster normalization)

        training_dataset.reserve(rows);                                                                                 // allocating in advance so no unnecessary reallocation

        for (const std::string& col : relevant_cols) { // {column name : column data}
            raw_dataset[col] = doc.GetColumn<float>(col);

            auto it = norm_map.find(col);                                                                               // if column is mentioned in norm_map
            // if it is use the mentioned norm function else use the fallback function
            normalizer.normalize(col, raw_dataset[col], it != norm_map.end() ? it->second : defaultType);
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

    std::vector<Sample> PreprocessDataset(const std::string& filename, std::vector<std::string> labels, Normalizer& normalizer, NormalizeType defaultType = NormalizeType::None) {
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

		return PreprocessDatasetHelper(doc, rows, relevant_cols, labels, features, normalizer, defaultType);
    }

    std::vector<Sample> PreprocessDataset(const std::string& filename, std::vector<std::string> labels, Normalizer& normalizer, NormalizeType defaultType, const std::unordered_map<std::string, NormalizeType>& norm_map) {
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
		
        return PreprocessDatasetHelper(doc, rows, relevant_cols, labels, features, normalizer, defaultType, norm_map);
    }
	// overload for dropping columns
    std::vector<Sample> PreprocessDataset(const std::string& filename, std::vector<std::string> labels, std::vector<std::string> dropCols, Normalizer& normalizer, const NormalizeType& type = NormalizeType::None, const std::unordered_map<std::string, NormalizeType>& norm_map = {}) {
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

		return PreprocessDatasetHelper(doc, rows, relevant_cols, labels, features, normalizer, type, norm_map);
    }

    /*
    std::vector<std::vector<float>> OrderedInputFormat(const std::unordered_map<std::string, std::vector<float>>& input, const std::vector<std::string>& expectedOrder, const std::unordered_set<std::string>& labels, int rows) {
        // Assumes DataUtil::Preprocess
        if (expectedOrder.empty()) throw std::runtime_error("Expected Order cannot be empty!\n");

        std::vector<std::vector<float>> formatInput(rows);

        for (int i = 0; i < rows; i++) {
            std::vector<float> new_input;
            for (const std::string& feature : expectedOrder) {
                auto it = input.find(feature);

                if (it != input.end() && !it->second.empty()) {
                    if (it->second.size() < rows) throw std::runtime_error("All columns should be of equal length: " + rows);
					if (labels.find(feature) == labels.end()) {
                        new_input.push_back(it->second[i]); 
					}
                }
            }
            formatInput[i] = std::move(new_input);
        }
        return formatInput;
    }
    */

}

