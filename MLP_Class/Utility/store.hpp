#pragma once

#include <fstream>
#include <vector>
#include <stdexcept>
#include <Eigen/Dense>


// This file simply servers as helper to write different type of variables in a binary file

namespace io {
    // writing simple datatypes (int, floats, doubles ...)
    template <typename T>
    void writeNumeric(std::fstream& file, const T& numeric) {
        if (!file.is_open()) throw std::runtime_error("File is not open or corrupted!\n");
        file.write(reinterpret_cast<const char*>(&numeric), sizeof(T));
    }

    template <typename T>
    T readNumeric(std::fstream& file) {
        if (!file.is_open()) throw std::runtime_error("File is not open or corrupted!\n");

        T value;    
        file.read(reinterpret_cast<char*>(&value), sizeof(T));
        return value;
    }

    void writeString(std::fstream& file, const std::string& str) {
        if (!file.is_open()) throw std::runtime_error("File is not open or corrupted!\n");
     
        size_t length = str.size();
        writeNumeric<size_t>(file, length);
        file.write(str.data(), length);
    }

    std::string readString(std::fstream& file) {
        if (!file.is_open()) throw std::runtime_error("File is not open or corrupted!\n");
        
        std::string str;
        size_t length = readNumeric<size_t>(file);

        str.resize(length);
        file.read(str.data(), length);
        return str;
    }

    template <typename T>
    void writeEnum(std::fstream& file, const T& enumVal) {
        if (!file.is_open()) throw std::runtime_error("File is not open or corrupted!\n");
        writeNumeric<int>(file, static_cast<int>(enumVal));
    }

    template <typename T>
    T readEnum(std::fstream& file) {
        if (!file.is_open()) throw std::runtime_error("File is not open or corrupted!\n");
        return static_cast<T>(readNumeric<int>(file));
    }

    template <typename T>
    void writePODStruct(std::fstream& file, const T& structure) {
        if (!file.is_open()) throw std::runtime_error("File is not open or corrupted!\n");
        file.write(reinterpret_cast<const char*>(&structure), sizeof(structure));
    }

    template <typename T>
    T readPODStruct(std::fstream& file) {
        T structure;
        if (!file.is_open()) throw std::runtime_error("File is not open or corrupted!\n");
        file.read(reinterpret_cast<char*>(&structure), sizeof(structure));

        return structure;
    }

    template <typename T>
    void writeNumericVector(std::fstream& file, const std::vector<T>& vec) {
        if (!file.is_open()) throw std::runtime_error("File is not open or corrupted!\n");
        size_t size = vec.size();
        writeNumeric<size_t>(file, size);

       file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(T));
    }

    template <typename T>
    std::vector<T> readNumericVector(std::fstream& file) {
        if (!file.is_open()) throw std::runtime_error("File is not open or corrupted!\n");
        
        std::vector<T> vec;
        size_t size = readNumeric<size_t>(file);
        vec.resize(size);
        
        file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
        
        return vec;
    }

    template <typename T>
    void writeEigenMatrix(std::fstream& file, const Eigen::MatrixX<T>& mat) {
        if (!file.is_open()) throw std::runtime_error("File is not open or corrupted!\n");

        Eigen::Index rows = mat.rows();
        Eigen::Index cols = mat.cols();

        writeNumeric<Eigen::Index>(file, rows);
        writeNumeric<Eigen::Index>(file, cols);

        file.write(reinterpret_cast<const char*>(mat.data()), mat.size() * sizeof(T));
    }

    template <typename T>
    Eigen::MatrixX<T> readEigenMatrix(std::fstream& file) {
        if (!file.is_open()) {
            throw std::runtime_error("File is corrupted or cannot be opened!\n");
        }

        Eigen::Index rows = readNumeric<Eigen::Index>(file);
        Eigen::Index cols = readNumeric<Eigen::Index>(file);

        Eigen::MatrixX<T> mat(rows, cols);
        file.read(reinterpret_cast<char*>(mat.data()), mat.size() * sizeof(T));

        return mat;
    }

    template <typename T>
    void writeEigenVector(std::fstream& file, const Eigen::VectorX<T>& mat) {
        if (!file.is_open()) throw std::runtime_error("File is not open or corrupted!\n");

        Eigen::Index rows = mat.rows();

        writeNumeric<Eigen::Index>(file, rows);

        file.write(reinterpret_cast<const char*>(mat.data()), mat.size() * sizeof(T));
    }

    template <typename T>
    Eigen::VectorX<T> readEigenVector(std::fstream& file) {
        if (!file.is_open()) {
            throw std::runtime_error("File Couldnt be opened or is corrupted!\n");
        }

        Eigen::Index rows = readNumeric<Eigen::Index>(file);

        Eigen::VectorX<T> mat(rows);
        file.read(reinterpret_cast<char*>(mat.data()), mat.size() * sizeof(T));

        return mat;
    }

    template <typename T>
    void writeEigenMatVector(std::fstream& file, const std::vector <Eigen::MatrixX<T>>& matVector) {
        if (!file.is_open()) return;

        // write total matrices
        writeNumeric<size_t>(file, matVector.size());
        for (const Eigen::MatrixX<T>& mat : matVector) {
            writeEigenMatrix<T>(file, mat);
        }
    }

    template <typename T>
    std::vector<Eigen::MatrixX<T>> readEigenMatVector(std::fstream& file) {
        if (!file.is_open()) {
            throw std::runtime_error("File is not open or corrupted!\n");
        }

        size_t size = readNumeric<size_t>(file);
        std::vector<Eigen::MatrixX<T>> matVector;
        matVector.resize(size);

        for (size_t s = 0; s < size; s++) {
            matVector[s] = readEigenMatrix<T>(file);

        }

        return matVector;
    }

    template <typename T>
    void writeEigenVecVector(std::fstream& file, const std::vector <Eigen::VectorX<T>>& vecVector) {
        if (!file.is_open()) throw std::runtime_error("File is not open or corrupted!\n");;

        // write total matrices
        writeNumeric<size_t>(file, vecVector.size());
        for (const Eigen::VectorX<T>& vec : vecVector) {
            writeEigenVector<T>(file, vec);
        }
    }

    template <typename T>
    std::vector<Eigen::VectorX<T>> readEigenVecVector(std::fstream& file) {
        if (!file.is_open()) {
            throw std::runtime_error("File is not open or corrupted!\n");
        }

        size_t size = readNumeric<size_t>(file);
        std::vector<Eigen::VectorX<T>> vecVector;
        vecVector.resize(size);

        for (size_t s = 0; s < size; s++) {
            vecVector[s] = readEigenVector<T>(file);
        }

        return vecVector;
    }
}
