# VOLT: High-Performance MLP Engine

VOLT is a lightweight, header-only, high-performance C++ Multi-Layer Perceptron (MLP) engine built from the ground up. It leverages Eigen for optimized linear algebra and OpenMP for multi-threaded training, achieving native execution speeds on classic datasets like MNIST.

### Performance Benchmark

The following benchmark reflects training on the full MNIST dataset under identical circumstances, comparing VOLT against scikit-learn's optimized native backend.

* **Dataset:** MNIST (60,000 training samples, 10,000 test samples)
* **Task:** Classification (Input: 784, Hidden Layers: [128], Output: 10)
* **Configuration:** Adam Optimizer (lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8), Batch Size = 64, Single-Precision Float32 Precision

#### Hardware Environment

* **Processor:** Intel Core i5-6300U @ 2.40GHz (2 Cores, 4 Threads)
* **Memory:** 8.00 GB DDR4 @ 2133 MHz (Single-Channel)
* **OS Environment:** Ubuntu Linux via WSL2

#### Execution Summary

| Framework | Optimization Backend | Convergence / Target | Total Training Time | Speedup Factor |
| --- | --- | --- | --- | --- |
| **VOLT (This Engine)** | Eigen (Zero-Allocation Buffers) | 15 Epochs (Early Stopping) | **19.25 seconds** | **3.0x (Baseline)** |
| **scikit-learn** | C/Cython (Native `fit()` loop) | 15 Epochs | 57.64 seconds | 1.0x |

---

## Key Features

### Core Neural Network Architecture

* **Layer Operations:** Native implementations of forward pass and backward propagation. Memory allocations are bound to the internal state of the layers at construction, minimizing runtime heap manipulation.
* **Initialization:** Array weight initialization strategies matching modern network standards.
* **Activations:** Full support for standard activation layers (including ReLU, Sigmoid, Softmax, Tanh, and Leaky ReLU).
* **Loss Functions:** Standard loss tracking implementations including Categorical Cross-Entropy and Mean Squared Error (MSE).
* **Regularization:** Integrated L1, L2, and Elastic Net regularizers evaluated directly within weight tracking steps.

### Optimization & Training

* **Parallelization:** Utilizes OpenMP multi-threading to parallelize core operation blocks across execution units.
* **Advanced Optimizers:** Native implementations of SGD, Momentum, Adam, and RMSprop.
* **Batching:** Native support for mini-batch gradient descent slicing.
* **Model Persistence:** High-performance model serialization to save and load trained configurations.

### Data & Usability

* **Custom Data Objects:** Native high-level wrappers over `std::vector` and `Eigen::Matrix` types to balance safety and performance.
* **Preprocessing:** Integrated CSV parsing (via rapidcsv), data normalization structures (MinMax and Standard scaling), and One-Hot label encoding.
* **Validation:** Automated Train/Test partition utilities with support for Stratified sampling methods.

---

## Getting Started

### Prerequisites

* **Compiler:** C++20 compatible compiler (GCC 11+ or MinGW-w64 recommended).
* **Library:** [Eigen](https://eigen.tuxfamily.org/) (Included as a submodule).

### Installation

```bash
git clone --recursive https://github.com/why-sobi/VOLT.git
cd VOLT
cmake -S . -B build -G "MinGW Makefiles" -D CMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

```

---

## Usage Example

Training a model on a dataset using the VOLT framework:

```cpp
#include <Model/MLP.hpp>

int main() {
    // 1. Load and Prepare Data
    auto [X, y] = DataUtility::readCSV<float>("mnist.csv", { "label" });
    y = DataUtility::one_hot_encode(y);
    auto [X_train, y_train, X_test, y_test] = DataUtility::train_test_split(X, y, 0.3f); // 30% test

    // 2. Define Architecture
    MultiLayerPerceptron model(
        static_cast<int>(X_train.cols()),
        Regularization::L2,
        0.0001f,
        Loss::Type::CategoricalCrossEntropy,
        new Adam(0.001f)
    );

    // 3. Preprocess
    model.normalizer.fit_transform(X_train, NormalizeType::MinMax);
    model.normalizer.transform(X_test);

    // 4. Add Layers
    model.addLayer(128, Activation::ActivationType::ReLU);
    model.addLayer(static_cast<int>(y.cols()), Activation::ActivationType::Softmax);

    // 5. Train
    model.train(X_train, y_train, 15, 64, 3); 

    return 0;
}

```

## Note

This repository is designed for high-performance systems research and understanding the foundations of structural neural network execution. Direct modification and profiling of the execution kernels are encouraged.
