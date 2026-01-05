# ⚡ VOLT: High-Performance MLP Engine

**VOLT** is a lightweight, header-only, high-performance C++ Multi-Layer Perceptron (MLP) engine built from the ground up. It leverages **Eigen** for optimized linear algebra and **OpenMP** for multi-threaded training, achieving lightning-fast convergence on classic datasets like MNIST.

### 🚀 Performance Spotlight

* **Dataset:** MNIST (7k train / 3k test)
* **Time:** ~3.0 seconds for 19 epochs
* **Accuracy:** 95% (Test Set)

---

## ✨ Key Features

### 🧠 Core Neural Network

* **Layer Operations:** Full forward pass and backpropagation implementations.
* **Initialization:** Optimized weight initialization strategies.
* **Activations:** Support for 8 types (ReLU, Sigmoid, Softmax, Tanh, Leaky ReLU, etc.).
* **Loss Functions:** 4 standard types including Categorical Cross-Entropy and MSE.
* **Regularization:** Prevent overfitting with L1, L2, or Elastic Net.

### 🛠️ Optimization & Training

* **Parallelization:** Powered by **OpenMP** for multi-core processing.
* **Advanced Optimizers:** Choose from SGD, Momentum, **Adam**, and RMSprop.
* **Batching:** Full support for mini-batch gradient descent.
* **Model Persistence:** Built-in methods to Save/Load models for deployment.

### 📊 Data & Usability

* **Custom Data Objects:** High-level wrappers over `std::vector` and `Eigen` for maximum compatibility and ease of use.
* **Preprocessing:** Integrated CSV reader (rapidcsv), normalization (MinMax/Standard), and One-Hot encoding.
* **Validation:** Automated Train/Test splitting with support for **Stratified** sampling.

---

## 🛠️ Getting Started

### Prerequisites

* **Compiler:** C++20 compatible (GCC/MinGW recommended).
* **Library:** [Eigen](https://eigen.tuxfamily.org/) (Included as a submodule).

### Installation

```bash
git clone --recursive https://github.com/why-sobi/VOLT.git
cd VOLT
cmake -S . -B build -G "MinGW Makefiles" -D CMAKE_BUILD_TYPE=Release

```

* Use any compiler you have.

---

## 💻 Usage Example

Training a model on MNIST is as simple as defining your layers and calling `.train()`:

```cpp
#include <Model/MLP.hpp>

int main() {
    // 1. Load and Prepare Data
    auto [X, y] = DataUtility::readCSV<float>("mnist.csv", { "label" });
    y = DataUtility::one_hot_encode(y);
    auto [X_train, y_train, X_test, y_test] = DataUtility::train_test_split(X, y, 0.3f);

    // 2. Define Architecture
    MultiLayerPerceptron model(
        static_cast<int>(X_train.cols),
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
    model.addLayer(64, Activation::ActivationType::ReLU);
    model.addLayer(static_cast<int>(y.cols), Activation::ActivationType::Softmax);

    // 5. Train
    model.train(X_train, y_train, 30, 64, 3); 

    return 0;
}

```

## NOTE

This repository is designed for learning and understanding the basics of AI. You are encouraged to dig deep and tweak parts of code as you desire.