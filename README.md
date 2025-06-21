# MLP Neural Network in C++

This project is a simple implementation of a Multi-Layer Perceptron (MLP) in pure C++, built for educational purposes and to understand how neural networks work under the hood — without relying on high-level Python libraries.

---

## 🧠 Goal

- Build a basic MLP model with customizable layers and activation functions
- Train it on logical gates like AND/XOR, and eventually the MNIST dataset
- Reinforce C++ design, memory handling, and low-level model understanding

---

## 🛠 Dependencies

> All third-party libraries should be placed in the `libs/` folder. This folder is excluded from version control via `.gitignore`.

### 🔹 [Eigen](https://eigen.tuxfamily.org)
- Header-only linear algebra library
- Used for matrix operations and efficient vector math

Structure:
libs/
└── eigen/


### 🔹 [OpenCV](https://opencv.org/releases/)
- Used for loading and displaying image data (e.g., for visualizing or reading MNIST)

Structure:
libs/
└── opencv/
  └── build/
    └── include/
      └── x64/


> Make sure to link `opencv_worldXXXX.lib` and have the corresponding `.dll` in your output folder

---

## 📦 Build

This project is set up using **Visual Studio**. If you’re using the `.sln` file:
1. Make sure all include/lib paths are correctly set in **Configuration Properties**
2. Set runtime DLLs (like OpenCV’s) to auto-copy or paste manually in `Debug/` or `Release/`

---

## 📁 Project Structure

MLP_Class/
├── src/ # Source files (Neuron, Layer, Model)
├── main.cpp # Entry point
├── libs/ # External libraries (ignored in Git)
├── pics/ # Test images 
├── README.md
├── .gitignore
└── MLP_Class.sln



---

## 🚧 Coming Soon

- ✅ Logical gates: AND, XOR
- ⏳ MNIST dataset
- ⏳ EASTL experiment (optional)

---

## 🧑‍💻 Author

Built by [why-sobi](https://github.com/why-sobi), out of curiosity and obsession with how things actually work under the hood.

---

