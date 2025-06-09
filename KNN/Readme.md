# 📌 Implemented KNN from Scratch

In this project, I've implemented the **K-Nearest Neighbors (KNN)** algorithm entirely from scratch using **NumPy only**, without relying on libraries like scikit-learn.

---

## 🤖 What is K-Nearest Neighbors?

**KNN** is a simple, intuitive machine learning algorithm used for **classification** and **regression** tasks. It works by:

- Calculating the distance (commonly Euclidean) from the test point to all training points.
- Selecting the `k` closest neighbors.
- Using **majority voting** (for classification) or **averaging** (for regression) to predict the label/value.

---

## 💡 Features

- ✅ KNN for binary and multi-class classification
- ✅ Pure NumPy implementation
- ✅ Visualize nearest neighbors with `matplotlib`
- ✅ No external ML libraries used

---

## 📦 Requirements

Install dependencies:

```bash
pip install numpy matplotlib
