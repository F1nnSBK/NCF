# 🧠 Neural Collaborative Filtering for Article Recommendations

Welcome to the NCF training project! This repository contains a Jupyter-based machine learning pipeline to train a **Neural Collaborative Filtering (NCF)** model for an **online media platform**. The goal is to deliver personalized article recommendations based on user interaction data (clicks, views, etc.).

---

## 🚀 Overview

> 📚 Recommender systems are a critical component for digital media companies aiming to improve **user engagement**, **session duration**, and **content discovery**.

This project leverages **deep learning** and **collaborative filtering** techniques to build a scalable, high-performing model tailored to user preferences.

---

## 📦 Project Structure

```Bash
ncf-recommender/
├── ncf.ipynb
├── data/
│ └── ncf_data_v1.csv
├── models/
│ └── ncf_model.pt
├── utils/
│ └── Helper.py
├── requirements.txt
└── README.md
```

---

## 🛠 Tech Stack

- 🐍 **Python 3.10+**
- 🧪 **PyTorch** – model training
- 🔍 **Polars / Pandas** – data handling
- 📊 **Matplotlib / Seaborn** – visualizations
- 📁 **FastAPI** – for potential API serving
- ☁️ **Google Cloud / GCS (optional)** – cloud deployment or storage

---

## 📈 Model Description

The model architecture follows the **Neural Collaborative Filtering (NCF)** framework, which includes:

- **Embedding layers** for users and items
- **Multi-layer perceptron (MLP)** for non-linear feature interactions
- Optional **GMF (Generalized Matrix Factorization)** pathway
- **Sigmoid output** to predict interaction probability

📝 The notebook supports hyperparameter tuning, early stopping, and live evaluation metrics.
