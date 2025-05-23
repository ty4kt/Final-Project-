# 💳 Fraud Detection Using Machine Learning (XGBoost)

This project presents a fully modular and production-ready machine learning pipeline designed to detect fraudulent financial transactions using anonymised, real-world payment data. Leveraging **XGBoost** — a state-of-the-art gradient boosting algorithm — the system prioritises **recall** while maintaining excellent overall accuracy, making it ideal for high-stakes environments like banking and e-commerce.

---

## 🔍 Project Overview

> Fraud detection is a classification task complicated by severe class imbalance, where fraudulent transactions represent a tiny fraction of total records. The cost of false negatives is high — so recall is critical.

This project:
- Uses anonymised real payment data (50,000 records sampled from over 1.2M)
- Builds a modular ML pipeline in Python
- Addresses data imbalance via stratified sampling and threshold tuning
- Trains and tunes an XGBoost model using `GridSearchCV`
- Evaluates with a focus on **recall**, **F1**, and **Fβ-score**

---

## 🧠 Skills Demonstrated

- 🐍 Python programming (Pandas, NumPy, XGBoost, Scikit-learn)
- 📊 Data preprocessing & anonymisation (SHA-256 hashing)
- 🏗️ Modular pipeline design with object-oriented programming
- ⚖️ Class imbalance handling (stratified sampling, threshold tuning)
- 🔍 Hyperparameter tuning & cross-validation (GridSearchCV)
- 📈 Evaluation (confusion matrix, F1, precision/recall, Fβ-score)
- 🔐 GDPR awareness (data privacy and minimisation)
- 🧪 Experiment tracking and reproducibility

---

## ⚙️ Pipeline Steps Explained

### 1. 📦 Data Loading and Initial Cleaning
- Loaded `fraudTrain.csv`, verified existence and format of target variable `is_fraud`
- Normalised labels: values `<1 → 0 (not fraud)`, values `≥1 → 1 (fraud)`

### 2. 🧼 Preprocessing (via `Preprocessing` class)
- **Anonymisation**: Applied SHA-256 to sensitive columns (e.g., name, dob)
- **Scaling**: StandardScaler to normalise numeric features
- **Encoding**: LabelEncoder used for categorical features (e.g., category, merchant)

### 3. 🧪 Sampling
- Used **stratified sampling** to downsample to 50,000 records
- Ensured class balance consistent with original data (~0.58% fraud)

### 4. 🧠 Feature Engineering
- Created derived features:
  - `amt_log`: Log-transformed transaction amount
  - `city_pop_log`: Log-transformed city population
  - `amt_category`: Interaction term between amount and merchant category
- Dropped low-utility features (e.g., cc_num, zip, unix_time)

### 5. 🏋️ Model Training (XGBoost)
- Used `GridSearchCV` to find optimal hyperparameters:
  - `n_estimators: 100–150`
  - `max_depth: 5–6`
  - `learning_rate: 0.05–0.1`
  - `scale_pos_weight: 25–50` (to address class imbalance)
- 5-fold cross-validation used during training

### 6. 🎯 Threshold Tuning
- Evaluated model across thresholds `0.2–0.5`
- Chose `0.5` as optimal balance point based on Fβ (β=2)

---

## 🚀 Results Summary

| Metric        | Value    |
|---------------|----------|
| **Accuracy**  | 99.45%   |
| **Recall**    | 72.41%   |
| **Precision** | 51.85%   |
| **F1-Score**  | 0.6043   |
| **Fβ-Score**  | 0.6709   |

📌 **At threshold 0.5**, the model achieved high recall with tolerable false positives — ideal for catching as much fraud as possible without overwhelming alerts.

🧮 **Confusion Matrix**:
- TP: 42
- FN: 16
- FP: 39
- TN: 9903

---

## 🗂️ Repository Structure

```bash
📁 vision_base.py       # Main pipeline controller (orchestration script)
📁 vision_mod.py        # Preprocessing class: scaling, encoding, anonymisation
📁 vision_train.py      # Training, tuning, threshold evaluation
📁 report.pdf           # Full technical dissertation/report (IEEE style)
📁 poster.jpg           # A3 visual poster (showcase ready)
📁 screenshot/          # Screenshots of dataset (not uploaded due to file size)
📁 README.md            # This file – public-facing project summary
