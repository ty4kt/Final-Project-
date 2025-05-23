# 💳 Fraud Detection Using Machine Learning (XGBoost)

This project presents a complete machine learning pipeline designed to detect fraudulent financial transactions using real-world anonymised payment data. Built with a focus on **modularity**, **performance**, and **scalability**, the system leverages the XGBoost algorithm, advanced feature engineering, and class balancing techniques to deliver a production-ready fraud detection model.

---

## 🔍 Project Highlights

- ✅ **Real-World Financial Data** (50,000 transactions)
- ⚙️ **XGBoost Classifier** optimised via GridSearchCV
- 🔐 **GDPR-Compliant Preprocessing** with SHA-256 anonymisation
- 📈 **Advanced Evaluation**: F1-score, Fβ-score (β=2), Confusion Matrix
- 🎯 **Threshold Tuning** for Recall-Precision optimisation
- 🧠 **Modular Python Architecture** with reusable components

---

## 🧠 Skills Demonstrated

- Python (Pandas, NumPy, Scikit-learn, XGBoost)
- Data preprocessing & feature engineering
- Handling class imbalance in imbalanced datasets
- Model evaluation & threshold tuning
- Modular code design and Git version control

---

## 🚀 Results

| Metric        | Score    |
|---------------|----------|
| **Accuracy**  | 99.45%   |
| **Recall**    | 72.41%   |
| **Precision** | 51.85%   |
| **F1-Score**  | 0.6043   |
| **Fβ-Score**  | 0.6709   |

The model successfully captured **72% of fraud cases** while keeping false positives low – making it highly suitable for banking and e-commerce platforms.

---

## 🗂️ Project Structure

📁 vision_base.py       # Pipeline orchestration
📁 vision_mod.py        # Preprocessing class (scaling, encoding, hashing)
📁 vision_train.py      # Model training and evaluation
📁 README.md            # Project overview and setup
📁 report.pdf           # Final technical report
📁 poster.jpg           # A3 summary poster (public showcase)
📁 screenshot           # Screenshot of dataset, as dataset link too large to attach
