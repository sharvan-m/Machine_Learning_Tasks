<div align="center">

# Machine Learning & NLP Tasks

[![GitHub Repo](https://img.shields.io/badge/GitHub-sharvan--m%2FMachine__Learning__Tasks-blue?logo=github&logoColor=white)](https://github.com/sharvan-m/Machine_Learning_Tasks)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20Pandas-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%7C%20LinearSVC-green)](#-task-1---movie-genre-classification)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#-license)

**A collection of production-grade Machine Learning and Natural Language Processing solutions. Features modular pipelines, pre-trained model artifacts, and interactive prediction CLIs.**

[Overview](#-overview) • [Task 1: Movie Genre](#-task-1---movie-genre-classification) • [Task 2: Customer Churn](#-task-2---customer-churn-prediction) • [Task 3: SMS Spam](#-task-3---sms-spam-detection) • [Tech Stack](#-technology-stack) • [Repository Structure](#-repository-structure) • [Quick Start](#-quick-start)

</div>

---

## 📖 Overview

This repository holds end-to-end Machine Learning pipelines covering text classification, customer analytics, and spam filtering. Each project is engineered with:
- **Portable Architecture:** Relative project paths allow seamless execution across any OS without hardcoded dependencies.
- **Model Persistence:** Serialized `.joblib` pipelines bundling feature extraction and trained estimators for immediate zero-training inference.
- **Interactive CLI Interfaces:** Dedicated `predict.py` scripts enabling real-time terminal testing with custom inputs.

---

## 📊 Projects & Performance Summary

| Task | Domain | Algorithm | Dataset Size | Primary Metric |
| :--- | :--- | :--- | :--- | :--- |
| **[Task 1: Movie Genre](#-task-1---movie-genre-classification)** | NLP / Text Classification | TF-IDF + [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) | 54,214 records (27 genres) | **57.26% Accuracy** (27 classes) |
| **[Task 2: Customer Churn](#-task-2---customer-churn-prediction)** | Tabular / Predictive Analytics | [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) | 10,000 bank customer profiles | **84.10% Accuracy** |
| **[Task 3: SMS Spam Detection](#-task-3---sms-spam-detection)** | NLP / Spam Filtering | TF-IDF + [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) | 5,572 mobile SMS messages | **98.74% Accuracy** |

---

## 🎬 Task 1 - Movie Genre Classification

Predicts a movie's genre based purely on its plot synopsis and title.

```
[Movie Title + Description] ──► [TF-IDF Vectorizer (ngram 1-2)] ──► [LinearSVC Classifier] ──► [Predicted Genre]
```

- **Objective:** Multi-class classification across **27 distinct genres** (Drama, Comedy, Thriller, Action, Sci-Fi, etc.).
- **Dataset:** 54,214 plot summaries from IMDb.
- **Model Pipeline:** Sublinear TF-IDF vectorization with stopword filtering coupled with a Support Vector Classifier (`LinearSVC`).
- **Validation Accuracy:** **57.26%** (strong performance over a 27-class baseline).
- **Interactive CLI:**
  ```bash
  cd Task-1-Movie-Genre-Classification
  python predict.py
  ```

---

## 📉 Task 2 - Customer Churn Prediction

Identifies bank clients at risk of discontinuing their accounts to enable proactive retention campaigns.

```
[Client Demographics & Account Metrics] ──► [Feature Scaler & Encoder] ──► [Random Forest Ensemble] ──► [Churn Risk (0 / 1)]
```

- **Features Analyzed:** Credit Score, Geography, Gender, Age, Tenure, Account Balance, Number of Products, Credit Card Ownership, Active Member Status, Estimated Salary.
- **Dataset:** 10,000 real-world banking records.
- **Algorithm:** **Random Forest Classifier** with balanced class weights and decision tree bagging.
- **Accuracy:** **84.10%**.
- **Interactive CLI:**
  ```bash
  cd Task-2-Customer-Churn-Prediction
  python predict.py
  ```

---

## 📱 Task 3 - SMS Spam Detection

Filters mobile SMS messages into **ham** (legitimate) or **spam** to protect users against phishing and automated spam.

```
[Raw SMS Text] ──► [Text Preprocessing & TF-IDF] ──► [Linear Support Vector Classifier] ──► [Ham / Spam Flag]
```

- **Dataset:** 5,572 labeled mobile text messages.
- **Architecture:** Term Frequency-Inverse Document Frequency (TF-IDF) feature weighting with Linear SVM.
- **Accuracy:** **98.74%** with near-zero false-positive rates on legitimate messages.
- **Interactive CLI:**
  ```bash
  cd Task-3-SMS-Spam-Detection
  python predict.py
  ```

---

## 🛠️ Technology Stack

| Library / Tool | Purpose |
| :--- | :--- |
| **[Python 3.12](https://www.python.org/)** | Runtime environment |
| **[Scikit-Learn](https://scikit-learn.org/)** | Estimators (`LinearSVC`, `RandomForestClassifier`), metrics, and pipelines |
| **[Pandas](https://pandas.pydata.org/)** | Tabular data manipulation, cleaning, and preprocessing |
| **[NumPy](https://numpy.org/)** | Efficient multidimensional array computations |
| **[Joblib](https://joblib.readthedocs.io/)** | High-performance serialization and deserialization of trained models |

---

## 📂 Repository Structure

```text
Machine_Learning_Tasks/
├── Task-1-Movie-Genre-Classification/
│   ├── data/                           # Training & test dataset text files
│   ├── movie_genre_model.joblib        # Pre-trained TF-IDF + LinearSVC pipeline
│   ├── test_predictions.csv            # Model inference outputs on evaluation data
│   ├── train_model.py                  # Training pipeline script
│   ├── predict.py                      # Interactive single-input CLI tool
│   └── requirements.txt                # Pinned dependencies
│
├── Task-2-Customer-Churn-Prediction/
│   ├── data/Churn_Modelling.csv        # Bank customer tabular dataset
│   ├── churn_model.joblib              # Pre-trained Random Forest model
│   ├── train_model.py                  # Model training and metric evaluation
│   ├── predict.py                      # Interactive customer churn risk calculator
│   └── requirements.txt                # Pinned dependencies
│
├── Task-3-SMS-Spam-Detection/
│   ├── data/spam.csv                   # Labeled SMS spam dataset
│   ├── spam_model.joblib               # Pre-trained SMS classification pipeline
│   ├── train_model.py                  # Training script with accuracy reports
│   ├── predict.py                      # Interactive spam detector CLI
│   └── requirements.txt                # Pinned dependencies
│
├── .gitignore                          # Cache & environment exclusions
└── README.md                           # Master repository documentation
```

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/sharvan-m/Machine_Learning_Tasks.git
cd Machine_Learning_Tasks
```

### 2. Set Up Virtual Environment
```bash
# Create and activate environment
python -m venv venv
.\venv\Scripts\Activate.ps1       # On Linux/macOS: source venv/bin/activate
```

### 3. Run Any Task
Navigate to the desired project directory and install dependencies:
```bash
# Example: Running the SMS Spam Detector
cd Task-3-SMS-Spam-Detection
pip install -r requirements.txt

# Launch interactive inference (uses pre-trained model)
python predict.py

# Or retrain from scratch
python train_model.py
```

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).

*Developed by [Sharvan M](https://github.com/sharvan-m).*
