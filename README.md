# 🤖 Custom AutoML Platform

A modular, Automated Machine Learning solution built using Python. This platform automates the entire machine learning lifecycle for structured **tabular datasets** — from preprocessing to model training and deployment.

## 📂 Folder Structure

CUSTOMAUTOML/
├── .idea/ # VSCode/IDE project settings (ignored in Git)
├── .streamlit/ # Streamlit UI configuration (if used)
├── utils/ # Custom utility modules
│ ├── data_handler.py # Data preprocessing and feature handling
│ ├── model_evaluator.py # Model evaluation metrics and visualization
│ ├── model_explainer.py # Model interpretability (e.g., SHAP, LIME)
│ ├── model_trainer.py # Training logic for classification/regression
│ └── visualizer.py # Custom visual plotting tools
├── venv/ # Python virtual environment (add to .gitignore)
├── main.py # Entry point for AutoML pipeline
└── app.py # FastAPI app for model deployment


---

## 🚀 Features

- **Automatic Task Detection** – Classification or Regression is auto-detected from target variable.
- **End-to-End Pipeline** – Covers preprocessing, feature engineering, model training, and evaluation.
- **Multiple Algorithms Supported** – Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, etc.
- **Hyperparameter Tuning** – Powered by Optuna for smart optimization.
- **Evaluation & Visuals** – Accuracy, ROC-AUC, R², Confusion Matrix, Feature Importance plots.
- **Local API Deployment** – Real-time predictions served through FastAPI (`app.py`).
- **Modular Design** – All logic abstracted into reusable scripts inside `utils/`.

---


