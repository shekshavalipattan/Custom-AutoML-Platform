import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Dict, Any, Tuple
import time

# Import models for classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Import models for regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


class ModelTrainer:
    """
    Handles model training, including:
    - Model selection
    - Hyperparameter tuning
    - Cross-validation
    - Model evaluation
    """

    def __init__(self, problem_type: str = 'Classification',
                 cv_folds: int = 5,
                 test_size: float = 0.2,
                 time_limit: int = 120):
        """
        Initialize the ModelTrainer.

        Parameters:
            problem_type: 'Classification' or 'Regression'
            cv_folds: Number of cross-validation folds
            test_size: Proportion of data to use for testing
            time_limit: Maximum training time in seconds
        """
        self.problem_type = problem_type
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.time_limit = time_limit

        # Define available models
        self.classification_models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(),
            "DecisionTree": DecisionTreeClassifier(random_state=42),
        }

        if XGB_AVAILABLE:
            self.classification_models["XGBoost"] = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )

        self.regression_models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "SVR": SVR(),
            "KNN": KNeighborsRegressor(),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
        }

        if XGB_AVAILABLE:
            self.regression_models["XGBoost"] = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )

    def train_models(self, X: np.ndarray, y: np.ndarray,
                     selected_models: List[str]) -> Dict[str, Any]:
        """
        Train multiple models and evaluate their performance.

        Parameters:
            X: Feature matrix
            y: Target values
            selected_models: List of model names to train

        Returns:
            Dictionary containing trained models and evaluation metrics
        """
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        # Select models based on problem type
        if self.problem_type == 'Classification':
            model_dict = self.classification_models
        else:
            model_dict = self.regression_models

        # Train selected models
        trained_models = {}
        evaluation_results = {}
        start_time = time.time()

        for model_name in selected_models:
            # Check time limit
            if time.time() - start_time > self.time_limit:
                break

            if model_name not in model_dict:
                continue

            # Get model
            model = model_dict[model_name]

            # Train model
            model.fit(X_train, y_train)

            # Evaluate model
            if self.problem_type == 'Classification':
                evaluation_results[model_name] = self._evaluate_classification(
                    model, X_train, y_train, X_test, y_test
                )
            else:
                evaluation_results[model_name] = self._evaluate_regression(
                    model, X_train, y_train, X_test, y_test
                )

            # Store trained model
            trained_models[model_name] = model

        return {
            "models": trained_models,
            "evaluation": evaluation_results
        }

    def _evaluate_classification(self,
                                 model: Any,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 X_test: np.ndarray,
                                 y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a classification model.

        Parameters:
            model: Trained model
            X_train, y_train: Training data
            X_test, y_test: Test data

        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        # For multi-class, use weighted metrics
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring='accuracy')
        cv_accuracy = cv_scores.mean()

        # Store raw values for reference
        raw_accuracy = accuracy
        raw_f1 = f1
        raw_precision = precision
        raw_recall = recall
        raw_cv_accuracy = cv_accuracy

        # We'll apply some scaling to ensure metrics show meaningful differentiation
        # This will help differentiate between models with similar performance
        # Typically, accuracy above 0.5 is meaningful (better than random guessing)
        # So we'll rescale the range of [0.5, 1.0] to [0, 1.0] for better visualization

        # Scale accuracy to represent improvement over random chance
        n_classes = len(np.unique(y_test))
        random_acc = 1.0 / n_classes if n_classes > 0 else 0.5

        # Normalize metrics to show spread among models
        def rescale(value, min_val=random_acc):
            if value <= min_val:
                return 0.0
            else:
                # Scale to [0, 1] range
                return (value - min_val) / (1.0 - min_val)

        # Apply the scaling to each metric
        accuracy = rescale(accuracy)
        f1 = rescale(f1)
        precision = rescale(precision)
        recall = rescale(recall)
        cv_accuracy = rescale(cv_accuracy)

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "cv_accuracy": cv_accuracy,
            "raw_accuracy": raw_accuracy,
            "raw_f1": raw_f1,
            "raw_precision": raw_precision,
            "raw_recall": raw_recall,
            "raw_cv_accuracy": raw_cv_accuracy
        }

    def _evaluate_regression(self,
                             model: Any,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_test: np.ndarray,
                             y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a regression model.

        Parameters:
            model: Trained model
            X_train, y_train: Training data
            X_test, y_test: Test data

        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())

        # For regression metrics, we'll use a relative scale
        # First, keep the raw error metrics for reference
        raw_mse = mse
        raw_rmse = rmse
        raw_mae = mae
        raw_cv_rmse = cv_rmse

        # R2 score is already in 0-1 range for good models (can be negative for bad models)
        # For other metrics, instead of arbitrary normalization, we'll show in terms of
        # relative performance, where lower is better for error metrics

        # For displaying metrics, we want higher = better performance
        # For error metrics (MSE, RMSE, MAE), we'll convert them to an accuracy-like metric
        # where 0 = worst (highest error) and 1 = best (lowest error)

        # Calculate relative scores based on comparing to mean predictor
        # This gives us a sense of how much better than baseline the model is
        # Mean value predictor would have MSE = variance
        y_variance = np.var(y_test)

        # Avoid division by zero
        if y_variance > 0:
            # For error metrics, convert to "improvement over baseline"
            # where 1 means no error (perfect) and 0 means as bad as baseline
            norm_mse = max(0, min(1 - (mse / (2 * y_variance)), 1))
            norm_rmse = max(0, min(1 - (rmse / (2 * np.sqrt(y_variance))), 1))
            norm_mae = max(0, min(1 - (mae / (2 * np.mean(np.abs(y_test - np.mean(y_test))))), 1))
            norm_cv_rmse = max(0, min(1 - (cv_rmse / (2 * np.sqrt(y_variance))), 1))
        else:
            # Fallback if y_variance is 0
            norm_mse = 1.0 if mse < 0.01 else 0.5  # Some reasonable value
            norm_rmse = 1.0 if rmse < 0.1 else 0.5
            norm_mae = 1.0 if mae < 0.1 else 0.5
            norm_cv_rmse = 1.0 if cv_rmse < 0.1 else 0.5

        # For visualization purposes, also provide percentage versions
        return {
            "mse": norm_mse,  # Normalized [0-1], higher is better
            "rmse": norm_rmse,  # Normalized [0-1], higher is better
            "mae": norm_mae,  # Normalized [0-1], higher is better
            "r2_score": max(0, min(r2, 1)),  # Clip R² to [0,1] range
            "cv_rmse": norm_cv_rmse,  # Normalized [0-1], higher is better
            "raw_mse": mse,  # Keep raw values for reference
            "raw_rmse": rmse,
            "raw_mae": mae,
            "raw_r2": r2
        }
