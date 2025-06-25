import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, Any, Tuple, List
import seaborn as sns


class ModelEvaluator:
    """
    Handles model evaluation and visualization, including:
    - Performance metrics calculation
    - Confusion matrix
    - ROC curves for classification
    - Residual plots for regression
    """

    def __init__(self, problem_type: str = 'Classification'):
        """
        Initialize the ModelEvaluator.

        Parameters:
            problem_type: 'Classification' or 'Regression'
        """
        self.problem_type = problem_type

    def get_best_model(self, evaluation_results: Dict[str, Dict[str, float]]) -> str:
        """
        Determine the best model based on evaluation metrics.

        Parameters:
            evaluation_results: Dictionary of model evaluation results

        Returns:
            Name of the best model
        """
        if self.problem_type == 'Classification':
            # Use accuracy as the primary metric for classification
            return max(evaluation_results.items(), key=lambda x: x[1]['accuracy'])[0]
        else:
            # Use R² score as the primary metric for regression
            return max(evaluation_results.items(), key=lambda x: x[1]['r2_score'])[0]

    def plot_confusion_matrix(self, model: Any, X: np.ndarray, y: np.ndarray) -> plt.Figure:
        """
        Plot a confusion matrix for a classification model.

        Parameters:
            model: Trained model
            X: Feature matrix
            y: True labels

        Returns:
            Matplotlib figure with confusion matrix
        """
        # Make predictions
        y_pred = model.predict(X)

        # Convert both y and y_pred to the same data type (strings) to avoid mixed type error
        y_true = y.astype(str) if isinstance(y, np.ndarray) else np.array([str(val) for val in y])
        y_predicted = y_pred.astype(str) if isinstance(y_pred, np.ndarray) else np.array([str(val) for val in y_pred])

        # Get unique classes from both actual and predicted values
        classes = np.unique(np.concatenate((y_true, y_predicted)))

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_predicted, labels=classes)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=classes, yticklabels=classes)

        # Set labels
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')

        # Ensure figure is tight
        plt.tight_layout()

        return fig

    def plot_roc_curve(self, model: Any, X: np.ndarray, y: np.ndarray) -> plt.Figure:
        """
        Plot ROC curve for a classification model.

        Parameters:
            model: Trained model
            X: Feature matrix
            y: True labels

        Returns:
            Matplotlib figure with ROC curve
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Check if binary classification
        unique_classes = np.unique(y)
        if len(unique_classes) == 2:
            # Binary classification
            try:
                # Get probability predictions for positive class
                y_score = model.predict_proba(X)[:, 1]

                # Compute ROC curve and AUC
                fpr, tpr, _ = roc_curve(y, y_score)
                roc_auc = auc(fpr, tpr)

                # Plot ROC curve
                ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], 'k--', lw=2)

                # Set labels and title
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC)')
                ax.legend(loc="lower right")
            except:
                ax.text(0.5, 0.5, "ROC curve not available for this model",
                        horizontalalignment='center', verticalalignment='center')
        else:
            # Multi-class not supported
            ax.text(0.5, 0.5, "ROC curve only available for binary classification",
                    horizontalalignment='center', verticalalignment='center')

        # Ensure figure is tight
        plt.tight_layout()

        return fig

    def plot_residuals(self, model: Any, X: np.ndarray, y: np.ndarray) -> plt.Figure:
        """
        Plot residuals for a regression model.

        Parameters:
            model: Trained model
            X: Feature matrix
            y: True values

        Returns:
            Matplotlib figure with residual plot
        """
        # Make predictions
        y_pred = model.predict(X)

        # Calculate residuals
        residuals = y - y_pred

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot residuals
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='-')

        # Set labels and title
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')

        # Ensure figure is tight
        plt.tight_layout()

        return fig

    def plot_prediction_vs_actual(self, model: Any, X: np.ndarray, y: np.ndarray) -> plt.Figure:
        """
        Plot predicted vs actual values for a regression model.

        Parameters:
            model: Trained model
            X: Feature matrix
            y: True values

        Returns:
            Matplotlib figure with prediction vs actual plot
        """
        # Make predictions
        y_pred = model.predict(X)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot predicted vs actual
        ax.scatter(y, y_pred, alpha=0.5)

        # Add a perfect prediction line
        min_val = min(np.min(y), np.min(y_pred))
        max_val = max(np.max(y), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')

        # Set labels and title
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs Actual Values')

        # Ensure figure is tight
        plt.tight_layout()

        return fig
