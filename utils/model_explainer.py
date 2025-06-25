import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
import warnings


class ModelExplainer:
    """
    Provides model explainability features, including:
    - Feature importance
    - Partial dependence plots
    - SHAP values (simplified implementation)
    """

    def __init__(self):
        """Initialize the ModelExplainer."""
        pass

    def get_feature_importance(self,
                               X: np.ndarray,
                               models: Dict[str, Any],
                               feature_names: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Extract feature importance from models.

        Parameters:
            X: Feature matrix
            models: Dictionary of trained models
            feature_names: List of feature names

        Returns:
            Dictionary of feature importance for each model
        """
        feature_importance = {}

        for model_name, model in models.items():
            try:
                # Try to get feature importance
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # Linear models
                    importances = np.abs(model.coef_).flatten()
                    # If multiclass, take the mean across classes
                    if importances.ndim > 1:
                        importances = np.mean(importances, axis=0)
                else:
                    # Skip models without feature importance
                    continue

                # Check length of feature names and importances
                if len(importances) > len(feature_names):
                    # Truncate importances if longer than feature names
                    importances = importances[:len(feature_names)]
                elif len(importances) < len(feature_names):
                    # Pad importances if shorter than feature names
                    padding = np.zeros(len(feature_names) - len(importances))
                    importances = np.concatenate([importances, padding])

                # Create DataFrame
                importance_df = pd.DataFrame({
                    'Feature': feature_names[:len(importances)],
                    'Importance': importances
                })

                # Sort by importance
                importance_df = importance_df.sort_values('Importance', ascending=False)

                # Add to dictionary
                feature_importance[model_name] = importance_df

            except Exception as e:
                # Skip if feature importance cannot be extracted
                warnings.warn(f"Could not extract feature importance from {model_name}: {str(e)}")

        return feature_importance

    def plot_partial_dependence(self,
                                model: Any,
                                X: np.ndarray,
                                feature_idx: int,
                                feature_name: Optional[str] = None) -> plt.Figure:
        """
        Create a partial dependence plot for a feature.

        Parameters:
            model: Trained model
            X: Feature matrix
            feature_idx: Index of the feature
            feature_name: Name of the feature

        Returns:
            Matplotlib figure with partial dependence plot
        """
        # Get feature values
        feature_values = X[:, feature_idx]

        # Create a grid of values
        grid = np.linspace(
            np.min(feature_values),
            np.max(feature_values),
            num=50
        )

        # Initialize predictions
        mean_predictions = []

        # For each value in the grid
        for value in grid:
            # Create a copy of X
            X_temp = X.copy()

            # Replace the feature with the grid value
            X_temp[:, feature_idx] = value

            # Get predictions
            predictions = model.predict(X_temp)

            # Calculate mean prediction
            mean_predictions.append(np.mean(predictions))

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot partial dependence
        ax.plot(grid, mean_predictions)

        # Set labels and title
        feature_title = feature_name or f"Feature {feature_idx}"
        ax.set_xlabel(feature_title)
        ax.set_ylabel('Predicted Value (average)')
        ax.set_title(f'Partial Dependence Plot for {feature_title}')

        # Add a rug plot at the bottom
        ax.plot(feature_values, [ax.get_ylim()[0]] * len(feature_values), '|', color='blue', alpha=0.2)

        # Ensure figure is tight
        plt.tight_layout()

        return fig

    def explain_prediction(self,
                           model: Any,
                           X_sample: np.ndarray,
                           feature_names: List[str]) -> Dict[str, float]:
        """
        Explain a prediction for a single sample.

        Parameters:
            model: Trained model
            X_sample: Sample to explain
            feature_names: List of feature names

        Returns:
            Dictionary of feature contributions
        """
        explanation = {}

        # If model has feature importance, use it
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            # Baseline prediction
            baseline_pred = model.predict(X_sample)[0]

            # Get contributions using a simple perturbation approach
            contributions = {}

            for i, feature_name in enumerate(feature_names):
                if i >= X_sample.shape[1]:
                    break

                # Create a perturbed sample with this feature zeroed
                X_perturbed = X_sample.copy()
                X_perturbed[0, i] = 0

                # Get prediction
                perturbed_pred = model.predict(X_perturbed)[0]

                # Calculate contribution
                contribution = baseline_pred - perturbed_pred

                # Store contribution
                contributions[feature_name] = contribution

            # Normalize contributions to sum to the prediction
            total_contribution = sum(abs(c) for c in contributions.values())
            if total_contribution > 0:
                for feature, contribution in contributions.items():
                    contributions[feature] = contribution / total_contribution * baseline_pred

            explanation = contributions
        else:
            # For models without feature importance, use a generic approach
            prediction = model.predict(X_sample)[0]
            explanation = {"prediction": prediction, "explanation": "Model does not support detailed explanations"}

        return explanation

    def plot_explanation(self,
                         explanation: Dict[str, float],
                         feature_names: List[str]) -> plt.Figure:
        """
        Plot the explanation for a prediction.

        Parameters:
            explanation: Dictionary of feature contributions
            feature_names: List of feature names

        Returns:
            Matplotlib figure with explanation plot
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        if isinstance(explanation, dict) and all(feature in feature_names for feature in explanation.keys()):
            # Sort contributions
            sorted_contributions = sorted(
                explanation.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            # Get top 10 features
            top_features = sorted_contributions[:10]

            # Extract features and contributions
            features, contributions = zip(*top_features)

            # Plot horizontal bar chart
            bars = ax.barh(
                range(len(features)),
                contributions,
                color=['red' if c < 0 else 'blue' for c in contributions]
            )

            # Add feature names
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)

            # Add contribution values
            for i, bar in enumerate(bars):
                width = bar.get_width()
                value = contributions[i]
                text_color = 'black'
                ax.text(
                    width * (1.01 if width >= 0 else 0.99),
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.3f}",
                    va='center',
                    color=text_color
                )

            # Set labels and title
            ax.set_xlabel('Contribution to Prediction')
            ax.set_title('Feature Contributions to Prediction')
        else:
            # Generic plot for unsupported models
            ax.text(0.5, 0.5, "Detailed explanation not available for this model",
                    horizontalalignment='center', verticalalignment='center')

        # Ensure figure is tight
        plt.tight_layout()

        return fig
