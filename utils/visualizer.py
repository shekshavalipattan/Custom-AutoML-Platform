import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import seaborn as sns


class Visualizer:
    """
    Handles data and result visualization, including:
    - Model performance comparison
    - Feature importance plots
    - Data distribution plots
    - Correlation plots
    """

    def __init__(self):
        """Initialize the Visualizer."""
        pass

    def plot_metrics_comparison(self,
                                evaluation_results: Dict[str, Dict[str, float]],
                                problem_type: str) -> go.Figure:
        """
        Create a comparison plot of model metrics.

        Parameters:
            evaluation_results: Dictionary of model evaluation results
            problem_type: 'Classification' or 'Regression'

        Returns:
            Plotly figure object
        """
        # Convert evaluation results to DataFrame
        results_df = pd.DataFrame.from_dict(evaluation_results, orient='index')

        # Separate raw metrics from normalized metrics
        normalized_metrics = {}
        raw_metrics = {}

        for col in results_df.columns:
            if col.startswith('raw_'):
                raw_metrics[col] = results_df[col]
            else:
                normalized_metrics[col] = results_df[col]

        # Create a new DataFrame with just the normalized metrics
        norm_df = pd.DataFrame(normalized_metrics, index=results_df.index)

        # Convert metrics to percentages (0-100 scale)
        for col in norm_df.columns:
            # Multiply by 100 to convert to percentage
            norm_df[col] = norm_df[col] * 100

        # Select metrics based on problem type
        if problem_type == 'Classification':
            metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall', 'cv_accuracy']
            # Rename metrics to make them clearer
            metric_labels = {
                'accuracy': 'Accuracy (%)',
                'f1_score': 'F1 Score (%)',
                'precision': 'Precision (%)',
                'recall': 'Recall (%)',
                'cv_accuracy': 'CV Accuracy (%)'
            }
        else:
            metrics_to_plot = ['r2_score', 'mse', 'rmse', 'mae', 'cv_rmse']
            # Rename metrics to make them clearer
            metric_labels = {
                'r2_score': 'R² Score (%)',
                'mse': 'MSE Score (%)',  # Already normalized to 0-100 (higher is better)
                'rmse': 'RMSE Score (%)',  # Already normalized to 0-100 (higher is better)
                'mae': 'MAE Score (%)',  # Already normalized to 0-100 (higher is better)
                'cv_rmse': 'CV RMSE Score (%)'  # Already normalized to 0-100 (higher is better)
            }

        # Rename columns for display
        norm_df.rename(columns=metric_labels, inplace=True)

        # Update metrics_to_plot with new labels
        metrics_to_plot = [metric_labels.get(m, m) for m in metrics_to_plot if m in norm_df.columns]

        # Create radar chart for comprehensive comparison
        fig = go.Figure()

        # Add a trace for each model, but check if we have metrics to plot first
        if len(metrics_to_plot) > 0:
            for model_name in norm_df.index:
                model_values = norm_df.loc[model_name, metrics_to_plot].tolist()

                if len(model_values) > 0:  # Only add trace if we have values
                    # Add an extra point to close the loop
                    model_values.append(model_values[0])
                    metrics = metrics_to_plot + [metrics_to_plot[0]]

                    fig.add_trace(go.Scatterpolar(
                        r=model_values,
                        theta=metrics,
                        fill='toself',
                        name=model_name
                    ))
                else:
                    # Fallback for when no metrics are available
                    fig.add_trace(go.Scatterpolar(
                        r=[0],
                        theta=["No metrics available"],
                        name=model_name
                    ))

        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Model Performance Comparison (Normalized Metrics)",
            showlegend=True,
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Create a second figure for bar chart comparison (alternative view)
        # This is helpful for seeing direct comparisons on specific metrics

        # Check if we have metrics to plot
        if len(metrics_to_plot) > 0:
            # Melt DataFrame for Plotly
            melted_df = norm_df[metrics_to_plot].reset_index().melt(
                id_vars='index',
                value_vars=metrics_to_plot,
                var_name='Metric',
                value_name='Value'
            )
            melted_df.rename(columns={'index': 'Model'}, inplace=True)

            # Create bar chart
            bar_fig = px.bar(
                melted_df,
                x='Model',
                y='Value',
                color='Metric',
                barmode='group',
                title='Model Performance Comparison (Bar Chart)',
                labels={'Value': 'Score (%)', 'Model': 'Model'},
                height=600
            )
        else:
            # Create an empty figure with a message
            bar_fig = go.Figure()
            bar_fig.add_annotation(
                text="No metrics available for comparison",
                showarrow=False,
                font=dict(size=14)
            )

        # Update layout
        bar_fig.update_layout(
            xaxis_title='Model',
            yaxis_title='Score (higher is better)',
            legend_title='Metric',
            hovermode='closest',
            yaxis=dict(
                range=[0, 100],
                ticksuffix='%'
            )
        )

        # Combine both charts in a subplot layout
        fig_combined = go.Figure()

        # Add traces from radar chart
        for i, trace in enumerate(fig.data):
            fig_combined.add_trace(trace)

        # Update layout to match the original radar chart
        fig_combined.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Model Performance Comparison (Radar Chart - higher is better for all metrics)",
            showlegend=True,
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig_combined

    def plot_feature_importance(self,
                                feature_importance: pd.DataFrame,
                                top_n: int = 15) -> go.Figure:
        """
        Create a feature importance bar chart.

        Parameters:
            feature_importance: DataFrame with Feature and Importance columns
            top_n: Number of top features to display

        Returns:
            Plotly figure object
        """
        # Sort and select top features
        sorted_features = feature_importance.sort_values('Importance', ascending=False)
        top_features = sorted_features.head(top_n)

        # Create figure
        fig = px.bar(
            top_features,
            y='Feature',
            x='Importance',
            orientation='h',
            title=f'Top {top_n} Feature Importance',
            labels={'Feature': 'Feature', 'Importance': 'Importance Score'},
            height=600,
            color='Importance',  # Color by importance
            color_continuous_scale='Blues'  # Blue color scale
        )

        # Update layout
        fig.update_layout(
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            yaxis={'categoryorder': 'total ascending'},  # Sort y-axis
            hovermode='closest'
        )

        return fig

    def plot_distributions(self,
                           data: pd.DataFrame,
                           columns: List[str]) -> go.Figure:
        """
        Create distribution plots for selected columns.

        Parameters:
            data: DataFrame containing the data
            columns: List of columns to plot

        Returns:
            Plotly figure object
        """
        # Limit to maximum 10 columns
        if len(columns) > 10:
            columns = columns[:10]

        # Create subplots
        fig = go.Figure()

        # Add histogram for each column
        for column in columns:
            fig.add_trace(
                go.Histogram(
                    x=data[column],
                    name=column,
                    opacity=0.7,
                    nbinsx=30
                )
            )

        # Update layout
        fig.update_layout(
            title='Feature Distributions',
            xaxis_title='Value',
            yaxis_title='Count',
            barmode='overlay',  # Overlay histograms
            height=500,
            legend_title='Feature',
            hovermode='closest'
        )

        return fig

    def plot_correlation(self, data: pd.DataFrame) -> go.Figure:
        """
        Create a correlation heatmap.

        Parameters:
            data: DataFrame containing numerical data

        Returns:
            Plotly figure object
        """
        # Calculate correlation
        corr_matrix = data.corr()

        # Create figure
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',  # Show correlation values
            aspect='auto',
            color_continuous_scale='RdBu_r',  # Red-Blue scale
            title='Feature Correlation Matrix',
            height=700,
            width=700
        )

        # Update layout
        fig.update_layout(
            xaxis_title='Feature',
            yaxis_title='Feature',
            coloraxis_colorbar=dict(
                title='Correlation'
            ),
            hovermode='closest'
        )

        return fig
