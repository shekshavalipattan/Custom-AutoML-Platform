import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from typing import Tuple, List, Optional, Dict, Any
import warnings


class DataHandler:
    """
    Handles data preprocessing tasks including:
    - Loading data
    - Handling missing values
    - Feature encoding
    - Feature scaling
    - Train-test splitting
    """

    def __init__(self):
        """Initialize the DataHandler."""
        self.categorical_cols = None
        self.numerical_cols = None
        self.target_encoder = None

    def identify_potential_targets(self, data: pd.DataFrame) -> List[str]:
        """
        Identify columns that could potentially be target variables.

        Parameters:
            data: DataFrame containing the data

        Returns:
            List of column names that could be target variables
        """
        # Get numerical and categorical columns
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # For categorical, only consider those with few unique values
        potential_categorical_targets = [col for col in categorical_cols
                                         if data[col].nunique() < max(10, len(data) // 20)]

        # Combine potential targets
        potential_targets = numerical_cols + potential_categorical_targets

        return potential_targets

    def preprocess_data(self,
                        data: pd.DataFrame,
                        target_column: str,
                        preprocessor: Optional[ColumnTransformer] = None,
                        include_preprocessing: bool = True) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
        """
        Preprocess the data for model training.

        Parameters:
            data: DataFrame containing the data
            target_column: Name of the target column
            preprocessor: Optional preprocessor for reusing in predictions
            include_preprocessing: Whether to include preprocessing steps

        Returns:
            X: Preprocessed features
            y: Target values
            preprocessor: Fitted column transformer for future use
        """
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column].values

        # If no preprocessing needed, return as is
        if not include_preprocessing:
            return X.values, y, None

        # If preprocessor is provided, use it
        if preprocessor is not None:
            X_processed = preprocessor.transform(X)
            return X_processed, y, preprocessor

        # Identify numerical and categorical columns
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Create preprocessing pipelines
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ],
            remainder='drop'  # Drop other columns
        )

        # Apply preprocessing
        X_processed = preprocessor.fit_transform(X)

        # Encode target if categorical
        if data[target_column].dtype == 'object' or data[target_column].dtype.name == 'category':
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y)

        return X_processed, y, preprocessor

    def get_feature_names(self, data: pd.DataFrame, target_column: str) -> List[str]:
        """
        Get feature names after preprocessing.

        Parameters:
            data: Original DataFrame
            target_column: Name of the target column

        Returns:
            List of feature names
        """
        # Get feature columns (excluding target)
        features = data.drop(columns=[target_column])

        # If no categorical columns, return numerical columns
        if not self.categorical_cols:
            return features.columns.tolist()

        # Try to get feature names (might need adjustment based on sklearn version)
        try:
            numerical_cols = self.numerical_cols or []
            categorical_cols = self.categorical_cols or []

            # For categorical features, expand with one-hot encoding
            feature_names = []

            # Add numerical columns
            feature_names.extend(numerical_cols)

            # Add categorical columns with one-hot encoding
            for col in categorical_cols:
                unique_values = features[col].dropna().unique()
                for val in unique_values:
                    feature_names.append(f"{col}_{val}")

            return feature_names
        except:
            # Fallback: return simplified feature names
            return [f"feature_{i}" for i in range(features.shape[1])]
