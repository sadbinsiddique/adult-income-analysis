"""
Data Transformation Component
Handles preprocessing pipeline for adult income data
"""
import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from joblib import dump, load

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import skew

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation"""
    preprocessor_path: str = os.path.join('notebook', 'models', 'preprocessor.pkl')
    skew_threshold: float = 0.5


class DataTransformation:
    """
    Data Transformation class for preprocessing features
    """
    def __init__(self, config: DataTransformationConfig = None):
        if config is None:
            self.config = DataTransformationConfig()
        else:
            self.config = config
        logging.info("DataTransformation initialized")

    def get_data_transformer(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create preprocessing pipeline based on column types
        
        Args:
            X: Feature DataFrame to analyze
            
        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        try:
            # Identify column types
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            logging.info(f"Numeric columns: {numeric_cols}")
            logging.info(f"Categorical columns: {categorical_cols}")
            
            # Numeric transformer with scaling
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            
            # Categorical transformer with one-hot encoding
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            # Combine transformers
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ],
                remainder='passthrough'
            )
            
            logging.info("Preprocessor pipeline created successfully")
            return preprocessor
            
        except Exception as e:
            logging.error(f"Error creating data transformer: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """
        Apply data transformation to train and test sets
        
        Args:
            X_train: Training features
            X_test: Testing features
            
        Returns:
            tuple: (X_train_transformed, X_test_transformed, preprocessor)
        """
        logging.info("Starting data transformation")
        
        try:
            preprocessor = self.get_data_transformer(X_train)
            
            # Fit and transform training data
            X_train_transformed = preprocessor.fit_transform(X_train)
            logging.info(f"X_train transformed shape: {X_train_transformed.shape}")
            
            # Transform test data
            X_test_transformed = preprocessor.transform(X_test)
            logging.info(f"X_test transformed shape: {X_test_transformed.shape}")
            
            # Save preprocessor
            os.makedirs(os.path.dirname(self.config.preprocessor_path), exist_ok=True)
            dump(preprocessor, self.config.preprocessor_path)
            logging.info(f"Preprocessor saved to {self.config.preprocessor_path}")
            
            return X_train_transformed, X_test_transformed, preprocessor
            
        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)

    def load_preprocessor(self) -> ColumnTransformer:
        """
        Load saved preprocessor
        
        Returns:
            ColumnTransformer: Loaded preprocessor
        """
        try:
            preprocessor = load(self.config.preprocessor_path)
            logging.info(f"Preprocessor loaded from {self.config.preprocessor_path}")
            return preprocessor
        except Exception as e:
            logging.error(f"Error loading preprocessor: {str(e)}")
            raise CustomException(e, sys)
