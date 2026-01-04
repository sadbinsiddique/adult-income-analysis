"""
Prediction Pipeline
Handles inference with trained model
"""
import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from joblib import load

from src.exception import CustomException
from src.logger import logging


@dataclass
class PredictPipelineConfig:
    """Configuration for prediction pipeline"""
    model_path: str = os.path.join('notebook', 'models', 'model.pkl')


class PredictPipeline:
    """
    Prediction Pipeline for making predictions with trained model
    """
    def __init__(self, config: PredictPipelineConfig = None):
        if config is None:
            self.config = PredictPipelineConfig()
        else:
            self.config = config
        self.model = None
        logging.info("PredictPipeline initialized")

    def load_model(self):
        """Load the trained model"""
        try:
            self.model = load(self.config.model_path)
            logging.info(f"Model loaded from {self.config.model_path}")
            return self.model
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on input features
        
        Args:
            features: DataFrame of input features (already preprocessed)
            
        Returns:
            np.ndarray: Predicted labels (0 = <=50K, 1 = >50K)
        """
        try:
            if self.model is None:
                self.load_model()
            
            predictions = self.model.predict(features)
            logging.info(f"Made predictions for {len(predictions)} samples")
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}")
            raise CustomException(e, sys)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Get decision function scores (confidence scores)
        
        Args:
            features: DataFrame of input features
            
        Returns:
            np.ndarray: Decision function scores
        """
        try:
            if self.model is None:
                self.load_model()
            
            scores = self.model.decision_function(features)
            logging.info(f"Generated scores for {len(scores)} samples")
            
            return scores
            
        except Exception as e:
            logging.error(f"Error getting decision scores: {str(e)}")
            raise CustomException(e, sys)

    def predict_with_label(self, features: pd.DataFrame) -> list:
        """
        Make predictions and return human-readable labels
        
        Args:
            features: DataFrame of input features
            
        Returns:
            list: Predicted income labels ('<=50K' or '>50K')
        """
        try:
            predictions = self.predict(features)
            labels = ['<=50K' if p == 0 else '>50K' for p in predictions]
            return labels
            
        except Exception as e:
            logging.error(f"Error in prediction with labels: {str(e)}")
            raise CustomException(e, sys)


class CustomData:
    """
    Class to handle custom input data for prediction
    """
    def __init__(
        self,
        age: float,
        workclass: str,
        fnlwgt: float,
        education: str,
        education_num: float,
        marital_status: str,
        occupation: str,
        relationship: str,
        race: str,
        sex: str,
        capital_gain: float,
        capital_loss: float,
        hours_per_week: float,
        native_country: str
    ):
        self.age = age
        self.workclass = workclass
        self.fnlwgt = fnlwgt
        self.education = education
        self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.native_country = native_country

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Convert custom data to DataFrame
        
        Returns:
            pd.DataFrame: Input data as DataFrame
        """
        try:
            data_dict = {
                'age': [self.age],
                'workclass': [self.workclass],
                'fnlwgt': [self.fnlwgt],
                'education': [self.education],
                'education-num': [self.education_num],
                'marital-status': [self.marital_status],
                'occupation': [self.occupation],
                'relationship': [self.relationship],
                'race': [self.race],
                'sex': [self.sex],
                'capital-gain': [self.capital_gain],
                'capital-loss': [self.capital_loss],
                'hours-per-week': [self.hours_per_week],
                'native-country': [self.native_country]
            }
            
            return pd.DataFrame(data_dict)
            
        except Exception as e:
            logging.error(f"Error creating DataFrame: {str(e)}")
            raise CustomException(e, sys)
