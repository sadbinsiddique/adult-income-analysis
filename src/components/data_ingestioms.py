"""
Data Ingestion Component
Handles loading of preprocessed data from CSV files
"""
import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion paths"""
    data_dir: str = os.path.join('notebook', 'data')
    train_data_path: str = os.path.join(data_dir, 'X_train_scaled.csv')
    test_data_path: str = os.path.join(data_dir, 'X_test_scaled.csv')
    train_target_path: str = os.path.join(data_dir, 'y_train.csv')
    test_target_path: str = os.path.join(data_dir, 'y_test.csv')


class DataIngestion:
    """
    Data Ingestion class to load preprocessed train/test data
    """
    def __init__(self, config: DataIngestionConfig = None):
        if config is None:
            self.config = DataIngestionConfig()
        else:
            self.config = config
        logging.info("DataIngestion initialized with config")

    def initiate_data_ingestion(self) -> tuple:
        """
        Load preprocessed data from CSV files
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logging.info("Starting data ingestion process")
        
        try:
            # Load feature data
            X_train = pd.read_csv(self.config.train_data_path)
            X_test = pd.read_csv(self.config.test_data_path)
            
            logging.info(f"Loaded X_train shape: {X_train.shape}")
            logging.info(f"Loaded X_test shape: {X_test.shape}")
            
            # Load target data
            y_train = pd.read_csv(self.config.train_target_path)['income']
            y_test = pd.read_csv(self.config.test_target_path)['income']
            
            logging.info(f"Loaded y_train shape: {y_train.shape}")
            logging.info(f"Loaded y_test shape: {y_test.shape}")
            
            logging.info("Data ingestion completed successfully")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise CustomException(e, sys)
