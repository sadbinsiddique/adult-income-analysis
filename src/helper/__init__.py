import os
import sys
import glob
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from joblib import load, dump

from src.exception import CustomException
from src.logger import logging


class DataHelper:
    @staticmethod
    def read_csv(file_path: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(file_path)
            logging.info(f"Successfully read CSV: {file_path}, shape: {data.shape}")
            return data
        except Exception as e:
            logging.error(f"Error reading CSV {file_path}: {str(e)}")
            raise CustomException(e, sys)
    
    @staticmethod
    def write_csv(data: pd.DataFrame, file_path: str, index: bool = False) -> str:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            data.to_csv(file_path, index=index)
            logging.info(f"Successfully saved CSV: {file_path}")
            return file_path
        except Exception as e:
            logging.error(f"Error writing CSV {file_path}: {str(e)}")
            raise CustomException(e, sys)
    
    @staticmethod
    def load_model(model_path: str) -> Any:
        try:
            model = load(model_path)
            logging.info(f"Successfully loaded model: {model_path}")
            return model
        except Exception as e:
            logging.error(f"Error loading model {model_path}: {str(e)}")
            raise CustomException(e, sys)
    
    @staticmethod
    def save_model(model: Any, model_path: str) -> str:
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            dump(model, model_path)
            logging.info(f"Successfully saved model: {model_path}")
            return model_path
        except Exception as e:
            logging.error(f"Error saving model {model_path}: {str(e)}")
            raise CustomException(e, sys)


class LogHelper:
    LOG_DIR = "logs"
    
    @classmethod
    def get_latest_log_file(cls) -> Optional[str]:
        try:
            log_files = glob.glob(os.path.join(cls.LOG_DIR, "log_*.log"))
            if not log_files:
                return None
            return max(log_files, key=os.path.getctime)
        except Exception as e:
            logging.error(f"Error getting latest log file: {str(e)}")
            return None
    
    @classmethod
    def read_logs(cls, num_lines: int = 50) -> Dict[str, Any]:
        try:
            log_file = cls.get_latest_log_file()
            
            if not log_file or not os.path.exists(log_file):
                return {"logs": [], "file": None, "error": "No log file found"}
            
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            recent_lines = lines[-num_lines:] if len(lines) > num_lines else lines
            
            return {
                "logs": [line.strip() for line in recent_lines if line.strip()],
                "file": os.path.basename(log_file),
                "total_lines": len(lines)
            }
        except Exception as e:
            return {"logs": [], "file": None, "error": str(e)}
    
    @classmethod
    def get_all_log_files(cls) -> List[str]:
        try:
            log_files = glob.glob(os.path.join(cls.LOG_DIR, "log_*.log"))
            return sorted(log_files, key=os.path.getctime, reverse=True)
        except Exception:
            return []


class InputDataHelper:
    """
    Helper class for processing input data for adult income prediction
    Handles label encoding to match training data format
    """
    
    # Label encoding mappings (must match LabelEncoder from preprocessing)
    # These are sorted alphabetically as LabelEncoder does
    WORKCLASS_ENCODING = {
        'Federal-gov': 0, 'Local-gov': 1, 'Never-worked': 2, 'Private': 3,
        'Self-emp-inc': 4, 'Self-emp-not-inc': 5, 'State-gov': 6, 'Without-pay': 7
    }
    
    EDUCATION_ENCODING = {
        '10th': 0, '11th': 1, '12th': 2, '1st-4th': 3, '5th-6th': 4, '7th-8th': 5,
        '9th': 6, 'Assoc-acdm': 7, 'Assoc-voc': 8, 'Bachelors': 9, 'Doctorate': 10,
        'HS-grad': 11, 'Masters': 12, 'Preschool': 13, 'Prof-school': 14, 'Some-college': 15
    }
    
    MARITAL_STATUS_ENCODING = {
        'Divorced': 0, 'Married-AF-spouse': 1, 'Married-civ-spouse': 2,
        'Married-spouse-absent': 3, 'Never-married': 4, 'Separated': 5, 'Widowed': 6
    }
    
    OCCUPATION_ENCODING = {
        'Adm-clerical': 0, 'Armed-Forces': 1, 'Craft-repair': 2, 'Exec-managerial': 3,
        'Farming-fishing': 4, 'Handlers-cleaners': 5, 'Machine-op-inspct': 6,
        'Other-service': 7, 'Priv-house-serv': 8, 'Prof-specialty': 9,
        'Protective-serv': 10, 'Sales': 11, 'Tech-support': 12, 'Transport-moving': 13
    }
    
    RELATIONSHIP_ENCODING = {
        'Husband': 0, 'Not-in-family': 1, 'Other-relative': 2,
        'Own-child': 3, 'Unmarried': 4, 'Wife': 5
    }
    
    RACE_ENCODING = {
        'Amer-Indian-Eskimo': 0, 'Asian-Pac-Islander': 1, 'Black': 2, 'Other': 3, 'White': 4
    }
    
    GENDER_ENCODING = {'Female': 0, 'Male': 1}
    
    COUNTRY_ENCODING = {
        'Cambodia': 0, 'Canada': 1, 'China': 2, 'Columbia': 3, 'Cuba': 4,
        'Dominican-Republic': 5, 'Ecuador': 6, 'El-Salvador': 7, 'England': 8,
        'France': 9, 'Germany': 10, 'Greece': 11, 'Guatemala': 12, 'Haiti': 13,
        'Holand-Netherlands': 14, 'Honduras': 15, 'Hong': 16, 'Hungary': 17,
        'India': 18, 'Iran': 19, 'Ireland': 20, 'Italy': 21, 'Jamaica': 22,
        'Japan': 23, 'Laos': 24, 'Mexico': 25, 'Nicaragua': 26,
        'Outlying-US(Guam-USVI-etc)': 27, 'Peru': 28, 'Philippines': 29,
        'Poland': 30, 'Portugal': 31, 'Puerto-Rico': 32, 'Scotland': 33,
        'South': 34, 'Taiwan': 35, 'Thailand': 36, 'Trinadad&Tobago': 37,
        'United-States': 38, 'Vietnam': 39, 'Yugoslavia': 40
    }
    
    # Form display values
    WORKCLASS_VALUES = list(WORKCLASS_ENCODING.keys())
    EDUCATION_VALUES = list(EDUCATION_ENCODING.keys())
    MARITAL_STATUS_VALUES = list(MARITAL_STATUS_ENCODING.keys())
    OCCUPATION_VALUES = list(OCCUPATION_ENCODING.keys())
    RELATIONSHIP_VALUES = list(RELATIONSHIP_ENCODING.keys())
    RACE_VALUES = list(RACE_ENCODING.keys())
    SEX_VALUES = list(GENDER_ENCODING.keys())
    COUNTRY_VALUES = list(COUNTRY_ENCODING.keys())
    
    @classmethod
    def encode_value(cls, value: str, encoding_map: Dict) -> int:
        """Encode a categorical value, return 0 if not found"""
        return encoding_map.get(value, 0)
    
    @classmethod
    def create_input_dataframe(cls, form_data: Dict) -> pd.DataFrame:
        """
        Create properly encoded DataFrame matching training data format
        """
        try:
            # Get raw values
            age = float(form_data.get('age', 30))
            hours_per_week = float(form_data.get('hours_per_week', 40))
            
            # Calculate is_full_time (same as preprocessing)
            is_full_time = 1 if hours_per_week >= 40 else 0
            
            # Scale numeric features (using approximate mean/std from training)
            # age: mean ~38.6, std ~13.6
            # hours-per-week: mean ~40.4, std ~12.3
            # is_full_time: mean ~0.67, std ~0.47
            age_scaled = (age - 38.6) / 13.6
            hours_scaled = (hours_per_week - 40.4) / 12.3
            is_full_time_scaled = (is_full_time - 0.67) / 0.47
            
            data_dict = {
                'age': [age_scaled],
                'workclass': [cls.encode_value(form_data.get('workclass', 'Private'), cls.WORKCLASS_ENCODING)],
                'fnlwgt': [int(form_data.get('fnlwgt', 189778))],
                'education': [cls.encode_value(form_data.get('education', 'HS-grad'), cls.EDUCATION_ENCODING)],
                'educational-num': [int(form_data.get('education_num', 10))],
                'marital-status': [cls.encode_value(form_data.get('marital_status', 'Never-married'), cls.MARITAL_STATUS_ENCODING)],
                'occupation': [cls.encode_value(form_data.get('occupation', 'Other-service'), cls.OCCUPATION_ENCODING)],
                'relationship': [cls.encode_value(form_data.get('relationship', 'Not-in-family'), cls.RELATIONSHIP_ENCODING)],
                'race': [cls.encode_value(form_data.get('race', 'White'), cls.RACE_ENCODING)],
                'gender': [cls.encode_value(form_data.get('sex', 'Male'), cls.GENDER_ENCODING)],
                'capital-gain': [int(form_data.get('capital_gain', 0))],
                'capital-loss': [int(form_data.get('capital_loss', 0))],
                'hours-per-week': [hours_scaled],
                'native-country': [cls.encode_value(form_data.get('native_country', 'United-States'), cls.COUNTRY_ENCODING)],
                'is_full_time': [is_full_time_scaled]
            }
            
            df = pd.DataFrame(data_dict)
            logging.info(f"Created encoded input DataFrame with shape: {df.shape}")
            logging.info(f"Input values: {data_dict}")
            return df
            
        except Exception as e:
            logging.error(f"Error creating input DataFrame: {str(e)}")
            raise CustomException(e, sys)
    
    @classmethod
    def get_form_options(cls) -> Dict[str, List[str]]:
        """Get dropdown options for the form"""
        return {
            'workclass': cls.WORKCLASS_VALUES,
            'education': cls.EDUCATION_VALUES,
            'marital_status': cls.MARITAL_STATUS_VALUES,
            'occupation': cls.OCCUPATION_VALUES,
            'relationship': cls.RELATIONSHIP_VALUES,
            'race': cls.RACE_VALUES,
            'sex': cls.SEX_VALUES,
            'native_country': cls.COUNTRY_VALUES
        }
