"""
Model Trainer Component
Handles training of LinearSVC model with KFold cross-validation
Based on model_selectionv2.ipynb
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from joblib import dump, load

from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    """Configuration for model training"""
    model_path: str = os.path.join('notebook', 'models', 'model.pkl')
    cv_splits: int = 10
    random_state: int = 3327
    max_iter: int = 5000


class ModelTrainer:
    """
    Model Trainer class using LinearSVC with KFold cross-validation
    """
    def __init__(self, config: ModelTrainerConfig = None):
        if config is None:
            self.config = ModelTrainerConfig()
        else:
            self.config = config
        logging.info("ModelTrainer initialized")

    def get_model(self):
        """
        Get LinearSVC model instance
        
        Returns:
            LinearSVC: Model instance
        """
        return LinearSVC(
            dual=False,
            random_state=self.config.random_state,
            max_iter=self.config.max_iter
        )

    def train_with_cv(self, X_train, y_train) -> tuple:
        """
        Train model with KFold cross-validation
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            tuple: (train_scores, val_scores, cv_time)
        """
        logging.info(f"Starting KFold CV with {self.config.cv_splits} splits")
        
        try:
            model = self.get_model()
            kf = KFold(
                n_splits=self.config.cv_splits,
                shuffle=True,
                random_state=self.config.random_state
            )
            
            train_scores = []
            val_scores = []
            
            start = time.time()
            
            for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                if isinstance(X_train, pd.DataFrame):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                else:
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    
                if isinstance(y_train, pd.Series):
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                else:
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]

                clf = clone(model)
                clf.fit(X_tr, y_tr)
                
                # Training accuracy
                y_train_pred_fold = clf.predict(X_tr)
                train_acc = accuracy_score(y_tr, y_train_pred_fold)
                train_scores.append(train_acc)
                
                # Validation accuracy
                y_val_pred = clf.predict(X_val)
                val_acc = accuracy_score(y_val, y_val_pred)
                val_scores.append(val_acc)
                
                logging.info(f"Fold {i+1}: Train={train_acc:.4f}, Val={val_acc:.4f}")

            cv_time = time.time() - start
            
            cv_mean_train = np.mean(train_scores)
            cv_mean_val = np.mean(val_scores)
            
            logging.info(f"CV completed in {cv_time:.2f}s")
            logging.info(f"Mean Train Accuracy: {cv_mean_train:.4f}")
            logging.info(f"Mean Val Accuracy: {cv_mean_val:.4f}")
            
            return train_scores, val_scores, cv_time
            
        except Exception as e:
            logging.error(f"Error in CV training: {str(e)}")
            raise CustomException(e, sys)

    def train_final_model(self, X_train, y_train):
        """
        Train final model on full training data
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            LinearSVC: Trained model
        """
        logging.info("Training final model on full training data")
        
        try:
            model = self.get_model()
            model.fit(X_train, y_train)
            logging.info("Final model trained successfully")
            return model
            
        except Exception as e:
            logging.error(f"Error training final model: {str(e)}")
            raise CustomException(e, sys)

    def evaluate_model(self, model, X, y, dataset_name: str = "Dataset") -> dict:
        """
        Evaluate model on given data
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            dataset_name: Name for logging
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            y_pred = model.predict(X)
            y_scores = model.decision_function(X)
            
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1': f1_score(y, y_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(y, y_pred)
            }
            
            # ROC AUC
            fpr, tpr, _ = roc_curve(y, y_scores)
            metrics['roc_auc'] = auc(fpr, tpr)
            metrics['fpr'] = fpr
            metrics['tpr'] = tpr
            
            logging.info(f"{dataset_name} Metrics:")
            logging.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logging.info(f"  Precision: {metrics['precision']:.4f}")
            logging.info(f"  Recall: {metrics['recall']:.4f}")
            logging.info(f"  F1-Score: {metrics['f1']:.4f}")
            logging.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error evaluating model: {str(e)}")
            raise CustomException(e, sys)

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test) -> tuple:
        """
        Complete model training pipeline
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training targets
            y_test: Testing targets
            
        Returns:
            tuple: (model, train_metrics, test_metrics, cv_scores)
        """
        logging.info("Starting model training pipeline")
        
        try:
            # Cross-validation
            train_scores, val_scores, cv_time = self.train_with_cv(X_train, y_train)
            
            # Train final model
            model = self.train_final_model(X_train, y_train)
            
            # Evaluate on train and test
            train_metrics = self.evaluate_model(model, X_train, y_train, "Training")
            test_metrics = self.evaluate_model(model, X_test, y_test, "Test")
            
            # Log gap analysis
            gap = train_metrics['accuracy'] - test_metrics['accuracy']
            logging.info(f"Train-Test Gap: {gap:.4f}")
            
            # Save model
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            dump(model, self.config.model_path)
            logging.info(f"Model saved to {self.config.model_path}")
            
            cv_scores = {
                'train_scores': train_scores,
                'val_scores': val_scores,
                'cv_time': cv_time,
                'cv_mean_train': np.mean(train_scores),
                'cv_mean_val': np.mean(val_scores)
            }
            
            return model, train_metrics, test_metrics, cv_scores
            
        except Exception as e:
            logging.error(f"Error in model training pipeline: {str(e)}")
            raise CustomException(e, sys)

    def load_model(self):
        """
        Load saved model
        
        Returns:
            LinearSVC: Loaded model
        """
        try:
            model = load(self.config.model_path)
            logging.info(f"Model loaded from {self.config.model_path}")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise CustomException(e, sys)
