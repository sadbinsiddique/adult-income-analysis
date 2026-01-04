"""
Training Pipeline
Orchestrates data ingestion, transformation, and model training
"""
import sys
from src.components.data_ingestioms import DataIngestion, DataIngestionConfig
from src.components.data_transfromation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    """
    Training Pipeline class that orchestrates the full training process
    """
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        logging.info("TrainPipeline initialized")

    def run_pipeline(self) -> dict:
        """
        Execute the complete training pipeline
        
        Returns:
            dict: Training results including metrics and model info
        """
        logging.info("=" * 50)
        logging.info("Starting Training Pipeline")
        logging.info("=" * 50)
        
        try:
            # Step 1: Data Ingestion
            logging.info("Step 1: Data Ingestion")
            X_train, X_test, y_train, y_test = self.data_ingestion.initiate_data_ingestion()
            
            # Step 2: Model Training (data is already preprocessed)
            logging.info("Step 2: Model Training with KFold CV")
            model, train_metrics, test_metrics, cv_scores = self.model_trainer.initiate_model_trainer(
                X_train, X_test, y_train, y_test
            )
            
            # Compile results
            results = {
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_scores': cv_scores,
                'data_shapes': {
                    'X_train': X_train.shape,
                    'X_test': X_test.shape,
                    'y_train': y_train.shape,
                    'y_test': y_test.shape
                }
            }
            
            logging.info("=" * 50)
            logging.info("Training Pipeline Completed Successfully")
            logging.info(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
            logging.info(f"Final Test F1-Score: {test_metrics['f1']:.4f}")
            logging.info(f"Final Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
            logging.info("=" * 50)
            
            return results
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise CustomException(e, sys)


def main():
    """Main function to run training pipeline"""
    pipeline = TrainPipeline()
    results = pipeline.run_pipeline()
    
    print("\n" + "=" * 50)
    print("Training Results Summary")
    print("=" * 50)
    print(f"CV Mean Train Accuracy: {results['cv_scores']['cv_mean_train']:.4f}")
    print(f"CV Mean Val Accuracy: {results['cv_scores']['cv_mean_val']:.4f}")
    print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Test Precision: {results['test_metrics']['precision']:.4f}")
    print(f"Test Recall: {results['test_metrics']['recall']:.4f}")
    print(f"Test F1-Score: {results['test_metrics']['f1']:.4f}")
    print(f"Test ROC-AUC: {results['test_metrics']['roc_auc']:.4f}")
    print("=" * 50)
    
    return results


if __name__ == "__main__":
    main()
