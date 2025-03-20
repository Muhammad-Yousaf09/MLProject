import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            # Use raw string (r'path') or forward slashes for Windows paths
            data_path = r'notebook/data/stud.csv'
            df = pd.read_csv(data_path)
            logging.info(f"Data loaded successfully from {data_path}")

            # Create the artifacts folder if it doesn't exist
            artifacts_folder = os.path.dirname(self.ingestion_config.train_data_path)
            if not os.path.exists(artifacts_folder):
                os.makedirs(artifacts_folder)
                logging.info(f"Created folder: {artifacts_folder}")

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split data
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)

 
if __name__ == "__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
    

  
