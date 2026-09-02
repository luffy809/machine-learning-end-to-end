import os
import sys

import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.Exceptions import CustomException
from src.loggers import logging

from src.components.Data_transformation import DataTransformation
from src.components.model_Trainer import ModelTrainer


# DATA INGESTION CONFIGURATION

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(
        "artifacts",
        "train.csv"
    )
    test_data_path: str = os.path.join(
        "artifacts",
        "test.csv"
    )
    raw_data_path: str = os.path.join(
        "artifacts",
        "raw.csv"
    )



#DATA INGESTION CLASS
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    #INITIATE DATA INGESTION
    def initiate_data_ingestion(self):
        logging.info(
            "Entered the data ingestion method or component"
        )

        try:

            #READ DATASET
            df = pd.read_csv(
                r"E:\Machine project end-to-end\Notebook\Notebook\Data\StudentsPerformance.csv"
            )
            logging.info(
                "Read the dataset as DataFrame"
            )

            # CREATE ARTIFACTS DIRECTORY
            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.train_data_path
                ),
                exist_ok=True
            )

        
            # SAVE RAW DATA
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )
            logging.info(
                "Raw data saved successfully"
            )

            # TRAIN TEST SPLIT
            logging.info(
                "Train test split initiated"
            )
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # SAVE TRAIN DATA
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            # SAVE TEST DATA
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )
            logging.info(
                "Train and test data saved successfully"
            )
            logging.info(
                "Ingestion of the data is completed"
            )
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:

            logging.error(
                "Exception occurred during data ingestion"
            )

            raise CustomException(
                e,
                sys
            )

if __name__ == "__main__":

    try:

        # DATA INGESTION

        obj = DataIngestion()

        train_data, test_data = (
            obj.initiate_data_ingestion()
        )

        print("\nData Ingestion Completed")
        print("Train Data:", train_data)
        print("Test Data :", test_data)

        # DATA TRANSFORMATION

        data_transformation = DataTransformation()

        train_arr, test_arr, _ = (
            data_transformation.initiate_data_transformation(
                train_data,
                test_data
            )
        )

        print("\nData Transformation Completed")

        # MODEL TRAINING

        modeltrainer = ModelTrainer()

        model_score = (
            modeltrainer.initiate_model_trainer(
                train_arr,
                test_arr
            )
        )

        print("\nModel Training Completed")
        print("Model Score:", model_score)

    except Exception as e:

        print("\nError occurred:")
        print(e)