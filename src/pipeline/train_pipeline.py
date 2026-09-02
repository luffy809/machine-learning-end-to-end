from src.components.Data_ingestion import DataIngestion
from src.components.Data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def train_pipeline():
    # Data Ingestion
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    print("Data Ingestion Completed")
    print(f"Train Data: {train_data}")
    print(f"Test Data: {test_data}")

    # Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = (
        data_transformation.initiate_data_transformation(
            train_data,
            test_data
        )
    )

    print("\nData Transformation Completed")
    print(f"Preprocessor: {preprocessor_path}")
    print(f"Train Shape: {train_arr.shape}")
    print(f"Test Shape: {test_arr.shape}")

    # Model Training
    model_trainer = ModelTrainer()
    model_score = model_trainer.initiate_model_trainer(
        train_arr,
        test_arr
    )

    print("\nModel Training Completed")
    print(f"Model R2 Score: {model_score}")


if __name__ == "__main__":
    train_pipeline()