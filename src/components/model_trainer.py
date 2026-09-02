import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.Exceptions import CustomException
from src.loggers import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor()
            }

            params = {
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64]
                },
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05],
                    "subsample": [0.6, 0.8, 0.9],
                    "n_estimators": [8, 16, 32, 64]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05],
                    "n_estimators": [8, 16, 32, 64]
                },
                "CatBoostRegressor": {
                    "depth": [6, 8],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5],
                    "n_estimators": [8, 16, 32, 64]
                },
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9]
                }
            }

            model_report = evaluate_models(
                X_trains=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]
            best_params = params.get(best_model_name, {})

            if best_params:
                from sklearn.model_selection import GridSearchCV
                gs = GridSearchCV(
                    best_model,
                    best_params,
                    cv=3,
                    scoring="r2",
                    n_jobs=-1
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                best_model.fit(X_train, y_train)

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"Final R2 Score: {r2_square}")
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)