import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.Exceptions import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_trains, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for model_name, model in models.items():
            para = param.get(model_name, {})
            if para:
                gs = GridSearchCV(model, para, cv=3, scoring="r2", n_jobs=-1)
                gs.fit(X_trains, y_train)
                model = gs.best_estimator_
            else:
                model.fit(X_trains, y_train)
            y_train_pred = model.predict(X_trains)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
            print(f"{model_name}: Train R2 = {train_model_score:.4f}, Test R2 = {test_model_score:.4f}")
        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)