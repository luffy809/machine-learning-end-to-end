#a utils file/folder inside src is used for common helper functions that are needed in multiple parts of the project.

import os 
import sys

import numpy 
import pandas as pd 
import dill
import pickle 
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


from src.Exceptions import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok = True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_trains,y_train,X_test,y_test,models,param):
    try:
        report={}
        
        for i in range(len(list(models))):
            model= list(model.values())[i]
            para = param[list(models.keys())[i]]
            
            gs = GridSearchCV(model,para,cv=3) # use the gridsearchCV
            gs.fit(X_train,y_train)
            
            model.set_params(**gs.best_params_) # set the parameter of model
            model.fit(X_trains,y_train)  # train model 

            y_train_pred = model.predict(X_trains) # predict the training model 
            y_test_pred = model.predict(X_test) # predict thee test model 
            
            train_model_score = r2_score(y_train, y_train_pred)# give the r2 score of train model
            test_model_score  = r2_score(y_test, y_test_pred) # give the r2 score of test model 
            
            report[list(model.keys())[i]] = test_model_score
            
        return report 
    except Exception as e:
        raise CustomException(e,sys)
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)