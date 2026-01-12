import os 
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.utils import save_object, evaluate_model

from dataclasses import dataclass

@dataclass
class ModelTrinerConfig:
    trained_model_file_path = os.path.join('artifacts','model.plk')

class ModelTrainer:
    def __init__(self):
        self.trained_model_file_path = ModelTrinerConfig()

    def initiate_model_trainer(self,train_arr, test_arr):
        try:
            logging.info("spliting train, test data into dependent and independent variable")

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,:-1],
                test_arr[:,:-1],
                test_arr[:,:-1]
            )

            models = {
                'LinearRegression':LinearRegression(),
                'Ridge': Ridge(), 
                'Lasso':Lasso(),
                'ElasticNet': ElasticNet(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'RandomForestRegressor':RandomForestRegressor(),
                'KNeighborsRegressor':KNeighborsRegressor()
            }

            model_report:dict=evaluate_model(models=models,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
            print("\n============================================================")
            logging.info("model report:{model_report}")

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]


            print(f'Best model found, model name : {best_model_name} model score {best_model_score}')
            print("\n=========================================================================")
            logging.info(f'Best model found, model name : {best_model_name} model score {best_model_score}')
            

            save_object(
                file_path= self.trained_model_file_path.trained_model_file_path,
                obj = best_model
            )


        except Exception as e:
            raise CustomException(e, sys)
