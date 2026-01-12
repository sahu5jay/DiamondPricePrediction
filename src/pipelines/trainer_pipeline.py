import pandas as pd
import os 
from src.components.data_ingestion import DataIngestion
from src.logger import logging
from src.exception import CustomException
import sys
from src.components.data_transformation import DataTransformation

if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, obj_path = data_transformation.ingitate_data_transformation(train_data_path, test_data_path)
    logging.info('Data transformation train, test array')