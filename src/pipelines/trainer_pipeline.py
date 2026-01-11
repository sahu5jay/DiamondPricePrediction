import pandas as pd
import os 
from src.components.data_ingestion import DataIngestion
from src.logger import logging
from src.exception import CustomException
import sys

if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initate_data_ingestion()
    print(train_data_path, test_data_path)