from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import sys,os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from dataclasses import dataclass

@dataclass
class DatatransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DatatransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("inside get_transformaton_method")
            logging.info('data transformation initated')

            df = pd.read_csv(os.path.join('notebook/data','gemstone.csv'))
            X = df.drop(labels=['price'],axis=1)
            Y = df[['price']]

            categorical_cols = X.select_dtypes(include='object').columns.to_list()
            numerical_cols = X.select_dtypes(exclude='object').columns.to_list()

            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Data Transformation initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]

            )

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            logging.info('Data Transformation completed..')
            logging.info("retutning the preprocessor out_side get_transformaton_method")


            return preprocessor
        except Exception as e:
            logging.info('Exception raised in Data Transformation')
            raise CustomException(e,sys)

    def ingitate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            
            logging.info("Train & Test data loaded")

            preprossor_obj = self.get_data_transformation_object()
            logging.info("preprossor_obj obtained-->")

            target = 'price'
            drop_column = [target]

            logging.info('Splitting the data into features and target')

            input_feature_train_df = train_df.drop(columns=drop_column, axis=1)
            target_feature_train_df = train_df[[target]]
            logging.info('Splitting done for the train data')

            

            input_feature_test_df = test_df.drop(columns=drop_column, axis=1)
            target_feature_test_df = test_df[[target]]

            logging.info('Splitting done for the test data')
            input_feature_train_arr = preprossor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprossor_obj.transform(input_feature_test_df)
            logging.info("applying the input featues to the preprocessor pipeline")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprossor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)


        