import joblib
import json 
import os 
import argparse
import pandas as pd 
from src.getData import read_params

# from predictionTansformer import _transformer

import warnings
warnings.filterwarnings('ignore')




def predictor(df):
    try:
        config = get_args_config()
        
        prod_model_path = config['ml_flow_config']['production_model_path']

        df = df.replace({'gender': {'Male': 1, 'Female' : 0}})
    
        df = df.replace({'bmi_category': {'Normal': 0, 'Normal Weight' : 1, 'Overweight' : 2, 'Obese' :3}})

        # data = pd.DataFrame(df, columns=df.keys(), index=[0]).copy()

        # data_for_pred =  _transformer(data, config)

        prod_model = joblib.load(prod_model_path)

        pred = prod_model.predict(df)

        # val = round(pred.tolist()[0], 2)
        
        return pred

        

    except Exception as e:
        print(e)
        # error ={"error": "Something went wrong try again"}
        error = {"error": e}
        return f"Try Again"



def get_args_config():
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    config = read_params(config_path=parsed_args.config)

    return config