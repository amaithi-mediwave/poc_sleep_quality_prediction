import argparse
import pandas as pd



from getData import get_data_frame, read_params

# set_config(transform_output="pandas")

def hyperparameter_tuning(config_path):
    
    

































    if __name__ == "__main__":
        args = argparse.ArgumentParser()
        args.add_argument("--config", default="params.yaml")
        parsed_args = args.parse_args()
        hyperparameter_tuning(parsed_args.config)