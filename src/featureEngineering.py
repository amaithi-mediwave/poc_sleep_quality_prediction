import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder



from getData import get_data_frame, read_params

# set_config(transform_output="pandas")

def feature_engineering(config_path):
    
    config = read_params(config_path)

    df = get_data_frame(config_path)
    
    # filling the missing values in Sleep Disorder
    df['sleep_disorder'].fillna('Normal', inplace=True)
    
    # drop the un wanted values in BP
    df.drop(df[df['blood_pressure'] == '_RARE_'].index, inplace=True)
    
    #  GENDER Variable
    df = df.replace({'gender': {'Male': 1, 'Female' : 0}})
    # df = pd.get_dummies(df, columns=['Gender'], dtype=int, drop_first=True)
    
    # BMI Category Variable
    df = df.replace({'bmi_category': {'Normal': 0, 'Normal Weight' : 1, 'Overweight' : 2, 'Obese' :3}})
    # df = pd.get_dummies(df, columns=['BMI Category'], dtype=int, drop_first=True)
    
    # Encoding the Target Variable
    # le = LabelEncoder()
    # df['sleep_disorder'] = le.fit_transform(df['Sleep Disorder'])
    
    # Handling the BP Variable
    df[['systolic', 'diastolic']] = df['blood_pressure'].str.split('/', expand=True).astype(int)
    df.drop(columns=['blood_pressure'], inplace=True)
    
    data_path = config['data_source']['local_data_source']['pre_processed_data']
    
    df.to_csv(data_path, index=False)
    
    
    
    
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    feature_engineering(parsed_args.config)