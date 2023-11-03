import pytest 
import logging 
import os 
import joblib 
import json 
from predictionService.predictor import predictor
import pandas as pd


incorrect_range = {
        "gender": 'trans',    # incorrect
        "age": 49,
        "sleep_duration":6.0,
        "sleep_quality":6,
        "physical_activity_level":90,
        "stress_level":8,
        "bmi_category":6, # incorrect
        "heart_rate":75,
        "systolic":140,
        "diastolic":95
    }

correct_range = {
        "gender": 1,
        "age": 49,
        "sleep_duration":6.0,
        "sleep_quality":6,
        "physical_activity_level":90,
        "stress_level":8,
        "bmi_category":2,
        "heart_rate":75,
        "systolic":140,
        "diastolic":95
        }
    
incorrect_cols = {
        "gender_": 1,
        "age_": 49,
        "sleep duration":6.0,
        "sleep quality":6,
        "physical activity_level":90,
        "stress level":8,
        "bmi category":2,
        "heart rate":75,
        "systolic":140,
        "diastolic":95}
   

TARGET_range = ['Sleep Apnea', 'Normal', 'Insomnia']

def test_incorrect_range(data= pd.DataFrame(incorrect_range, columns=incorrect_range.keys(), index=[0])):
    res = predictor(data)
    assert res == 'ValueError'

def test_correct_range(data= pd.DataFrame(correct_range, columns=correct_range.keys(), index=[0])):
    res = predictor(data)
    assert res[0] in TARGET_range
 
def test_incorrect_col_range(data= pd.DataFrame(incorrect_cols, columns=incorrect_cols.keys(), index=[0])):
    res = predictor(data)
    assert res == 'ValueError'