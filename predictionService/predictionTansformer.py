import joblib
import json 
import os 
import sys
  



import argparse
import numpy as np
import pandas as pd 
from pprint import pprint





def _transformer(df):

    # config = gt.read_params(config_path)
    
    df = df.replace({'gender': {'Male': 1, 'Female' : 0}})
    
    df = df.replace({'bmi_category': {'Normal': 0, 'Normal Weight' : 1, 'Overweight' : 2, 'Obese' :3}})

    print(df)
    
    
# BMI Category_Normal Weight

# BMI Category_Obese

# BMI Category_Overweight
    
    
    
    
    
    
    
    #  Age Gender BMI_Category  Sleep_Duration  Quality_of_Sleep  Physical_Activity_Level  Stress_Level  Heart_Rate  Systolic  Diastolic
    #   33   male   Overweight             7.1                 6                       68             5          74       125         85
    
    
    
    
    
    
    
    
    
    
    
    
    

    # df['rest_type'] = df['rest_type'].apply(lister)
    # df['cuisines'] = df['cuisines'].apply(spacer)

    # b = rest_encoder.transform(df['rest_type'])

    # cols = []
    # for idx, val in enumerate(rest_encoder.classes_):
    #     val = f"type_{val}"
    #     cols.append(val)

    # df[cols] = b

    # d_cui = pd.DataFrame(df['cuisines'])

    # d_cui = cuisine_encoder.transform(d_cui)

    # df1 = pd.concat([d_cui, df], axis="columns")

    # df1.drop(['rest_type', 'cuisines'], axis=1, inplace=True)

    # df1 = df1.loc[:,['cuisines_0', 'cuisines_1', 'cuisines_2', 'cuisines_3', 'cuisines_4',
    #     'cuisines_5', 'cuisines_6', 'cuisines_7', 'cuisines_8', 'cuisines_9',
    #     'cuisines_10', 'cuisines_11', 'cost_per_person', 'online_order_Yes',
    #     'book_table_Yes', 'loc_pincode', 'type_Bakery', 'type_Bar',
    #     'type_Beverage Shop', 'type_Bhojanalya', 'type_Cafe',
    #     'type_Casual Dining', 'type_Club', 'type_Confectionery',
    #     'type_Delivery', 'type_Dessert Parlor', 'type_Dhaba',
    #     'type_Fine Dining', 'type_Food Court', 'type_Food Truck',
    #     'type_Irani Cafee', 'type_Kiosk', 'type_Lounge', 'type_Meat Shop',
    #     'type_Mess', 'type_Microbrewery', 'type_Pop Up', 'type_Pub',
    #     'type_Quick Bites', 'type_Sweet Shop', 'type_Takeaway']]
    
    # # print(df1, end="\n\n")
    # print(df1.columns)
    # return df1






   




def lister(val):
    '''
    this function takes the value and return it in a list
    '''
    if ',' in val:
        ls = val.split(', ')
        return ls
    else:
        return [val]
    


def spacer(val):
    
    if ',' in val:
        
        n_val =str(val).replace(',', ', ', 20)
        return n_val
    else:
        return val
    