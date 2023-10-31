import streamlit as st
import pandas as pd




from predictionService.predictor import predictor

st.write("""
# Insomnia Prediction App

This app predicts the **Insomnia & Sleep Apnea** !

Data obtained from the [kaggle]()
""")

st.sidebar.header('Insomnia Predictor')


def user_input_features():
    Gender = st.sidebar.selectbox('Gender',('Male','Female'))
    Age = st.sidebar.slider('Age', min_value=27, max_value=59, step=1)
    BMI_Category = st.sidebar.selectbox('BMI Category',('Overweight', 'Normal', 'Normal Weight', 'Obese'))
    Sleep_Duration = st.sidebar.slider('Sleep Duration', min_value=5.5, max_value=9.0, step=0.1)
    Quality_of_Sleep = st.sidebar.slider('Quality of Sleep', min_value=4, max_value=9, step=1)
    Physical_Activity_Level = st.sidebar.slider('Physical Activity Level', min_value=30, max_value=90, step=1)
    Stress_Level = st.sidebar.slider('Stress Level', min_value=3, max_value=8, step=1)
    Heart_Rate = st.sidebar.slider('Heart Rate', min_value=65, max_value=86, step=1)
    
    
    Systolic = st.sidebar.slider('BP - Systolic', min_value=115, max_value=140, step=5)
    Diastolic = st.sidebar.slider('BP - Diastolic', min_value=75, max_value=95, step=5)
    
       
    data = {
        'gender' : Gender,
        'age': Age,
        'sleep_duration': Sleep_Duration,
        'sleep_quality': Quality_of_Sleep,
        'physical_activity_level': Physical_Activity_Level,
        'stress_level': Stress_Level,
        'bmi_category' : BMI_Category, 
        'heart_rate': Heart_Rate,
        'systolic' : Systolic,
        'diastolic' : Diastolic 
        }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()


prediction = predictor(input_df)

# print(val)

# Displays the user input features
st.subheader('User Input features')


st.subheader('Prediction')

st.write(prediction[0])

# st.subheader('Prediction Probability')
# st.write(prediction_proba)