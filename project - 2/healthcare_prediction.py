# -*- coding: utf-8 -*-
"""
Created on Mon May 13 00:35:42 2024

@author: SRI LAVANYA
"""


import numpy as np
import streamlit as st
import joblib

def model_prediction(input_data, loaded_model):
    input_data_as_array = np.asarray(input_data)
    input_data1 = input_data_as_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data1)
    if prediction[0] == 'Abnormal':
        return 'Person having the Abnormal condition'
    else:
        return 'Person having the Inconclusive condition'

def main():
    # Loading the model
    try:
        loaded_model = joblib.load("C:/Users/SRI LAVANYA/Downloads/Disease7_model.sav")
    except Exception as e:
        st.error("Error loading the model: {}".format(e))
        return
    
    # Giving a title
    st.title('Healthcare Prediction Web App')
    Age = st.text_input('Enter the age of the patient')
    Gender = st.text_input('Enter the gender of the patient')
    BloodType = st.text_input('Enter the blood type of the patient')
    MedicalCondition = st.text_input('Enter the medical condition')
    InsuranceProvider = st.text_input('Enter the insurance provider')
    BillingAmount = st.text_input('Enter the billing amount')
    AdmissionType = st.text_input('Enter the admission type')
    
    # Code for prediction
    diagnosis = ''
    
    # Creating a button for prediction
    if st.button('Healthcare Test Result'):
        input_data = [Age, Gender, BloodType, MedicalCondition, InsuranceProvider, BillingAmount, AdmissionType]
        diagnosis = model_prediction(input_data, loaded_model)
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    
    
