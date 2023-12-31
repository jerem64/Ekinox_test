import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import sklearn
from sklearn.kernel_ridge import KernelRidge
import numpy as np



# Load the model and the encoders
model = joblib.load('resources/finalGrade_prediction_model.joblib')
encoder_address = joblib.load('resources/encoder_address.joblib')
encoder_famsize = joblib.load('resources/encoder_famsize.joblib')
encoder_Pstatus = joblib.load('resources/encoder_Pstatus.joblib')
scaler = joblib.load('resources/scaler.joblib')
students_encoded = joblib.load('resources/students_encoded.joblib')

#prepare the dataframe
if "df_students" not in st.session_state:
    st.session_state.df_students= pd.DataFrame(columns=['StudentID', 'FirstName', 'FamilyName', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
    'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
    'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'FinalGrade'])


#Create the form
st.title('Student improvement evaluator')

st.header('Add a student')
col1, col2, col3 = st.columns(3)
with col1:
    StudentID = st.text_input('Student ID', '')
    sex = st.radio('Select Gender', ['F', 'M'])
with col2:
    FirstName = st.text_input('First Name', '')
    age = st.number_input('Age', min_value=15, max_value=22, value=18)
with col3:    
    FamilyName = st.text_input('Family Name', '')
        
st.text("")

with st.expander("Details"):
    col1b, col2b, col3b = st.columns(3)
    with col1b:
        address = st.radio('Select address', ['U', 'R'])
        Medu = st.number_input('Medu', min_value=0, max_value=4, value=0)
        Mjob = st.radio('Select Mjob', ['teacher', 'health', 'services', 'at_home', 'other'])
        reason = st.radio('Select reason', ['home', 'reputation', 'course', 'other'])
    with col2b:
        famsize = st.radio('Select famsize', ['LE3', 'GT3'])
        Fedu = st.number_input('Fedu', min_value=0, max_value=4, value=0)
        Fjob = st.radio('Select Fjob', ['teacher', 'health', 'services', 'at_home', 'other'])
        guardian = st.radio('Select guardian', ['mother', 'father', 'other'])        
    with col3b:
        Pstatus = st.radio('Select Pstatus', ['T', 'A'])
        traveltime = st.number_input('traveltime', min_value=1, max_value=4, value=1)
        schoolsup = st.radio('Select schoolsup', ['yes', 'no'])

    col1t, col2t, col3t = st.columns(3)
    with col1t:
        studytime = st.number_input('studytime', min_value=1, max_value=4, value=1)
    with col2t:
        failures = st.number_input('failures', min_value=0, max_value=4, value=0)        
    with col3t:
        st.text("")
    
    col1q, col2q, col3q = st.columns(3)
    with col1q:
        famsup = st.radio('Select famsup', ['yes', 'no'])
        nursery =  st.radio('Select nursery', ['yes', 'no'])
        romantic =  st.radio('Select romantic', ['yes', 'no'])
    with col2q:
        paid =  st.radio('Select paid', ['yes', 'no'])
        higher =  st.radio('Select higher', ['yes', 'no'])        
    with col3q:
        activities =  st.radio('Select activities', ['yes', 'no'])
        internet =  st.radio('Select internet', ['yes', 'no'])

    col1d, col2d, col3d, col4d = st.columns(4)
    with col1d:
        famrel = st.number_input('famrel', min_value=1, max_value=5, value=1)
        Walc = st.number_input('Walc', min_value=1, max_value=5, value=1)
    with col2d:
        freetime = st.number_input('freetime', min_value=1, max_value=5, value=1)
        Dalc = st.number_input('Dalc', min_value=1, max_value=5, value=1)    
    with col3d:
        goout = st.number_input('goout', min_value=1, max_value=5, value=1)
        absences = st.number_input('absences', min_value=0, max_value=93, value=0)
    with col4d:
        health = st.number_input('health', min_value=1, max_value=5, value=1)
    

col1r, col2r, col3r = st.columns(3)
with col1r:
    FinalGrade = st.number_input('FinalGrade', min_value=0, max_value=20, value=10)


#Button to add a student, ne need to create a function
if st.button('Add student'):
    new_student = {'StudentID': StudentID, 'FirstName': FirstName, 'FamilyName': FamilyName, 'sex': sex, 'age': age, 
        'address': address, 'famsize': famsize, 'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu, 'Mjob': Mjob, 'Fjob': Fjob, 'reason': reason,
        'guardian': guardian, 'traveltime': traveltime,'studytime': studytime, 'failures': failures, 'schoolsup': schoolsup, 'famsup': famsup, 
        'paid': paid, 'activities': activities, 'nursery': nursery, 'higher': higher, 'internet': internet, 'romantic': romantic, 'famrel': famrel,
        'freetime': freetime, 'goout': goout, 'Dalc': Dalc, 'Walc': Walc, 'health': health, 'absences': absences, 'FinalGrade': FinalGrade}
        
    st.session_state.df_students = pd.concat([st.session_state.df_students, pd.DataFrame([new_student])], ignore_index=True)


st.header('Students to evaluate')
#Display the students
if "df_students" in st.session_state:
    st.table(st.session_state["df_students"])

#Button to evaluate the students
if st.button('Evaluate'):
    #Copy of the student dataframe
    df_eval = st.session_state["df_students"].copy()
    #Drop the column generated by the previous evaluation
    df_eval.drop('prediction', axis=1, errors='ignore', inplace=True)

    #Prepare the new data to match the model format like we did on the notebook to create the model
    df_eval['Pedu'] = (df_eval['Medu'] + df_eval['Fedu']) / 2
    df_eval['alc'] = (df_eval['Dalc'] / 5 + df_eval['Walc'] / 2)
    df_eval.drop(columns=['Medu', 'Fedu', 'Dalc', 'Walc'], inplace=True)

    #One hot encoding
    df_eval = pd.get_dummies(df_eval, columns=['sex', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'Mjob', 'Fjob'])

    #Ordinal encoding
    mapping = {'home': 0, 'other': 1, 'reputation': 2, 'course': 3}
    df_eval['reason'] = df_eval['reason'].map(mapping)
    df_eval['famsize'] = encoder_famsize.transform(df_eval['famsize'])
    df_eval['Pstatus'] = encoder_Pstatus.transform(df_eval['Pstatus'])
    df_eval['address'] = encoder_address.transform(df_eval['address'])
   
    #Clean the columns for training
    df_eval.drop(columns=['StudentID', 'FirstName', 'FamilyName'], inplace=True)

    #In case of missing columns due to the small range of categories in data we add the missing column that may not have been created during the encoding
    missing_columns = set(students_encoded.columns) - set(df_eval.columns)
    for column in missing_columns:
        df_eval[column] = 0 

    #Remove the value to predict
    X = df_eval.drop("FinalGrade", axis=1)

    #Match the order for the scaler
    desired_order=['age', 'address', 'famsize', 'Pstatus', 'reason', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'health', 'absences', 'Pedu', 'alc',
        'sex_F', 'sex_M', 'guardian_father', 'guardian_mother', 'guardian_other', 'schoolsup_no', 'schoolsup_yes', 'famsup_no', 'famsup_yes', 'paid_no', 'paid_yes', 'activities_no',
        'activities_yes', 'nursery_no', 'nursery_yes', 'higher_no', 'higher_yes', 'internet_no', 'internet_yes', 'romantic_no', 'romantic_yes', 'Mjob_at_home', 'Mjob_health', 'Mjob_other',
        'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher']    
    X = X[desired_order]

    #Normalize the data
    X_normalized = scaler.transform(X)

    #Prediction
    y_pred = model.predict(X_normalized)

    #Add the prediction and the gap between the prediction and the true value
    st.session_state["df_students"]['prediction'] = 0
    st.session_state["df_students"].loc[st.session_state["df_students"].index, 'prediction'] = y_pred
    st.session_state["df_students"]['gap'] = st.session_state["df_students"]['prediction'] - st.session_state["df_students"]['FinalGrade']
    st.session_state["df_students"] = st.session_state["df_students"].sort_values(by='gap', ascending=False)

    #Display the dataframe
    st.table(st.session_state["df_students"])

    #Display the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(st.session_state["df_students"]['StudentID'], st.session_state["df_students"]['FinalGrade'], label='FinalGrade', s=50)
    scatter = ax.scatter(st.session_state["df_students"]['StudentID'], st.session_state["df_students"]['prediction'], marker='x', label='Prediction', s=50)
    ax.set_title('Scatter Plot of FinalGrade and Prediction')
    ax.set_xlabel('StudentID')
    ax.set_ylabel('Grades')
    ax.set_yticks(np.arange(0, 21))
    ax.axhline(0, color='black',linewidth=0.5)
    ax.legend()

    st.pyplot(fig)