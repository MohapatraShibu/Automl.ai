import streamlit as st
import pandas as pd
from io import StringIO
import os
import pickle
from operator import index
import plotly.express as px

# profiling imports
import pandas_profiling
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report 

# autoMl imports
import model.classifier_model as cm
import model.regression_model as rm

with st.sidebar:
    st.image("logo.png")
    st.title('Developed by Shibu')
    nav_choice = st.radio("NAVIGATION",['Uploading','Profiling','ML_Modelling','Forecasting'])
    st.info("This application helps you build and explore your data.")

if nav_choice =="Uploading":
    st.title("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file,index_col=None)
        st.dataframe(df)
        df.to_csv('model_data.csv',index=False)
    st.write('Try with sample Space Titanic Dataset')
    
    train_df = open('./demo/_train.csv','rb')
    test_df = open('./demo/_test.csv','rb')

    st.download_button('Download Train Data',train_df,'train_data.csv')
    st.download_button('Download Test Data',test_df,'test_data.csv')

source_data_exists = os.path.exists("model_data.csv")
if source_data_exists:
    df=pd.read_csv('model_data.csv',index_col=None)

if nav_choice=="Profiling":
    st.title("Exploratory Data Analysis")
    if source_data_exists:
        data_report = df.profile_report()
        st_profile_report(data_report)
    else:
        st.write("Please upload your dataser in upload menu")

model_type = "Classification"

if nav_choice == 'ML_Modelling':
    st.title("Machine Learning model selection")
    model_type = st.radio('Select model type',
    ('Classification','Regression'))
    
    target = st.selectbox('Select the target',df.columns)
    if st.button('Train Model'):
        if model_type == 'Classification':
            model_list = cm.get_model(df,target) #[ml experiment settings, model compare results, best model]

        else:
            model_list = rm.get_model(df,target) #[ml experiment settings, model compare results, best model]
            
        with open('best_model.pkl','rb') as f :
            st.download_button('Download Model',f,'best_model.pkl')

if nav_choice == 'Forecasting':
    st.title("Predict target with the model")
    try:
        if os.path.exists("best_model.pkl"):
            test_file = st.file_uploader("Choose a file")
            if test_file:
                test_df = pd.read_csv(test_file,index_col=None)
                if model_type == "Classification":
                    test_result = cm.predict_test(test_df)
                else:
                    test_result = rm.predict_test(test_df)
                test_result.to_csv('test_result.csv',index=False)
                if st.button('Predict'):
                    st.dataframe(test_result)
                    with open('test_result.csv','rb') as f :
                        st.download_button('Download Result',f,'test_result.csv')
        else:
            st.write("Please train a Model in ML menu!")
            
    except Exception as e:
        st.write("Something went worng, please check if your target and test data set variable names match")
        st.write(f"Train Data: {list(df.columns)}")
        st.dataframe(df.head())
        st.write(f"Test Data: {list(test_df.columns)}")
        st.dataframe(test_df.head())
        st.write(e)