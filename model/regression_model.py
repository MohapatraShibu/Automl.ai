import pandas as pd
from pycaret.regression import*
import os
from operator import index
import streamlit as st
import plotly.express as px

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

def get_model(df,target):
    setup(df,target=target)
    setup_df = pull()
    best_model = compare_models()
    compare_df = pull()
    save_model(best_model,'best_model')
    return([setup_df,compare_df,best_model])

def predict_test(test_df):
    best_model = load_model('best_model') 
    return predict_model(best_model,data=test_df)
