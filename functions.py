import pandas as pd
import numpy as np
import logging
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
#from sklearn.externals import joblib
from functions import date_type, outlier, encode
from keras.models import load_model
from sklearn import metrics
import mlflow
import mlflow.sklearn




def date_type(df):
    df['Date']=pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.weekofyear
    return df

def outlier(df):
    column_name=['CompetitionDistance']
    for i in column_name:
        upper_quartile=df[i].quantile(0.75)
        lower_quartile=df[i].quantile(0.25)
        df[i]=np.where(df[i]>upper_quartile,df[i].mean(),np.where(df[i]<lower_quartile,df[i].median(),df[i]))
    return df

def encode(df):
    for i in df.select_dtypes('object').columns:
        le = LabelEncoder().fit(df[i])
        df[i] = le.transform(df[i])