import pandas as pd
import numpy as np
import logging
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table as dt
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
# from sklearn.externals import joblib
from keras.models import load_model
from sklearn import metrics
import os
import warnings
import sys
import pathlib
from app import app

loaded_model = tf.keras.models.load_model('lstm_model.h5')
# load the datasets
store_data = pd.read_csv('./data/store.csv')  # describes the characteristics of a store
train_data = pd.read_csv('./data/train.csv')


# test_data = pd.read_csv('./data/test.csv')


def date_type(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.weekofyear
    return df


def outlier(df):
    column_name = ['CompetitionDistance']
    for i in column_name:
        upper_quartile = df[i].quantile(0.75)
        lower_quartile = df[i].quantile(0.25)
        df[i] = np.where(df[i] > upper_quartile, df[i].mean(), np.where(df[i] < lower_quartile, df[i].median(), df[i]))
    return df


class train1:
    def __init__(self, df, store):
        self.df = df
        self.store = store

    def get_df(self):
        return self.df

    def set_df(self, new_df):
        self.df = new_df

    def clean(self):
        df = self.df
        store = self.store

        df = df.merge(store, on=["Store"], how="outer")
        df = df.fillna(0)
        df = date_type(df)
        self.set_df(df)
        # print(df)

    def competition_duration(self):
        df = self.df
        df['CompetitionOpen '] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + (
                df['Month'] - df['CompetitionOpenSinceMonth'])
        df = df.drop(['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'], axis=1)
        self.set_df(df)
        # print(df)

    def promotion_duration(self):
        df = self.df
        df['PromoOpen'] = 12 * (df['Year'] - df['Promo2SinceYear']) + (df['WeekOfYear'] - df['Promo2SinceWeek']) / 4.0
        df = df.drop(['Promo2SinceYear', 'Promo2SinceWeek'], axis=1)
        self.set_df(df)
        # print(df)

    def promotion_interval(self):
        df = self.df
        s = df['PromoInterval'].str.split(',', n=3, expand=True)
        s.columns = ['PromoInterval0', 'PromoInterval1', 'PromoInterval2', 'PromoInterval3']
        df = df.join(s)

        # Converting Promointerval columns to numerical.
        month_to_num_dict = {
            'Jan': 1,
            'Feb': 2,
            'Mar': 3,
            'Apr': 4,
            'May': 5,
            'Jun': 6,
            'Jul': 7,
            'Aug': 8,
            'Sept': 9,
            'Oct': 10,
            'Nov': 11,
            'Dec': 12,
            'nan': np.NaN
        }

        df['PromoInterval0'] = df['PromoInterval0'].map(month_to_num_dict)
        df['PromoInterval1'] = df['PromoInterval1'].map(month_to_num_dict)
        df['PromoInterval2'] = df['PromoInterval2'].map(month_to_num_dict)
        df['PromoInterval3'] = df['PromoInterval3'].map(month_to_num_dict)
        df = df.drop(['PromoInterval'], axis=1)
        df = df.fillna(0)
        self.set_df(df)

    def outlier(self):
        df = self.df
        df = outlier(df)
        self.set_df(df)
        # print(df)

    def encode(self):
        df = self.df
        df['StateHoliday'] = df['StateHoliday'].astype('str')
        df['StateHoliday'] = df['StateHoliday'].astype('str')
        lb_make = LabelEncoder()
        df['StateHoliday'] = lb_make.fit_transform(df['StateHoliday'])
        df['StoreType'] = lb_make.fit_transform(df['StoreType'])
        df['Assortment'] = lb_make.fit_transform(df['Assortment'])
        df.set_index(['Date'], inplace=True)
        return df


class predict1:
    def __init__(self, df):
        self.df = df
        self.loaded_model = loaded_model

    def get_df(self):
        return self.df

    def set_df(self, new_df):
        self.df = new_df

    def deep_model(self):
        df = self.df
        loaded_model = self.loaded_model
        X_scaler = MinMaxScaler()
        Y_scaler = MinMaxScaler()
        Y_scaler.fit_transform(df[['Sales']])
        data_val = X_scaler.fit_transform(df.drop(['Sales', 'Customers', 'Store'], axis=1).tail(100))
        val_rescaled = data_val.reshape(1, data_val.shape[0], data_val.shape[1])
        pred = loaded_model.predict(val_rescaled)
        pred_Inverse = Y_scaler.inverse_transform(pred)
        pred_Inverse_df = pd.DataFrame((pred_Inverse).T, columns=['Sales']).reset_index()
        pred_Inverse_df.columns = ['Day', 'Sales']
        return pred_Inverse_df


m = train1(train_data, store_data)
m.clean()
m.competition_duration()
m.promotion_duration()
m.promotion_interval()
m.outlier()
full_test = m.encode()


def init_dashboard(server):
    dash_app = dash.Dash(
        __name__, server=server, external_stylesheets=external_stylesheets,
        routes_pathname_prefix='/dashapp/',
        meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    )

    layout = dash_app.layout = html.Div([
        dbc.Col([
            dbc.Row([
                dbc.Col([

                    html.Div([
                        dcc.Graph(

                            id='line-1'
                        )
                    ])

                ])
            ])
        ],
            md=6
        ),

    ], id="overview")
    return dash_app.server


@app.callback(Output('line-1', 'figure'),
              [Input('overview', 'value')])
def prediction_page(value):
    df = full_test
    m = predict1(df)
    predictions = m.deep_model()
    fig1 = px.line(predictions, x='Day', y="Sales", title='Sales Forecast')
    # print(full_test)

    return fig1
