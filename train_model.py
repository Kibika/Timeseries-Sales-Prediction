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
# from sklearn.externals import joblib
from keras.models import load_model
from sklearn import metrics
import mlflow
import mlflow.sklearn
import mlflow.keras
# from functions import date_type,outlier,encode
import os
import warnings
import sys
import pathlib

loaded_model = tf.keras.models.load_model('lstm_model.h5')

# load the datasets
store_data = pd.read_csv('./data/store.csv')  # describes the characteristics of a store
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

mlflow.set_experiment("sales_prediction")


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


def timeseries_evaluation_metrics_func(y_true, y_pred):
    MSE = metrics.mean_squared_error(y_true, y_pred)
    MAE = metrics.mean_absolute_error(y_true, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    R2 = metrics.r2_score(y_true, y_pred)
    return MSE, MAE, RMSE, R2


def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset) - horizon
    for i in range(start, end):
        indices = range(i - window, i)
        X.append(dataset[indices])
        indicey = range(i + 1, i + 1 + horizon)
        y.append(target[indicey])
    return np.array(X), np.array(y)


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


m = train1(train_data, store_data)
m.clean()
m.competition_duration()
m.promotion_duration()
m.promotion_interval()
m.outlier()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    full_train = m.encode()
    full_train = full_train.iloc[:200000, :]
    # print(full_train.shape)
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()
    X_data = X_scaler.fit_transform(full_train.drop(['Sales', 'Store', 'Customers'], axis=1))
    Y_data = Y_scaler.fit_transform(full_train[['Sales']])
    validate = full_train.tail(42)
    hist_window = 100
    horizon = 42
    TRAIN_SPLIT = 160000
    x_train, y_train = custom_ts_multi_data_prep(X_data, Y_data, 0, TRAIN_SPLIT, hist_window, horizon)
    x_vali, y_vali = custom_ts_multi_data_prep(X_data, Y_data, TRAIN_SPLIT, None, hist_window, horizon)
    batch_size = 256
    buffer_size = 150
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()
    val_data = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))
    val_data = val_data.batch(batch_size).repeat()

    epochs = 10
    steps_per_epoch = 5

    # lstm_model = tf.keras.models.Sequential([
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True),
    #                                   input_shape=x_train.shape[-2:]),
    #     tf.keras.layers.Dense(20, activation='tanh'),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
    #     tf.keras.layers.Dense(20, activation='tanh'),
    #     tf.keras.layers.Dense(20, activation='tanh'),
    #     tf.keras.layers.Dropout(0.25),
    #     tf.keras.layers.Dense(units=horizon),
    # ])
    lstm_model = loaded_model
    lstm_model.compile(optimizer='adam', loss='mse')
    #print(lstm_model.summary())
    # mlflow.sklearn.log_model(lstm_model, "model")
    mlflow.keras.autolog()

    model_path = 'Bidirectional_LSTM_Multivariate.h5'
    early_stopings = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1,
                                                      mode='min')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min',
                                                    verbose=0)
    callbacks = [early_stopings, checkpoint]
    history = lstm_model.fit(train_data, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_data,
                             validation_steps=50,
                             verbose=1, callbacks=callbacks)
    plt.figure(figsize=(16, 9))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'validation loss'])
    plt.close()
    plt.savefig("model_loss.png")

    mlflow.log_artifact("model_loss.png")

    data_val = X_scaler.fit_transform(full_train.drop(['Sales', 'Customers', 'Store'], axis=1).tail(100))
    val_rescaled = data_val.reshape(1, data_val.shape[0], data_val.shape[1])
    pred = lstm_model.predict(val_rescaled)
    pred_Inverse = Y_scaler.inverse_transform(pred)

    (MSE, MAE, RMSE, R2) = timeseries_evaluation_metrics_func(validate['Sales'], pred_Inverse[0])
    plt.figure(figsize=(16, 9))
    plt.plot(list(validate['Sales']))
    plt.plot(list(pred_Inverse[0]))
    plt.title("Actual vs Predicted")
    plt.ylabel("Sales")
    plt.legend(('Actual', 'predicted'))
    plt.close()
    plt.savefig("actualvspred.png")

    mlflow.log_artifact("actualvspred.png")
    # lstm_model.save('lstm_model.h5')
