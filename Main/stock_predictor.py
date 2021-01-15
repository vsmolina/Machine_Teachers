import numpy as np
import pandas as pd
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class Stock_predictor():

    def __init__(self, df, window_size = 5):
        self.window = window_size
        self.data = df
    
    '''
    This function accepts the column number for the features (X) and the target (y)
    It chunks the data up with a rolling window of Xt-n to predict Xt
    It returns a numpy array of X any y
    '''
    def window_data(self, feature_col_number = 0, target_col_number = 0):
        X = []
        y = []
        for i in range(len(self.data) - self.window - 1):
            features = self.data.iloc[i:(i + self.window), feature_col_number]
            target = self.data.iloc[(i + self.window), target_col_number]
            X.append(features)
            y.append(target)
        self.features = np.array(X) 
        self.target = np.array(y).reshape(-1, 1)
        
    def prepare_data(self):
        X = self.features
        y = self.target
        

        split = int(0.7 * len(X))

        X_train = X[: split]
        X_test = X[: split]

        y_train = y[: split]
        y_test = y[: split]

        scaler = MinMaxScaler()
        scaler.fit(X)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        scaler.fit(y)

        self.y_train = scaler.transform(y_train)
        self.y_test = scaler.transform(y_test)
        self.X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        self.X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        self.scaler = scaler

    def setup_model(self, optimize = "adam", loss = "mean_squared_error"):
        model = Sequential()

        number_units = 30
        dropout_fraction = 0.2

        
        model.add(LSTM(
            units=number_units,
            return_sequences=True,
            input_shape=(self.X_train.shape[1], 1)
        ))
        model.add(Dropout(dropout_fraction))

        
        model.add(LSTM(units=number_units, return_sequences=True))
        model.add(Dropout(dropout_fraction))

        
        model.add(LSTM(units=number_units))
        model.add(Dropout(dropout_fraction))

        model.add(Dense(1))

        model.compile(optimizer= optimize, loss=loss)

        self.model = model

    def train_model(self, model_epochs = 30, model_batch_size = 90):
        self.model.fit(self.X_train, self.y_train, epochs=model_epochs, shuffle=False, batch_size=model_batch_size, verbose=1)

    def evaluate_model(self):
        return self.model.evaluate(self.X_test, self.y_test)
    
    def predictions_df(self):
        predicted = self.model.predict(self.X_test)
        predicted_prices = self.scaler.inverse_transform(predicted)
        real_prices = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        self.stocks = pd.DataFrame({
            "Real": real_prices.ravel(),
            "Predicted": predicted_prices.ravel(),
        }, index = self.data.index[-len(real_prices): ])
        return self.stocks

    def plot_predictions(self):
        self.stocks.plot()
    
    def get_signals(self, signal_percentage = 0.05):
        self.stocks["Signal"] = 0.0

        for stock in self.stocks:
            if ((stock["Predicted"] / stock["Real"]) - 1) >= signal_percentage:
                stock["Signal"] = 1.0
            elif (1 - (stock["Predicted"] / stock["Real"])) >= signal_percentage:
                stock["Signal"] = -1.0
            else:
                stock["Signal"] = 0.0 
        return self.stocks
             