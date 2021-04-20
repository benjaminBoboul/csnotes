import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from rich.console import Console

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout

console = Console()

# CRIM per capita crime rate by town
# ZN proportion of residential land zoned for lots over 25,000 sq.ft.
# INDUS proportion of non-retail business acres per town
# CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# NOX nitric oxides concentration (parts per 10 million)
# RM average number of rooms per dwelling
# AGE proportion of owner-occupied units built prior to 1940
# DIS weighted distances to five Boston employment centres
# RAD index of accessibility to radial highways
# TAX full-value property-tax rate per $10,000
# PTRATIO pupil-teacher ratio by town
# B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# LSTAT % lower status of the population
# MEDV Median value of owner-occupied homes in $1000’s

boston_dataset = load_boston()
# Here we split our dataset by roughly 80/20, 80% used to train our model and 20% used to ensure predictions are correct
x_train, x_test, y_train, y_test = train_test_split(
    boston_dataset["data"], boston_dataset["target"], random_state=0
)
boston_dataframe = pd.DataFrame(x_train, columns=boston_dataset.feature_names)

console.rule("Boston dataset")
console.print(boston_dataframe)

# feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_dataset = scaler.fit_transform(x_train)

console.rule("Scaled Boston dataset")
console.print(pd.DataFrame(scaled_dataset, columns=boston_dataset.feature_names))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=scaled_dataset.shape))