# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load your data
data = pd.read_csv('dataset.csv')

# Feature selection
features = ['Max_Temperature', 'Min_Temperature', 'Rainfall']
X = data[features].values

# Target variable (Drought or not)
# Assuming you have a binary drought indicator (1 for drought, 0 for not)
# Adjust the column name accordingly
y = data['Drought'].values

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# SVM
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train, y_train)

# Predictions
svm_predictions = svm_model.predict(X_test)

# Evaluate SVM
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy}")
print(classification_report(y_test, svm_predictions))

# LSTM
# Assuming you have a time series structure in your data
look_back = 1  # You may need to adjust this based on your data characteristics

# Reshape the data for LSTM
X_lstm, y_lstm = [], []
for i in range(len(X_scaled) - look_back):
    X_lstm.append(X_scaled[i:(i + look_back), :])
    y_lstm.append(y[i + look_back])

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

# Split the data into training and testing sets
train_size = int(len(X_lstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:train_size], X_lstm[train_size:]
y_train_lstm, y_test_lstm = y_lstm[:train_size], y_lstm[train_size:]

# Build LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, len(features))))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=1, verbose=2)

# Predictions
lstm_predictions = model.predict(X_test_lstm)
lstm_predictions = (lstm_predictions > 0.5).astype(int)

# Evaluate LSTM
lstm_accuracy = accuracy_score(y_test_lstm, lstm_predictions)
print(f"LSTM Accuracy: {lstm_accuracy}")
print(classification_report(y_test_lstm, lstm_predictions))

