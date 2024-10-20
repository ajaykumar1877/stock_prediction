#Import required libraries
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Step 1: Load the stock data
# Choose the stock ticker symbol (e.g., "AAPL" for Apple)
stock = "AAPL"
data = yf.download(stock, start="2010-01-01", end="2023-01-01")

# Display the data
print(data.head())

# Step 2: Data Preprocessing
# We'll use only the 'Close' price for prediction
data = data[['Close']]

# Fill any missing values
data.fillna(method='ffill', inplace=True)

# Convert the data into a NumPy array
dataset = data.values

# Scale the data using MinMaxScaler (LSTM performs better with scaled data)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Step 3: Prepare the training data
training_data_len = int(np.ceil(len(scaled_data) * 0.8))  # Use 80% of data for training

# Create the training data
train_data = scaled_data[0:int(training_data_len), :]

# Split the data into x_train and y_train datasets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])  # 60 previous values as features
    y_train.append(train_data[i, 0])  # The next value as target

# Convert x_train and y_train to NumPy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape x_train for LSTM model input
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Step 4: Build the LSTM model
model = Sequential()

# Add LSTM layers
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))

# Add dense layers
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Step 6: Create the test dataset
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]  # Actual prices

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert x_test to a NumPy array
x_test = np.array(x_test)

# Reshape x_test for LSTM model input
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Step 7: Get the model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Step 8: Visualize the results
# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Visualize the predicted stock prices with the actual stock prices
plt.figure(figsize=(16,8))
plt.title('Stock Price Prediction Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Actual', 'Predicted'], loc='lower right')
plt.show()

# Display the valid and predicted prices
print(valid)

