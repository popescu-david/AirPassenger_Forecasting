import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast.utils import AirPassengersDF
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.api.layers import LSTM, Dense, Dropout
from keras.api.optimizers import AdamW
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt

Y_df = AirPassengersDF

Y_train_df = Y_df[Y_df.ds <= '1959-12-31']
Y_test_df = Y_df[Y_df.ds > '1959-12-31']

scaler = MinMaxScaler(feature_range=(0, 1))
Y_train_scaled = scaler.fit_transform(Y_train_df[['y']])
Y_test_scaled = scaler.transform(Y_test_df[['y']])

def create_dataset(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 6
X_train, y_train = create_dataset(Y_train_scaled, look_back)
X_test, y_test = create_dataset(Y_test_scaled, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(LSTM(50, activation="relu", return_sequences=True, input_shape=(look_back, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, activation="relu", return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1))

optimizer = AdamW(learning_rate=0.003)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.summary()

model.fit(X_train, y_train, epochs=500, verbose=2)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])

mse = mean_squared_error(y_test[0], test_predict[:, 0])
mae = mean_absolute_error(y_test[0], test_predict[:, 0])
rmse = sqrt(mse)
mape = mean_absolute_percentage_error(y_test[0], test_predict[:, 0])

print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2%}')

Y_test_df = Y_test_df[look_back:]
Y_test_df['LSTM'] = test_predict

fig, ax = plt.subplots(figsize=(10, 6))
plot_df = pd.concat([Y_train_df, Y_test_df]).set_index('ds')
plot_df[['y', 'LSTM']].plot(ax=ax, linewidth=2)

ax.set_title('LSTM Forecast vs Actuals')
plt.savefig('model_lstm.png', format='png', bbox_inches='tight')
plt.close(fig)