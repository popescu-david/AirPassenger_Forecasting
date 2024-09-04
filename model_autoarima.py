import pandas as pd
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from neuralforecast.utils import AirPassengersDF
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt

Y_df = AirPassengersDF

Y_train_df = Y_df[Y_df.ds <= '1959-12-31']
Y_test_df = Y_df[Y_df.ds > '1959-12-31']

scaler = MinMaxScaler(feature_range=(0, 1))
Y_train_scaled = scaler.fit_transform(Y_train_df[['y']])
Y_test_scaled = scaler.transform(Y_test_df[['y']])

Y_train_scaled_df = pd.DataFrame({
    'unique_id': Y_train_df['unique_id'].values,
    'ds': Y_train_df['ds'].values,
    'y': Y_train_scaled.flatten()
})

Y_test_scaled_df = pd.DataFrame({
    'unique_id': Y_test_df['unique_id'].values,
    'ds': Y_test_df['ds'].values,
    'y': Y_test_scaled.flatten()
})

horizon = len(Y_test_scaled)
model = AutoARIMA(season_length=12)
sf = StatsForecast(models=[model], freq='M')
sf.fit(df=Y_train_scaled_df)

Y_hat_scaled = sf.predict(horizon).reset_index()

Y_hat_values_scaled = Y_hat_scaled[['AutoARIMA']]
Y_hat_values = scaler.inverse_transform(Y_hat_values_scaled)

Y_hat_df = pd.DataFrame({
    'unique_id': Y_test_df['unique_id'].values,
    'ds': Y_hat_scaled['ds'].values,
    'AutoARIMA': Y_hat_values.flatten()
})

Y_hat_df = Y_test_df.merge(Y_hat_df, how='left', on=['unique_id', 'ds'])

fig, ax = plt.subplots(figsize=(10, 6))
plot_df = pd.concat([Y_train_df, Y_hat_df]).set_index('ds')
plot_df[['y', 'AutoARIMA']].plot(ax=ax, linewidth=2)

ax.set_title('ARIMA Forecast vs Actuals')
plt.savefig('model_autoarima.png', format='png', bbox_inches='tight')
plt.close(fig)

mse = mean_squared_error(Y_test_df['y'], Y_hat_df['AutoARIMA'])
mae = mean_absolute_error(Y_test_df['y'], Y_hat_df['AutoARIMA'])
rmse = sqrt(mse)
mape = mean_absolute_percentage_error(Y_test_df['y'], Y_hat_df['AutoARIMA'])

print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2%}')
