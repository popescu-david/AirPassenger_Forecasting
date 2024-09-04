import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from neuralforecast.utils import AirPassengersDF

Y_df = AirPassengersDF

print(Y_df.info())
missing_values = Y_df.isnull().sum()
print(f'Missing values number: {missing_values}')
duplicate_values = Y_df.duplicated().sum()
print(f'Duplicate values number: {duplicate_values}')
Y_df['z_score'] = (Y_df['y'] - Y_df['y'].mean()) / Y_df['y'].std()
outliers = Y_df[Y_df['z_score'].abs() > 3].copy()
print(f'Number of outliers: {max(len(outliers),0)}')

decomposition = seasonal_decompose(Y_df.set_index('ds')['y'], model='multiplicative')
fig = decomposition.plot()
plt.savefig('decomposition_plot.png', format='png', bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
plot_acf(Y_df['y'], lags=40, ax=ax[0])
plot_pacf(Y_df['y'], lags=40, ax=ax[1])
plt.savefig('acf_pacf_plot.png', format='png', bbox_inches='tight')
plt.close(fig)