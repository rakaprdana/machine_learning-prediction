import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv('./data/sample2.csv', parse_dates=['Timestamp'], index_col='Timestamp')

#visualisasi
# plt.figure(figsize=(10,6))
# sns.lineplot(x='DateTime', y='Vehicles', data=df)
# plt.title('Line Chart Time Series')
# plt.xlabel('DATE')
# plt.ylabel('SUM VEHICLES')

# plt.xlim(pd.Timestamp('2015-11-01'), pd.Timestamp('2015-11-30'))
# plt.grid(True)
# plt.show()

#differencing
df_diff = df.diff().dropna()
#visualisasi after diff
plt.figure(figsize=(10,6))
sns.lineplot(x='Timestamp', y='Vehicles', data=df)
plt.title('Differenced Time Series')
plt.xlabel('DATE')
plt.ylabel('SUM VEHICLES')

plt.xlim(pd.Timestamp('2015-11-01'), pd.Timestamp('2015-11-30'))
plt.grid(True)
plt.show()

#ACF & PACF plots
plot_acf(df_diff['Timestamp'])
plt.show()

plot_pacf(df_diff)
plt.show()

#choose model order ARIMA (p, d, q)
p = 1 #order AR FROM PACF
d = 1 #deg differencing
q = 1 #order MA fROM ACF

order = (p, d, q)

train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

#trainning model
model = ARIMA(train, order=order)
fitted_model = model.fitt()

#Eval
forecast, stder, conf_int = fitted_model.forecast(len(test))

#output visual
plt.plot(train, label='Training Data')
plt.plot(test, label='Test Data')
plt.plot(forecast, label='Forecast', linestyle='dashed')
plt.fill_between(test.index, conf_int[:, 0], conf_int[:, 1], color='gray', aplha=0.2, label='Confidence Interval')
plt.legend()
plt.show()

#eval kinerja
from scikitlearn.metrics import mean_squared_error

mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)