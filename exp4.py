import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url, header=0, parse_dates=[0], index_col=0, date_parser=pd.to_datetime)

# Plot original time series
data.plot()
plt.title("Monthly Airline Passengers")
plt.xlabel("Date")
plt.ylabel("Number of Passengers")
plt.show()

# Decompose time series
result = seasonal_decompose(data, model='multiplicative', period=12)
result.plot()
plt.show()

# Augmented Dickey-Fuller test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] <= 0.05:
        print("Series is stationary")
    else:
        print("Series is not stationary")

adf_test(data)

# SARIMA model and forecast
model = SARIMAX(data, order=(1,1,1), seasonal_order=(1,1,0,12))
model_fit = model.fit(disp=False)
forecast = model_fit.forecast(steps=12)

print(f"Forecasted values: \n{forecast}")

plt.plot(data, label='Historical Data')
plt.plot(pd.date_range(start=data.index[-1], periods=13, freq='M')[1:], forecast, label='Forecast', color='red')
plt.title("Airline Passengers Forecast")
plt.xlabel("Date")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()