import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url, header=0, parse_dates=[0], index_col=0, date_parser=pd.to_datetime)

data.plot()
plt.title("Monthly Airline Passengers")
plt.xlabel("Date")
plt.ylabel("Number of Passengers")
plt.show()

def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] <= 0.05:
        print("Series is stationary")
    else:
        print("Series is not stationary")

adf_test(data)

data_diff = data.diff().dropna()
plot_acf(data_diff)
plot_pacf(data_diff)
plt.show()

model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

forecast = model_fit.forecast(steps=12)
plt.plot(data, label='Historical Data')
plt.plot(pd.date_range(start=data.index[-1], periods=13, freq='M')[1:], forecast, label='Forecast', color='red')
plt.title("Airline Passengers Forecast")
plt.xlabel("Date")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()