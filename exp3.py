import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'X2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Y': [1.1, 1.9, 3.1, 4.2, 5.1, 6.2, 7.1, 8.0, 9.0, 10.1]
}
df = pd.DataFrame(data)
X = df[['X1', 'X2']]
Y = df['Y']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(Y_test, Y_pred))
print("R^2 Score:", r2_score(Y_test, Y_pred))

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(df['X1'], df['Y'], color='blue', label='Actual data')
plt.plot(df['X1'], model.predict(df[['X1', 'X2']]), color='red', label='Fitted line')
plt.xlabel('X1')
plt.ylabel('Y')
plt.title('X1 vs Y')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(df['X2'], df['Y'], color='green', label='Actual data')
plt.plot(df['X2'], model.predict(df[['X1', 'X2']]), color='red', label='Fitted line')
plt.xlabel('X2')
plt.ylabel('Y')
plt.title('X2 vs Y')
plt.legend()

plt.tight_layout()
plt.show()