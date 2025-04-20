import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Y = np.array([2, 4, 5, 4, 5, 7, 8, 8, 9, 10])
n = len(X)

sum_x = np.sum(X)
sum_y = np.sum(Y)
sum_xx = np.sum(X**2)
sum_xy = np.sum(X*Y)

m = (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x**2)
b = (sum_y - m*sum_x) / n

print(f"Slope(m): {m}")
print(f"Intercept(b): {b}")

Y_pred = m*X + b

plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, Y_pred, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Simple Linear Regression')
plt.show()