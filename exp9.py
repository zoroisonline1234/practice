import pandas as pd
import numpy as np

df = pd.read_csv('/content/synthetic_dataset.csv')
print("Loaded Dataset:")
print(df.head())

df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2

print(df.info())
print(df.describe())
print(df.isnull().sum())

df_older_than_30 = df[df['Age'] > 30]
print(df_older_than_30)

mean_values = df.mean()
median_values = df.median()
mode_values = df.mode().iloc[0]

print(mean_values)
print(median_values)
print(mode_values)

std_devs = np.std(df[['Age', 'Height', 'Salary', 'Weight', 'BMI']], axis=0)
variances = np.var(df[['Age', 'Height', 'Salary', 'Weight', 'BMI']], axis=0)

percentiles_25 = {col: np.percentile(df[col], 25) for col in ['Age', 'Height', 'Salary', 'Weight', 'BMI']}
percentiles_75 = {col: np.percentile(df[col], 75) for col in ['Age', 'Height', 'Salary', 'Weight', 'BMI']}

print(std_devs)
print(variances)
print(percentiles_25)
print(percentiles_75)

print(df.corr())
print(df.cov())

ages_first_5 = df['Age'].values[5:7]
print(ages_first_5)

age_array = df['Age'].values.reshape(-1, 1)
print(age_array)

salary_array = df['Salary'].values
concatenated_data = np.column_stack((age_array, salary_array))
print(concatenated_data)