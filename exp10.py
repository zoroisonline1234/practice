import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = sns.load_dataset('tips')
np.random.seed(0)
data = np.random.randn(100)

# Histogram
plt.hist(df['total_bill'], bins=20, edgecolor='black')
plt.title('Histogram of Total Bill')
plt.xlabel('Total Bill')
plt.ylabel('Frequency')
plt.show()

sns.histplot(df['total_bill'], kde=True)
plt.title('Histogram of Total Bill with KDE')
plt.show()

# Bar Chart
categories = df['day'].value_counts().index
counts = df['day'].value_counts()
plt.bar(categories, counts, color='skyblue')
plt.title('Bar Chart of Days')
plt.xlabel('Days')
plt.ylabel('Count')
plt.show()

sns.countplot(data=df, x='day', palette='coolwarm')
plt.title('Bar Chart of Days using Seaborn')
plt.show()

# Pie Chart
category_counts = df['day'].value_counts()
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Pie Chart of Days')
plt.show()

# Box Plot
plt.boxplot(df['total_bill'], vert=False)
plt.title('Box Plot of Total Bill')
plt.xlabel('Total Bill')
plt.show()

sns.boxplot(data=df, x='total_bill', palette='Set2')
plt.title('Box Plot of Total Bill using Seaborn')
plt.show()

# Violin Plot
plt.violinplot(df['total_bill'])
plt.title('Violin Plot of Total Bill')
plt.xlabel('Total Bill')
plt.show()

sns.violinplot(data=df, x='day', y='total_bill', palette='muted')
plt.title('Violin Plot of Total Bill by Day')
plt.show()

# Regression Plot
sns.regplot(x='total_bill', y='tip', data=df, scatter_kws={'s': 50}, line_kws={'color': 'red'})
plt.title('Regression Plot of Total Bill vs Tip')
plt.show()