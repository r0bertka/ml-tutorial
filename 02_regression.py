import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

path = 'data/AB_NYC_2019.txt'
df = pd.read_csv(path)

# Question 1

df = df[['latitude',
         'longitude',
         'price',
         'minimum_nights',
         'number_of_reviews',
         'reviews_per_month',
         'calculated_host_listings_count',
         'availability_365']]

missing_value_count_per_column = df.isna().sum()
cols_w_missing_values = missing_value_count_per_column[missing_value_count_per_column > 0]

print(f'Columns with missing values:\n {cols_w_missing_values}')


# Question 2

median_min_nights = df['minimum_nights'].median()
ax = df['minimum_nights'].plot(kind='hist', bins=100, logy=True)
plt.show()

df_sampled = df.sample(frac=1, random_state=42)
X = df.drop('price', axis=1)
y = np.log1p(df['price'])

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5)



# Question 3


