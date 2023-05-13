import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv("hacktoncleanedDF.csv", low_memory=False)
df.drop(columns="Unnamed: 0", inplace=True)
df.reset_index(inplace=True, drop=True)

# Group by pickup hour and count
pickup_counts = df.groupby(['pickupDay', 'pickupDayName', 'pickupHour', 'pickupMonth', 'is_holiday', 'is_tourist_season', 'rush_hour_ride', 'season'])

# Calculate total count for each hour and day
df_4 = pickup_counts.size().reset_index(name='count')
print(len(df_4))

# Create a list of input/output pairs
days = []
for _, row in df_4.iterrows():
    days.append([[row['pickupHour'], row['pickupDay'], row['pickupMonth'], row['is_tourist_season'], row['is_holiday'], row['season'], row['rush_hour_ride']], row['count']])

import random
random_rows = random.sample(days, k=40)
# Save the randomly selected rows to a DataFrame
random_df = pd.DataFrame(random_rows)
#random_df.to_csv('TestV2.csv', index=False)
# Create a new list without the randomly selected rows
remaining_rows = [row for row in days if row not in random_rows]

# Create input and output DataFrames
X = pd.DataFrame([day[0] for day in remaining_rows], columns=['pickupHour', 'pickupDay', 'pickupMonth', 'is_tourist_season', 'is_holiday', 'season', 'rush_hour_ride'])
y = pd.DataFrame([day[1] for day in remaining_rows], columns=['count'])

# Convert count column to int type
y['count'] = y['count'].astype('int64')
# Convert object type columns to appropriate types
X['pickupHour'] = X['pickupHour'].astype('int64')
X['pickupDay'] = X['pickupDay'].astype('int64')
X['pickupMonth'] = X['pickupMonth'].astype('int64')
X['rush_hour_ride'] = X['rush_hour_ride'].replace({True: 1, False: 0})
X['is_tourist_season'] = X['is_tourist_season'].replace({True: 1, False: 0})
X['is_holiday'] = X['is_holiday'].replace({True: 1, False: 0})
X['season'] = X['season'].replace({'Winter': 1, 'Spring': 2, 'Summer' : 3, 'Fall' : 4})

# Normalize the input data using MinMaxScaler
scaler = MinMaxScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])
# Create training and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
#Build and fit KNN model
# Perform elbow method to find optimal value of K
k_values = range(1, 21)  # Range of K values to consider
mse_values = []  # List to store mean squared errors for different K values

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    mse = np.mean((knn.predict(X_test) - y_test) ** 2)
    mse_values.append(mse)

# Find the optimal value of K
optimal_k = np.argmin(mse_values) + 1
# Plot K values vs. MSE
plt.plot(k_values, mse_values, 'bo-')
plt.xlabel('K')
plt.ylabel('Mean Squared Error')
plt.title('Elbow Method for Optimal K')
plt.show()
knn = KNeighborsRegressor(n_neighbors=optimal_k)
knn.fit(X_train, y_train)

# Evaluate the model
train_score = knn.score(X_train, y_train)
test_score = knn.score(X_test, y_test)
print(f"Train score: {train_score:.3f}, Test score: {test_score:.3f}")

random_X = pd.DataFrame([row[0] for row in random_rows], columns=['pickupHour', 'pickupDay', 'pickupMonth', 'is_tourist_season', 'is_holiday', 'season', 'rush_hour_ride'])
random_y = pd.DataFrame([row[1] for row in random_rows], columns=['count'])
random_y['count'] = random_y['count'].astype('int64')
random_X['pickupHour'] = random_X['pickupHour'].astype('int64')
random_X['pickupDay'] = random_X['pickupDay'].astype('int64')
random_X['pickupMonth'] = random_X['pickupMonth'].astype('int64')
random_X['rush_hour_ride'] = random_X['rush_hour_ride'].replace({True: 1, False: 0})
random_X['is_tourist_season'] = random_X['is_tourist_season'].replace({True: 1, False: 0})
random_X['is_holiday'] = random_X['is_holiday'].replace({True: 1, False: 0})
random_X['season'] = random_X['season'].replace({'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4})
random_X[random_X.columns] = scaler.transform(random_X[random_X.columns])

predictions = knn.predict(random_X)

# Calculate R-squared score
r2_score = knn.score(random_X, random_y)
print(f"External data test R-squared score: {r2_score:.3f}")

# Calculate the absolute differences between predicted count and actual count
abs_diff = abs(random_y['count'].values.flatten() - predictions.flatten())

# Create a DataFrame with columns for the actual count, predicted count, and score
results_df = pd.DataFrame({
    'count': random_y['count'].values.flatten(),
    'predicted_count': predictions.flatten(),
    'score': abs_diff
})


results_df.to_csv("resultsDF.csv")
