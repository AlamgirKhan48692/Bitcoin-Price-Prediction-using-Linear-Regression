
import pandas as pd

# Load the dataset
df = pd.read_csv('BitcoinPrediction.csv')
df.head()


#Preprocessing

# Convert 'Open Time' to datetime
df['Open Time'] = pd.to_datetime(df['Open Time'])

# Sort by date
df = df.sort_values('Open Time')

# Create target variable: next day's Close
df['Target'] = df['Close'].shift(-1)

# Drop last row (NaN target)
df = df.dropna()

# Select features for linear regression
features = ['Open', 'High', 'Low', 'Volume']
X = df[features]
y = df['Target']


#Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)



#Train Linear Regression Model

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)


# Evaluate the Model

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# Optional: Visualize Results

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Bitcoin Closing Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
