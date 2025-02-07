
# Load the dataset
from google.colab import files
uploaded = files.upload()
# Function to load CSV file using Pandas
def load_csv(filepath):
    return pd.read_csv(filepath)
import pandas as pd
df = pd.read_csv('Housing Dataset.csv')
df.head()

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# Selecting relevant features (area, bedrooms) and target (price)
X = df[['area', 'bedrooms']]  # Features: Square footage, number of rooms
y = df['price']  # Target variable: Price of the house

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. KNN Model with different values of k 
k_values = [3, 5, 7, 9, 11]  # Experimenting with different k values
knn_mae = []
knn_mse = []
knn_rmse = []

# Train and evaluate KNN for each value of k
for k in k_values:
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred_knn = knn_model.predict(X_test_scaled)

    # Evaluate KNN model using MAE, MSE, RMSE
    mae_knn = mean_absolute_error(y_test, y_pred_knn)
    mse_knn = mean_squared_error(y_test, y_pred_knn)
    rmse_knn = mse_knn ** 0.5

    # Store the metrics
    knn_mae.append(mae_knn)
    knn_mse.append(mse_knn)
    knn_rmse.append(rmse_knn)

# 2. Linear Regression Model 
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_lr = linear_model.predict(X_test_scaled)

# Evaluate Linear Regression model using MAE, MSE, RMSE
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = mse_lr ** 0.5

# 3. Comparing KNN with Linear Regression 
print("Linear Regression Performance:")
print(f"MAE: {mae_lr}, MSE: {mse_lr}, RMSE: {rmse_lr}
")

# Output KNN performance for each k
print("KNN Model Performance (for different values of k):")
for i, k in enumerate(k_values):
    print(f"KNN (k={k}) - MAE: {knn_mae[i]}, MSE: {knn_mse[i]}, RMSE: {knn_rmse[i]}")

# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
# 4. Visualization: Plot K vs RMSE (or any other metric) 
plt.figure(figsize=(10, 6))

# Plot RMSE for different k values (KNN)
plt.plot(k_values, knn_rmse, label="KNN RMSE", marker='o', color='blue')

# Plot RMSE for Linear Regression
plt.axhline(rmse_lr, label="Linear Regression RMSE", color='red', linestyle='--')

# Labels and title
plt.title("Relationship Between k and Model Accuracy (RMSE)")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("RMSE")
plt.legend()
# Show the plot
plt.show()
