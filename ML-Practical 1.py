
# Load the dataset
from google.colab import files
uploaded = files.upload()
# Function to load CSV file using Pandas
import pandas as pd
dataset = pd.read_csv('Iris Plant Dataset.csv')
dataset.head()

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
# Features (sepal length, sepal width, petal length, petal width)
X = iris.data
# Target labels (species)
y = iris.target
# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Feature names and target names
feature_names = iris.feature_names
target_names = iris.target_names

print("Features:", feature_names)
print("Target classes:", target_names)

# Use numpy for Data Processing
import numpy as np
print("Mean of features:
", np.mean(X, axis=0))
print("Standard deviation of features:
", np.std(X, axis=0))

# Filter rows where petal length (3rd column) is greater than 2
filtered_data = X[X[:, 2] > 2]
filtered_labels = y[X[:, 2] > 2]

# Combine filtered data and labels
labeled_data = np.column_stack((filtered_data, filtered_labels))

# Print the filtered dataset with labels
print("Filtered labeled dataset (petal length > 2):")
print(f"{'Sepal Length':<15}{'Sepal Width':<15}{'Petal Length':<15}{'Petal Width':<15}{'Label'}")
for row in labeled_data:
    print(f"{row[0]:<15.2f}{row[1]:<15.2f}{row[2]:<15.2f}{row[3]:<15.2f}{target_names[int(row[4])]:<15}")

# Use matplotlib for Visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
for i, target_name in enumerate(target_names):
    plt.scatter(
        X[y == i, 0],  # Sepal length
        X[y == i, 1],  # Sepal width
        label=target_name
    )
plt.title("Sepal Length vs. Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend()
plt.show()

import tensorflow as tf
# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes for Iris dataset
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Train the model with validation data
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# Plot the training and validation loss/accuracy over epochs
plt.figure(figsize=(12, 6))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model performance on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')

# Make predictions
y_pred = model.predict(X_test)

# Convert the predicted values to class labels
y_pred_class = np.argmax(y_pred, axis=1)

# Display first few predictions
print("Predicted Class Labels:", y_pred_class[:10])
