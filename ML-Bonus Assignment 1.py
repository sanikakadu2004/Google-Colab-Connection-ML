
# Load the dataset using pandas
import pandas as pd 
# Load the dataset using pandas
heart_data = pd.read_csv('Heart Disease Dataset.csv') 
display(heart_data)

# Use Seaborn libararies for random forest classifier
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Prepare the features (X) and target (y)
X = heart.drop(columns=["target"])
y = heart["target"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a classifier (e.g., Random Forest)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)

# Feature for visualization
import matplotlib.pyplot as plt
# Feature importance visualization
importance = clf.feature_importances_
feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": importance})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Library for test accuracy & classification
from sklearn.metrics import accuracy_score, classification_report

# Make predictions and evaluate the model
y_pred = clf.predict(X_test_scaled)

# Calculate accuracy & classification
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:
", classification_report(y_test, y_pred))
