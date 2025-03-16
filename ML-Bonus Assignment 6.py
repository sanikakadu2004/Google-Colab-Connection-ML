
# Load the dataset
from google.colab import files
uploaded = files.upload()
# Function to load CSV file using Pandas
import pandas as pd
df = pd.read_csv('Wine Quality Dataset.csv')
df.head()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 2. Preprocess the data
# Fill missing values with the median (for numerical columns)
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode the categorical feature 'type' (e.g., red = 0, white = 1)
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Separate features and target variable (quality)
X = df.drop(columns=['quality'])
y = df['quality']

# Split the dataset into training (80%) and testing (20%) sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train a Decision Tree Classifier
# Limiting the tree depth to 3 for easier visualization
clf = DecisionTreeClassifier(random_state=42, max_depth=2)
clf.fit(X_train_scaled, y_train)

# Visualize the Decision Tree
plt.figure(figsize=(30, 10))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=[str(c) for c in sorted(y.unique())],
    filled=True,
    rounded=True
)
plt.title("Decision Tree Visualization (max_depth=3)")
plt.show()

# 4. Evaluate the model's performance on the test set
y_pred = clf.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Initial Decision Tree Performance:")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)

# 5. Hyperparameter Tuning with GridSearchCV
param_grid = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

# Get the best model and its performance
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)

best_accuracy = accuracy_score(y_test, y_pred_best)
best_precision = precision_score(y_test, y_pred_best, average='weighted', zero_division=1)
best_recall = recall_score(y_test, y_pred_best, average='weighted')
best_f1 = f1_score(y_test, y_pred_best, average='weighted')

print("
Best Hyperparameters:", grid_search.best_params_)
print("Best Model Performance:")
print("Accuracy :", best_accuracy)
print("Precision:", best_precision)
print("Recall   :", best_recall)
print("F1-score :", best_f1)
