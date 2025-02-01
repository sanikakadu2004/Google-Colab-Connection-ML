
# Load the dataset
from google.colab import files
uploaded = files.upload()
# Function to load CSV file using Pandas
def load_csv(filepath):
    return pd.read_csv(filepath)
import pandas as pd
df = pd.read_csv('Bank Customer Churn Dataset.csv')
df.head()

import pandas as pd
import numpy as np
# Fill missing values for numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
categorical_cols = df.select_dtypes(include=[object]).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
# One-hot encoding for categorical variables
df = pd.get_dummies(df, drop_first=True)
# Separate features (X) and target (y)
X = df.drop('churn', axis=1)  # Replace 'churn' with your target column name
y = df['churn']
print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Assign to X_train_scaled
X_test_scaled = scaler.transform(X_test)       # Assign to X_test_scaled
# Feature Selection
selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train_scaled, y_train) # Use X_train_scaled
X_test_selected = selector.transform(X_test_scaled) 
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# Function to evaluate and print metrics
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train) # Train model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    # Print metrics
    print(f"
{model_name} Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Confusion Matrix:
{cm}
")
  # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Churned', 'Churned'], 
                yticklabels=['Not Churned', 'Churned'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()
# Train and evaluate SVM
svm_model = SVC(probability=True, random_state=42)
evaluate_model(svm_model, X_train, X_test, y_train, y_test, "SVM")
# Train and evaluate Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
evaluate_model(gb_model, X_train, X_test, y_train, y_test, "Gradient Boosting")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# Get predicted probabilities for SVM and Gradient Boosting
svm_prob = svm_model.predict_proba(X_test)[:, 1]
gb_prob = gb_model.predict_proba(X_test)[:, 1]
# Calculate ROC curve
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_prob)
fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_prob)
# Calculate AUC
auc_svm = auc(fpr_svm, tpr_svm)
auc_gb = auc(fpr_gb, tpr_gb)
# Plot ROC curves
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.2f})')
plt.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC = {auc_gb:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
# Add labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
y_pred_svm_reg = svm_model.predict(X_test)  # Get predictions for SVM
y_pred_gb_reg = gb_model.predict(X_test)  # Get predictions for Gradient Boosting
# For regression models, calculate MAE, MSE, RMSE, and R-squared
for y_pred_reg, name in zip([y_pred_svm_reg, y_pred_gb_reg], ['SVM Regression', 'Gradient Boosting Regression']):
    mae = mean_absolute_error(y_test, y_pred_reg)
    mse = mean_squared_error(y_test, y_pred_reg)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_reg)
    print(f"{name} Evaluation:")
    print(f"Mean Absolute Error(MAE): {mae:.4f}")
    print(f"Mean Squared Error(MSE): {mse:.4f}")
    print(f"Root Mean Squared Error(RMSE): {rmse:.4f}")
    print(f"R-squared: {r2:.4f}
")

# Residual Plot
residuals = y_test - y_pred_reg# Difference between actual and predicted values
# Create a histogram and KDE plot for the residuals
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel('Residuals')
plt.title('Residual Plot')
plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
# Get probabilities for both models (positive class probabilities)
svm_prob = svm_model.predict_proba(X_test)[:, 1]  # Probability of the positive class for SVM
gb_prob = gb_model.predict_proba(X_test)[:, 1]    # Probability of the positive class for Gradient Boosting
# Calculate precision and recall for both models
precision_svm, recall_svm, _ = precision_recall_curve(y_test, svm_prob)
precision_gb, recall_gb, _ = precision_recall_curve(y_test, gb_prob)
# Calculate Average Precision (AP) for both models
ap_svm = average_precision_score(y_test, svm_prob) 
ap_gb = average_precision_score(y_test, gb_prob)
# Plot Precision-Recall Curve for both models
plt.figure(figsize=(8, 6))
plt.plot(recall_svm, precision_svm, color='darkorange', lw=2, label=f'SVM (AP = {ap_svm:.2f})')
plt.plot(recall_gb, precision_gb, color='blue', lw=2, label=f'Gradient Boosting (AP = {ap_gb:.2f})')
# Add labels and title
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend(loc='lower left')
# Show plot
plt.show()
