
# Load the dataset
from google.colab import files
uploaded = files.upload()
# Function to load CSV file using Pandas
import pandas as pd
df = pd.read_csv('Titanic Dataset.csv')
df.head()

import numpy as np
# Check for missing values in the dataset
print("Checking for missing values:")
print(df.isnull().sum())

# Fill missing values in 'Age' with the mean
df['Age'] = df['Age'].fillna(df['Age'].mean())
# Fill missing values in 'Embarked' with the mode (most frequent value)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# Check if 'Cabin' column exists before attempting to drop it
if 'Cabin' in df.columns:
    # Drop rows with missing 'Cabin' values as it may not be a critical feature for analysis
    df.drop(columns=['Cabin'], inplace=True)
else:
    print("Column 'Cabin' not found in the DataFrame.")
# Alternatively, fill missing 'Fare' values with the median
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Display the cleaned dataset 
print(df.head(30))
# Check if any missing values remain
print("
Checking for missing values after handling:")
print(df.isnull().sum())

# Perform basic EDA
import matplotlib.pyplot as plt
import seaborn as sns

# Basic statistics summary for numerical columns
print(df.describe())
# Count of unique values for categorical columns
print("Pclass value counts:", df['Pclass'].value_counts())
print("Sex value counts:", df['Sex'].value_counts())
print("Embarked value counts:", df['Embarked'].value_counts())

# Visualize the distribution of numerical columns
for column in df.columns:  
    plt.figure(figsize=(6, 4))
    sns.histplot(df[column], kde=True, bins=20, color='blue')
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()  # Adjust spacing for each individual plot
    plt.show()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# Visualize only some columns
# Visualize the distribution of 'Age'
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.show()

# Visualize the distribution of 'Fare'
sns.histplot(df['Fare'], kde=True, bins=30)
plt.title('Fare Distribution')
plt.show()

# Countplot for the 'Survived' column
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

# Countplot for the 'Sex' column
sns.countplot(x='Sex', data=df)
plt.title('Gender Distribution')
plt.show()

# Export the cleaned dataset to a new CSV file
cleaned_file_path = 'cleaned_titanic.csv'  # Define the variable with the file path
df.to_csv(cleaned_file_path, index=False)
print(f"
Cleaned data exported to: {cleaned_file_path}")
