
# Load the dataset
from google.colab import files
uploaded = files.upload()
# Function to load CSV file using Pandas
import pandas as pd
df = pd.read_csv('E-commerce.csv')
df.head()

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to extract number of purchases from Purchase History
def extract_purchase_count(purchase_history):
    try:
        purchases = json.loads(purchase_history.replace("'", """))
        return len(purchases) if isinstance(purchases, list) else 1
    except:
        return 0

# Function to extract number of browsed items from Browsing History
def extract_browsing_count(browsing_history):
    try:
        history = json.loads(browsing_history.replace("'", """))
        return len(history) if isinstance(history, list) else 1
    except:
        return 0

# Apply feature extraction functions
df['Purchase Count'] = df['Purchase History'].apply(extract_purchase_count)
df['Browsing Count'] = df['Browsing History'].apply(extract_browsing_count)

# Encoding categorical variables
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_location = LabelEncoder()
df['Location'] = le_location.fit_transform(df['Location'])

# Defining features and target variable
X = df[['Age', 'Gender', 'Location', 'Annual Income', 'Time on Site', 'Purchase Count', 'Browsing Count']]
y = (df['Purchase Count'] > 0).astype(int)  # Binary target: 1 if purchase made, 0 otherwise

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building and training the Decision Tree model
dt_model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Making predictions
y_pred = dt_model.predict(X_test)

# Accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print("
Accuracy:", accuracy)
print('
Classification Report:
', classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Visualizing the Decision Tree
plt.figure(figsize=(5, 3))
plot_tree(dt_model, feature_names=X.columns, class_names=['No Purchase', 'Purchase'], filled=True, rounded=True)
plt.show()
