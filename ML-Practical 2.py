
# Load the dataset
from google.colab import files
uploaded = files.upload()
# Function to load CSV file using Pandas
import pandas as pd
sales_data = pd.read_csv('Market Sales Dataset.csv')
sales_data.head()

# Fill missing values in Item_Weight with the mean
sales_data['Item_Weight'].fillna(sales_data['Item_Weight'].mean(), inplace=True)

# Normalize Item_Fat_Content to make it consistent
sales_data['Item_Fat_Content'].replace({
    'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'
}, inplace=True)
sales_data.head(30)
# Data Modifications
# Add a New Column for Sales Tax (e.g., 5% of Item_Outlet_Sales)
sales_data['Sales_Tax'] = sales_data['Item_Outlet_Sales'] * 0.05

# Add a new column for total amount
sales_data['Total_Amount'] = sales_data['Item_Outlet_Sales'] + sales_data['Sales_Tax']

# Display the first 30 rows of the dataset
sales_data.head(30)
# Export the updated data to a new CSV file
sales_data.to_csv("updated_sales_data.csv", index=False)
print("Updated data exported successfully!")
print("Data cleaned and modified successfully.")
from google.colab import files
# Download the updated CSV file
files.download('updated_sales_data.csv')

