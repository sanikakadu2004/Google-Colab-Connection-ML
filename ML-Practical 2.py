
# Load the dataset
from google.colab import files
uploaded = files.upload()
# Function to load CSV file using Pandas
def load_csv(filepath):
    return pd.read_csv(filepath)
import pandas as pd
sales_data = pd.read_csv('Market Sales Dataset.csv')
sales_data.head()

# Basic Data Cleaning
def clean_and_modify_data(df):
    # Fill missing values in Item_Weight with the mean
    df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
    # Replace 0 values in Item_Visibility with the mean 
    df['Item_Visibility'] = df['Item_Visibility'].replace(0, df['Item_Visibility'].mean())
    # Fill missing values in Outlet_Size with the most frequent value (mode)
    df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])
    # Normalize Item_Fat_Content to make it consistent
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})

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

# Function to export the updated data
def export_data(df, updated_sales_data):
    try:
        df.to_csv(updated_sales_data, index=False)
        print(f"Updated data exported successfully to {updated_sales_data}.")
    except Exception as e:
        print(f"Error exporting data: {e}")

# Main execution block
if __name__ == "__main__":
    input_file = "Market Sales Dataset.csv"  # Replace with the actual file path
    output_file = "updated_sales_data.csv"

    # Import data
    sales_data = load_csv(input_file)

    if sales_data is not None:
        # Perform data cleaning and modification
        updated_data = clean_and_modify_data(sales_data)

        if updated_data is not None:
            # Export updated data
            export_data(updated_data, output_file)

from google.colab import files
# Download the updated CSV file
files.download('updated_sales_data.csv')
