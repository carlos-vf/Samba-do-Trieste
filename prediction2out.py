import pandas as pd

# Load your data
df = pd.read_csv("predictions_1.csv")

# Convert Month and Year to string format like "Jan2004"
df['Month'] = pd.to_datetime(df['Month'], format='%m').dt.strftime('%b')
df['Month'] = df['Month'] + df['Year'].astype(str)

# Get all columns that start with 'Country_' or 'Product_'
country_cols = [col for col in df.columns if col.startswith('Country_')]
product_cols = [col for col in df.columns if col.startswith('Product_')]

# Get the actual country and product names from column names
countries = [col.replace('Country_', '') for col in country_cols]
products = [col.replace('Product_', '') for col in product_cols]

# Collect transformed rows here
rows = []

# Loop through each row of the DataFrame
for _, row in df.iterrows():
    month = row['Month']
    quantity = row['Quantity']
    
    # Find the country that is True
    country = next((c for col, c in zip(country_cols, countries) if row[col] == True), None)
    
    # Find the product that is True
    product = next((p for col, p in zip(product_cols, products) if row[col] == True), None)

    # If both found, add to the result
    if country and product:
        rows.append({
            'Country': country,
            'Product': product,
            'Month': month,
            'Quantity': quantity
        })

# Create a new DataFrame
df_long = pd.DataFrame(rows)

# Save the result
df_long.to_csv("01_output_prediction_8475.csv", index=False)
