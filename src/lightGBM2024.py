import pandas as pd
import numpy as np
import lightgbm as lgb # Assuming 'model' is your loaded LGBM model
import os


# Absolute path
FILE_PATH = 'data/01_input_history.csv'
abs_path = os.path.abspath(os.path.join("..", FILE_PATH))


if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Il file {FILE_PATH} non esiste.")


class Preprocess:
    
    def __init__(self, data):
        self.rolled_data = None
        self.df = data.copy()
    
    def get_data(self):
        return self.df
    
    def rolling_stats(self, window=3):
        """
        Create lagged features and rolling statistics for the time series.
        We assume we still have the columns ['Country', 'Product', 'Year', 'Month', 'Quantity'] 
        in self.data or some variant of these. 
        """
        df = self.df.copy()
        
        # Ensure we have numeric Year, Month for sorting
        # (Adapt if your data uses 'Date' or some other date-like column.)
        df.sort_values(by=['Country', 'Product', 'Year', 'Month'], inplace=True)

        # Group by Country, Product to treat each as a separate time series
        group_cols = ['Country', 'Product']
        
        # Create lag features (shift the Quantity by 1, 2, 3 months, etc.)
        df['lag_1'] = df.groupby(group_cols)['Quantity'].shift(1)
        df['lag_2'] = df.groupby(group_cols)['Quantity'].shift(2)
        df['lag_3'] = df.groupby(group_cols)['Quantity'].shift(3)

        # Example rolling features: rolling mean, rolling std
        # We'll do rolling after shift(1) so the current row isn't included in its own average.
        # But a simpler approach is to directly use the .rolling() on the unshifted 'Quantity' column,
        # just be mindful to set min_periods if needed.
        
        # rolling_mean
        df['rolling_mean_6'] = (
            df.groupby(group_cols)['Quantity']
              .rolling(window=window, min_periods=1)
              .mean()
              .reset_index(level=group_cols, drop=True)
        )
        
        # rolling_std
        df['rolling_std_6'] = (
            df.groupby(group_cols)['Quantity']
              .rolling(window=window, min_periods=1)
              .std()
              .reset_index(level=group_cols, drop=True)
        )

        # Assign the resulting DataFrame to self.rolled_data
        self.rolled_data = df
    
    def preprocess_data(self):
        month_map = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
            "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
        }
        
        def parse_month_year(mmyyyy):
            # First 3 letters => month abbreviation
            month_abbrev = mmyyyy[:3]
            # The rest => year string
            year_str = mmyyyy[3:]
            # Convert the abbreviation and year to integer
            year = int(year_str)
            month = month_map[month_abbrev]
            # Construct a datetime (use day=1 as a placeholder)
            return pd.Timestamp(year=year, month=month, day=1)
        
        self.df['Date'] = self.df['Month'].apply(parse_month_year)

        # Create separate columns for year and month if desired
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month

        self.df = self.df[self.df['Year'] < 2024]
        
        
        self.rolling_stats(window=6)
        
        self.df = self.rolled_data.dropna(subset=['lag_1', 'lag_2', 'lag_3', 'rolling_mean_6', 'rolling_std_6'])
        

    
    def save_on_csv(self, path):
        """
        Save the processed DataFrame to a CSV file.
        """
        self.df.to_csv(path, index=False)



data = pd.read_csv(FILE_PATH)
    
preprocess = Preprocess(data)
preprocess.preprocess_data()
df_history = preprocess.get_data().copy()

# Save the processed data to a CSV file
preprocess.save_on_csv('data/processed_data_2.csv')

# 3. Engineer Cyclical Month Features
df_history['month_sin'] = np.sin(2 * np.pi * df_history['Month']/12)
df_history['month_cos'] = np.cos(2 * np.pi * df_history['Month']/12)
#df_history['time_idx'] = range(len(df_history))

# --- Update your feature list ---
# Original features might include lags, rolling stats, OHE/categorical Country/Product
original_features = ['Year', 'Month', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean_6', 'rolling_std_6', 'Country', 'Product', ...] # Use original Country/Product if using category dtypes
time_features = ['month_sin', 'month_cos']

df_history = df_history.sort_values(by='Date')
last_date = df_history['Date'].max() # Assuming a 'Date' column/index
unique_combinations = df_history[['Country', 'Product']].drop_duplicates().to_records(index=False)

df_working = df_history.copy() # Copy to append predictions

future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')


# MODEL
target = 'Quantity'

# Feature columns (all columns except the target)
train_features = [col for col in df_history.columns if col != target and col != 'Date']

X = df_history[train_features]
y = df_history[target]

# Define model parameters (you'll likely need to tune these)
params = {
    'objective': 'regression_l1', # L1 loss (Mean Absolute Error) is often more robust to outliers than L2 (MSE)
                                  # Consider 'regression' (L2) or implementing your custom objective if possible
    'metric': 'mae',              # Use 'mae' or 'rmse' for monitoring. Evaluation needs your custom metric.
    'n_estimators': 1000,         # Number of boosting rounds
    'learning_rate': 0.05,
    'feature_fraction': 0.8,      # Fraction of features considered per iteration
    'bagging_fraction': 0.8,      # Fraction of data used per iteration (requires bagging_freq > 0)
    'bagging_freq': 1,
    'verbose': -1,                # Suppress verbose output
    'n_jobs': -1,                 # Use all available CPU cores
    'seed': 42,
    'boosting_type': 'gbdt',
    # Add other parameters like 'num_leaves', 'max_depth', etc. for tuning
}

# Initialize the model
model = lgb.LGBMRegressor(**params)

X['Country'] = X['Country'].astype('category')
X['Product'] = X['Product'].astype('category')

model.fit(X, y, categorical_feature=['Country', 'Product'])


print("Starting recursive forecasting...")
all_predictions = []
for target_date in future_dates:
    target_year = target_date.year
    target_month = target_date.month
    print(f"  Predicting for: {target_year}-{target_month:02d}")

    # Create a temporary DataFrame to hold features for this month's predictions
    features_this_month = []
    rows_to_predict = [] # Store identifying info for merging predictions back

    for country, product in unique_combinations:
        # Isolate data for this specific series
        series_data = df_working[(df_working['Country'] == country) &
                                 (df_working['Product'] == product)].set_index('Date').sort_index()

        # --- Calculate Features ---
        features = {}
        features['Year'] = target_year
        features['Month'] = target_month
        features['Country'] = country  # Add categorical feature directly
        features['Product'] = product  # Add categorical feature directly

        # Lags (handle potential errors if history is short)
        try:
            features['lag_1'] = series_data['Quantity'].iloc[-1]
        except IndexError: features['lag_1'] = 0
        try:
            features['lag_2'] = series_data['Quantity'].iloc[-2]
        except IndexError: features['lag_2'] = 0
        try:
            features['lag_3'] = series_data['Quantity'].iloc[-3]
        except IndexError: features['lag_3'] = 0

        # Rolling Stats (handle potential errors if history is short)
        recent_6_months = series_data['Quantity'].iloc[-6:]
        if len(recent_6_months) > 1: # Need at least 2 points for std dev
             features['rolling_mean_6'] = recent_6_months.mean()
             features['rolling_std_6'] = recent_6_months.std()
        elif len(recent_6_months) == 1:
             features['rolling_mean_6'] = recent_6_months.mean()
             features['rolling_std_6'] = 0
        else:
             features['rolling_mean_6'] = 0
             features['rolling_std_6'] = 0

        # Fill NaNs that might result from std dev of constant series, etc.
        features['rolling_std_6'] = np.nan_to_num(features['rolling_std_6'])

        # Append feature dictionary
        features_this_month.append(features)
        rows_to_predict.append({'Country': country, 'Product': product, 'Year': target_year, 'Month': target_month})

    # Create the DataFrame for this month's predictions
    features_this_month = pd.DataFrame(features_this_month)

    # Ensure all expected columns are present in the DataFrame
    for col in train_features:
        if col not in features_this_month.columns and not (col.startswith('Country_') or col.startswith('Product_')):
            features_this_month[col] = 0  # Fill missing non-hot-encoded columns with 0

    features_this_month['Country'] = features_this_month['Country'].astype('category')
    features_this_month['Product'] = features_this_month['Product'].astype('category')

    # Make predictions for this month
    predictions = model.predict(features_this_month[train_features])

    # Create a DataFrame of predictions for this month
    predictions_df = pd.DataFrame(rows_to_predict)
    predictions_df['Quantity'] = np.round(predictions).astype(int)
    predictions_df['Date'] = target_date

    # Append predictions to the overall list
    all_predictions.append(predictions_df)

    # Update df_working with the new predictions (for the next iteration's lag features)
    df_working = pd.concat([df_working, predictions_df[['Date', 'Country', 'Product', 'Quantity']]], ignore_index=True)
    df_working['Year'] = df_working['Date'].dt.year
    df_working['Month'] = df_working['Date'].dt.month

print("Recursive forecasting complete.")

# --- Format Final Output ---
final_predictions = pd.concat(all_predictions, ignore_index=True)
output_df = final_predictions[['Country', 'Product', 'Month', 'Quantity']]

# Save the output
output_filename = '01_output_prediction_CODE.csv'
output_df.to_csv(output_filename, sep=',', index=False)

print(f"Predictions saved to {output_filename}")