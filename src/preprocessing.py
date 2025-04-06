import numpy as np
import pandas as pd
import sklearn
import os
import sys
from pathlib import Path

# Absolute path
INPUT_HISTORY_CSV = '01_input_history.csv'
PROCESSED_DATA_CSV = 'processed_data.csv'

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"

abs_path = os.path.abspath(os.path.join("..", INPUT_HISTORY_CSV))

if not os.path.exists(DATA_DIR / INPUT_HISTORY_CSV):
    raise FileNotFoundError(f"Il file {DATA_DIR / INPUT_HISTORY_CSV} non esiste.")

class Preprocess:
    
    def __init__(self, data):
        self.rolled_data = None
        self.df = data.copy()
    
    def get_data(self):
        return self.df
    
    def parse_dates(self, prophet=False):
        """Parses the Month column into Year and Month numbers."""
        month_map = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
            "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
        }

        def parse_month_year(mmyyyy):
            month_abbrev = mmyyyy[:3]
            year_str = mmyyyy[3:]
            year = int(year_str)
            month = month_map[month_abbrev]
            return year, month

        if prophet:
            # If Month is already numeric, just build ds. Otherwise parse the text format.
            if self.df['Month'].dtype == 'int':
                self.df['ds'] = pd.to_datetime(self.df['Year'].astype(str) + '-' + 
                                               self.df['Month'].astype(str).str.zfill(2) + '-01')
                self.df['y'] = self.df['Quantity']
                self.df.drop(columns=['Quantity', 'Year', 'Month'], inplace=True)
            else:
                parsed_dates = self.df['Month'].apply(lambda x: pd.Series(parse_month_year(x), 
                                                                          index=['Year', 'Month_Num']))
                self.df['Year'] = parsed_dates['Year']
                self.df['Month'] = parsed_dates['Month_Num']
                self.df['ds'] = pd.to_datetime(self.df['Year'].astype(str) + '-' + 
                                               self.df['Month'].astype(str).str.zfill(2) + '-01')
                self.df['y'] = self.df['Quantity']
                self.df.drop(columns=['Quantity', 'Year', 'Month'], inplace=True)
        else:
            parsed_dates = self.df['Month'].apply(lambda x: pd.Series(parse_month_year(x), 
                                                                      index=['Year', 'Month_Num']))
            self.df['Year'] = parsed_dates['Year']
            self.df['Month'] = parsed_dates['Month_Num']
            self.df['month_sin'] = np.sin(2 * np.pi * parsed_dates['Month_Num']/12)
            self.df['month_cos'] = np.cos(2 * np.pi * parsed_dates['Month_Num']/12)
            
        
    def rolling_stats(self, window=3):
        """Creates lagged and rolling features."""
        # Ensure data is sorted for correct lagging/rolling
        self.df.sort_values(by=['Country', 'Product', 'Year', 'Month'], inplace=True)
        group_cols = ['Country', 'Product']

        self.df['lag_1'] = self.df.groupby(group_cols)['Quantity'].shift(1)
        self.df['lag_2'] = self.df.groupby(group_cols)['Quantity'].shift(2)
        self.df['lag_3'] = self.df.groupby(group_cols)['Quantity'].shift(3)

        # Use transform for potentially cleaner rolling calculations
        self.df['rolling_mean_6'] = self.df.groupby(group_cols)['Quantity'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        self.df['rolling_std_6'] = self.df.groupby(group_cols)['Quantity'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        # Fill initial NaNs in std dev (first element of each group)
        self.df['rolling_std_6'] = self.df.groupby(group_cols)['rolling_std_6'].ffill()
    
    def handle_dummies(self):
        """Creates one-hot encoded features."""
        self.df = pd.get_dummies(self.df, columns=['Country', 'Product'], drop_first=False)

    def drop_na_features(self):
        """Drops rows with NaN in lag/rolling features."""
        feature_cols = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_6', 'rolling_std_6']
        self.df.dropna(subset=feature_cols, inplace=True)
        
    def preprocess_data(self, window=6):
        """Consolidated preprocessing steps."""
        self.parse_dates()
        self.rolling_stats(window=window)
        self.drop_na_features() # Drop NAs *after* creating features
        self.handle_dummies()
                
    def split_test_train(data, val_year = 2022, test_year=2023, target_col='Quantity'):
        """
        Splits the data into train and test sets based on the year.
        """
        train_data = data[(data['Year'] != val_year) & (data['Year'] != test_year)]
        val_data = data[data['Year'] == val_year]
        test_data = data[data['Year'] == test_year]
        
        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col]
        X_val = val_data.drop(columns=[target_col])
        y_val = val_data[target_col]
        X_test = test_data.drop(columns=[target_col])
        y_test = test_data[target_col]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def save_on_csv(self, path):
        """
        Save the processed DataFrame to a CSV file.
        """
        self.df.to_csv(path, index=False)
                
if __name__ == "__main__":
    data = pd.read_csv(DATA_DIR / INPUT_HISTORY_CSV)
    
    preprocess = Preprocess(data)
    preprocess.preprocess_data()
    processed_data = preprocess.get_data()
    
    # Save the processed data to a CSV file
    preprocess.save_on_csv('data/processed_data.csv')

    X_train, y_train, X_val, y_val, X_test, y_test = preprocess.split_test_train()



