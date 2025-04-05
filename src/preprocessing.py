import numpy
import pandas as pd
import sklearn
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical

# Absolute path
FILE_PATH = 'data/01_input_history.csv'
abs_path = os.path.abspath(os.path.join("..", FILE_PATH))


if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Il file {FILE_PATH} non esiste.")

data = pd.read_csv(FILE_PATH)

class Preprocess:
    
    def __init__(self, data):
        self.data = data.copy()
        self.rolled_data = None
        self.df = None
    
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
        self.df = self.data.copy()
        
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
        
        self.df['Month'] = self.df['Month'].apply(parse_month_year)

        # Create separate columns for year and month if desired
        self.df['Year'] = self.df['Month'].dt.year
        self.df['Month'] = self.df['Month'].dt.month
        
        self.rolling_stats(window=6)
        
        self.df = self.rolled_data.dropna(subset=['lag_1', 'lag_2', 'lag_3', 'rolling_mean_6', 'rolling_std_6'])
        
        self.df = pd.get_dummies(self.df, columns=['Country', 'Product'], drop_first=False)
                
    def split_test_train(self, test_year=2023, target_col='Quantity'):
        """
        Splits the data into train and test sets based on the year.
        """
        train_data = self.df[self.df['Year'] < test_year]
        test_data = self.df[self.df['Year'] == test_year]
        
        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col]
        X_test = test_data.drop(columns=[target_col])
        y_test = test_data[target_col]
        
        return X_train, y_train, X_test, y_test
                
preprocess = Preprocess(data)
preprocess.preprocess_data()
processed_data = preprocess.get_data()

X_train, y_train, X_test, y_test = preprocess.split_test_train()

print(X_train.head(1))
print(y_train.head(1))
print(X_test.head(1))
print(y_test.head(1))




# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural Network Architecture
model = models.Sequential()

# Input layer (shape is the number of features)
model.add(layers.Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(layers.Dropout(0.2))  # Dropout for regularization
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))  # Dropout for regularization
model.add(layers.Dense(32, activation='relu'))

# Output layer (single neuron for regression, no activation function)
model.add(layers.Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

# Make predictions
y_pred = model.predict(X_test)