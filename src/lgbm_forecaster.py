# lgbm_forecaster.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error # Use MSE as a standard metric too
import joblib # For saving the model
import time
import logging
from pathlib import Path

# Assuming preprocessing.py is in the same directory or Python path
try:
    from preprocessing import Preprocess, DATA_DIR, INPUT_HISTORY_CSV
except ImportError:
    logging.error("Could not import Preprocess class. Make sure preprocessing.py is accessible.")
    raise

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / "output"
MODEL_SAVE_PATH = OUTPUT_DIR / "lgbm_demand_forecast_model.joblib"
PREDICTIONS_SAVE_PATH = OUTPUT_DIR / "01_output_prediction_8475.csv"
TEAM_CODE = "8475"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Custom Scorer for Competition ---
def competition_loss_metric(y_true, y_pred):
    """
    Competition's loss metric calculation.
    Assumes y_true and y_pred are numpy arrays or pandas Series.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure predictions are non-negative integers (as required by competition rules)
    y_pred = np.round(np.maximum(0, y_pred)).astype(int)

    # Calculate denominator based on y_true
    denominator = np.where(y_true != 0, y_true, 1.0) # Use 1.0 for float division

    # Calculate squared error
    squared_error = (y_pred - y_true) ** 2

    # Calculate loss per item
    loss = squared_error / denominator

    # Return the mean loss
    return np.mean(loss)

# Make it usable with GridSearchCV (lower score is better)
competition_scorer = make_scorer(competition_loss_metric, greater_is_better=False)


# --- LightGBM Forecaster Class ---
class LGBMForecaster:
    def __init__(self,
                 model_params=None,
                 train_params=None,
                 categorical_features='auto'): # Let LGBM detect or pass from Preprocess
        """
        Initializes the LightGBM Forecaster.

        Args:
            model_params (dict, optional): Parameters for LGBMRegressor. Defaults to None (uses basic defaults).
            train_params (dict, optional): Parameters for the fit method (e.g., early stopping). Defaults to None.
            categorical_features (list or 'auto', optional): List of categorical feature names. Defaults to 'auto'.
        """
        self.model_params = model_params if model_params else {}
        self.train_params = train_params if train_params else {}
        self.categorical_features = categorical_features
        self.model = None
        self.best_params = None # Store best params from tuning
        self.preprocessor = None # Will hold the Preprocess instance
        self.feature_columns = None # Store feature names used for training

    def _prepare_data(self, input_path=DATA_DIR / INPUT_HISTORY_CSV, **preprocess_args):
        """Loads and preprocesses the data using the Preprocess class."""
        logging.info(f"Loading data from {input_path}")
        raw_data = pd.read_csv(input_path)
        self.preprocessor = Preprocess(raw_data)
        processed_df = self.preprocessor.full_preprocess(**preprocess_args)
        self.feature_columns = self.preprocessor.get_feature_columns()

        # Ensure categorical features are correctly set if using 'auto' or list
        if self.categorical_features == 'auto':
            self.categorical_features = self.preprocessor.categorical_features
        logging.info(f"Using categorical features: {self.categorical_features}")
        logging.info(f"Using feature columns: {self.feature_columns}")

        return processed_df

    def tune_hyperparameters(self,
                             data,
                             param_grid,
                             cv=3,
                             scoring=competition_scorer,
                             val_year=2022,
                             target_col='Quantity',
                             n_jobs = 15):
        """
        Performs GridSearchCV to find the best hyperparameters.

        Args:
            data (pd.DataFrame): The fully preprocessed DataFrame.
            param_grid (dict): Dictionary with parameters names (string) as keys and lists of
                               parameter settings to try as values.
            cv (int): Number of cross-validation folds. TimeSeriesSplit might be better.
            scoring (callable): Scorer to use for evaluation (e.g., competition_scorer).
            val_year (int): Year to use for the validation set in the train/val split.
            target_col (str): Name of the target column.
        """
        logging.info("--- Starting Hyperparameter Tuning ---")
        train_df, val_df = Preprocess.get_train_val_split(data, val_year=val_year)

        X_train = train_df[self.feature_columns]
        y_train = train_df[target_col]
        # X_val = val_df[self.feature_columns] # Not directly used by GridSearchCV folds
        # y_val = val_df[target_col]

        # Note: Standard CV might shuffle data, which is bad for time series.
        # Consider using TimeSeriesSplit from sklearn.model_selection
        # from sklearn.model_selection import TimeSeriesSplit
        # tscv = TimeSeriesSplit(n_splits=cv)

        lgbm = lgb.LGBMRegressor(random_state=42, objective='regression_l1') # L1 (MAE) often robust

        grid_search = GridSearchCV(estimator=lgbm,
                                   param_grid=param_grid,
                                   scoring=scoring,
                                   cv=cv, # Or tscv for time series split
                                   n_jobs=n_jobs, # Use all available cores
                                   verbose=3)

        start_time = time.time()
        grid_search.fit(X_train, y_train,
                        categorical_feature=self.categorical_features
                        # eval_set=[(X_val, y_val)], # Can't use eval_set directly with GridSearchCV's fit
                        # callbacks=[lgb.early_stopping(10, verbose=True)] # Also tricky with GridSearchCV CV
                       )
        end_time = time.time()
        logging.info(f"GridSearchCV completed in {end_time - start_time:.2f} seconds.")

        self.best_params = grid_search.best_params_
        logging.info(f"Best parameters found: {self.best_params}")
        logging.info(f"Best score (lower is better for competition loss): {grid_search.best_score_}")

        # Update model_params with best found params for future training
        self.model_params.update(self.best_params)

        return self.best_params

    def train(self,
              data,
              forecast_start_year=2023,
              target_col='Quantity',
              use_best_params=True,
              save_model=True):
        """
        Trains the final LightGBM model on data before the forecast period.

        Args:
            data (pd.DataFrame): The fully preprocessed DataFrame.
            forecast_start_year (int): The first year to forecast (data before this is used for training).
            target_col (str): Name of the target column.
            use_best_params (bool): If True and best_params exist (from tuning), use them. Otherwise uses self.model_params.
            save_model (bool): If True, save the trained model.
        """
        logging.info("--- Starting Final Model Training ---")
        final_train_df = Preprocess.get_data_for_final_train(data, forecast_start_year)

        X_train = final_train_df[self.feature_columns]
        y_train = final_train_df[target_col]

        logging.info(f"Final training data shape: {X_train.shape}")

        current_params = self.model_params.copy()
        if use_best_params and self.best_params:
            logging.info("Using best parameters found during tuning.")
            current_params.update(self.best_params) # Ensure tuned params override defaults
        else:
             logging.info("Using parameters provided during initialization or defaults.")

        self.model = lgb.LGBMRegressor(**current_params, random_state=42)#, objective='regression_l1')

        start_time = time.time()
        self.model.fit(X_train, y_train,
                       categorical_feature=self.categorical_features,
                       **self.train_params) # Pass train_params like early stopping if defined
        end_time = time.time()
        logging.info(f"Final model training completed in {end_time - start_time:.2f} seconds.")

        if save_model:
            self.save_model()

    def _prepare_forecast_features(self, forecast_year=2023):
        """
        Prepares the feature DataFrame for the forecast period.
        Relies on the self.preprocessor instance having the full historical data.
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not initialized. Run _prepare_data first.")

        logging.info(f"--- Preparing features for forecast year: {forecast_year} ---")
        
        # Get processed data BUT EXCLUDE the forecast year if it exists
        all_processed_data = self.preprocessor.get_data()
        historical_df = all_processed_data[all_processed_data['Year'] < forecast_year].copy()

        logging.info(f"Shape of all_processed_data: {all_processed_data.shape}")
        logging.info(f"Shape of historical_df (years < {forecast_year}): {historical_df.shape}")

        # Identify unique entities from historical data only
        unique_countries = historical_df['Country'].unique()
        unique_products = historical_df['Product'].unique()

        # Create future date range
        future_index = pd.MultiIndex.from_product(
            [unique_countries, unique_products, [forecast_year], range(1, 13)],
            names=['Country', 'Product', 'Year', 'Month_Num']
        )
        future_df = pd.DataFrame(index=future_index).reset_index()
        future_df['Quantity'] = np.nan

        logging.info(f"Shape of future_df skeleton for {forecast_year}: {future_df.shape}")
        expected_forecast_rows = len(future_df)

        # Combine historical with future skeleton
        combined_df = pd.concat([historical_df, future_df], ignore_index=True)
        combined_df.sort_values(by=['Country', 'Product', 'Year', 'Month_Num'], inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
        logging.info(f"Shape of combined_df after concat: {combined_df.shape}")

        logging.info("Calculating month_sin/month_cos for combined data...")
        combined_df['month_sin'] = np.sin(2 * np.pi * combined_df['Month_Num'] / 12)
        combined_df['month_cos'] = np.cos(2 * np.pi * combined_df['Month_Num'] / 12)

        preprocess_args = getattr(self.preprocessor, '_last_preprocess_args', {})
        rolling_window = preprocess_args.get('rolling_window', 6)
        lag_periods = preprocess_args.get('lag_periods', [1, 2, 3, 12])

        group_cols = ['Country', 'Product']
        logging.info("Recalculating lags for forecast period...")
        for lag in lag_periods:
            combined_df[f'lag_{lag}'] = combined_df.groupby(group_cols, observed=False)['Quantity'].shift(lag)

        logging.info("Recalculating rolling features for forecast period...")
        combined_df[f'rolling_mean_{rolling_window}'] = combined_df.groupby(group_cols, observed=False)['Quantity'].transform(
            lambda x: x.shift(1).rolling(window=rolling_window, min_periods=1).mean()
        )
        combined_df[f'rolling_std_{rolling_window}'] = combined_df.groupby(group_cols, observed=False)['Quantity'].transform(
            lambda x: x.shift(1).rolling(window=rolling_window, min_periods=1).std()
        )

        feature_cols_to_fill = [f'lag_{lag}' for lag in lag_periods] + [
            f'rolling_mean_{rolling_window}',
            f'rolling_std_{rolling_window}'
        ]
        logging.info("Filling NaNs in recalculated features...")
        combined_df.fillna({col: 0 for col in feature_cols_to_fill}, inplace=True)

        logging.info("Applying categorical encoding for forecast period...")
        for col in self.preprocessor.categorical_features:
            if col in combined_df.columns:
                # GETTING CATEGORIES FROM ORIGINAL DATA - GOOD
                train_categories = self.preprocessor.df[col].cat.categories
                # APPLYING CATEGORICAL DTYPE - SHOULD WORK
                combined_df[col] = pd.Categorical(combined_df[col], categories=train_categories)
                if combined_df[col].isnull().any():
                     logging.warning(f"NaNs introduced in categorical column '{col}' during forecast preparation.") # Could this cause issues? Maybe, but unlikely the dtype error.
            else:
                logging.warning(f"Categorical column '{col}' expected but not found in combined forecast data.")

        # Filtering data for the forecast year
        forecast_features_df = combined_df[combined_df['Year'] == forecast_year].copy()
        logging.info(f"Shape of forecast_features_df AFTER filtering for year {forecast_year}: {forecast_features_df.shape}")

        # Handling missing columns - Potential Issue Here?
        missing_cols = set(self.feature_columns) - set(forecast_features_df.columns)
        if missing_cols:
            logging.warning(f"Missing columns in forecast features: {missing_cols}. Adding with 0.")
            for c in missing_cols:
                forecast_features_df[c] = 0 # If 'Country' or 'Product' were missing and added as 0, they lose categorical type!

        # Reordering/Selecting columns - SHOULD preserve dtype
        forecast_features_df = forecast_features_df[self.feature_columns]

        numerical_features_in_use = forecast_features_df[self.feature_columns].select_dtypes(include=np.number).columns.tolist()
        nan_check_df = forecast_features_df[numerical_features_in_use].isnull()
        if nan_check_df.any().any():
            cols_with_nan = nan_check_df.columns[nan_check_df.any()].tolist()
            logging.warning(f"NaN values in {cols_with_nan}. Filling with 0.")
            for col in numerical_features_in_use:
                if forecast_features_df[col].isnull().any():
                    forecast_features_df[col].fillna(0, inplace=True)

        categorical_cols_in_features = [
            col for col in self.categorical_features
            if col in self.feature_columns and col in forecast_features_df.columns
        ]
        if forecast_features_df[categorical_cols_in_features].isnull().any().any():
            logging.error("FATAL: NaNs detected in categorical features.")
            raise ValueError("NaNs found in categorical input features.")

        logging.info(f"Forecast features prepared with shape: {forecast_features_df.shape}")

        return forecast_features_df

    def predict(self, forecast_year=2023):
        """
        Generates predictions for the specified forecast year.

        Args:
            forecast_year (int): The year to generate predictions for.

        Returns:
            pd.DataFrame: DataFrame with predictions in the required format.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        if self.feature_columns is None:
             raise RuntimeError("Feature columns not set. Ensure data preparation and training ran.")

        logging.info(f"--- Generating Predictions for {forecast_year} ---")

        # Prepare features for the forecast period
        X_forecast = self._prepare_forecast_features(forecast_year)

        # Make predictions
        predictions_raw = self.model.predict(X_forecast)

        # Post-process predictions (non-negative integers)
        predictions_final = np.round(np.maximum(0, predictions_raw)).astype(int)
        logging.info("Predictions generated and post-processed.")

        # --- Format output ---
        # Need Country, Product, Month (MmmYYYY), Quantity
        # Get identifiers from the X_forecast index or related df
        # The _prepare_forecast_features should ideally return the df with identifiers

        # Retrieve identifiers from the original future_df structure used in _prepare_forecast_features
        historical_df = self.preprocessor.get_data()
        unique_countries = historical_df['Country'].unique()
        unique_products = historical_df['Product'].unique()
        future_index = pd.MultiIndex.from_product(
            [unique_countries, unique_products, [forecast_year], range(1, 13)],
            names=['Country', 'Product', 'Year', 'Month_Num']
        )
        # Create the base DataFrame again to easily map predictions
        output_df = pd.DataFrame(index=future_index).reset_index()

        # Check length consistency
        if len(output_df) != len(predictions_final):
             logging.error(f"Mismatch between number of expected forecast points ({len(output_df)}) and predictions made ({len(predictions_final)}). Check feature preparation.")
             # Attempt to align based on available features if possible, or raise error
             # For now, let's assume X_forecast rows correspond directly to output_df rows
             # This relies on _prepare_forecast_features preserving the order.
             min_len = min(len(output_df), len(predictions_final))
             output_df = output_df.iloc[:min_len]
             predictions_final = predictions_final[:min_len]
             raise ValueError("Prediction length mismatch.")


        output_df['Quantity'] = predictions_final

        # Format Month to MmmYYYY
        month_num_to_abbrev = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        output_df['Month'] = output_df['Month_Num'].map(month_num_to_abbrev) + output_df['Year'].astype(str)

        # Select and order final columns
        output_df = output_df[['Country', 'Product', 'Month', 'Quantity']]

        return output_df

    def save_predictions(self, predictions_df, path=None):
        """Saves the predictions DataFrame to a CSV file."""
        if path is None:
             # Use default path with team code
             path = OUTPUT_DIR / f"01_output_prediction_{TEAM_CODE}.csv"
        else:
             path = Path(path) # Ensure it's a Path object

        logging.info(f"Saving predictions to {path}")
        try:
            predictions_df.to_csv(path, index=False)
            logging.info("Predictions saved successfully.")
        except Exception as e:
            logging.error(f"Error saving predictions: {e}")

    def save_model(self, path=MODEL_SAVE_PATH):
        """Saves the trained LightGBM model."""
        if self.model is None:
            logging.warning("No model trained yet, nothing to save.")
            return

        logging.info(f"Saving trained model to {path}")
        try:
            joblib.dump(self.model, path)
            logging.info("Model saved successfully.")
        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def load_model(self, path=MODEL_SAVE_PATH):
        """Loads a pre-trained LightGBM model."""
        path = Path(path)
        if not path.exists():
            logging.error(f"Model file not found at {path}")
            raise FileNotFoundError(f"Model file not found at {path}")

        logging.info(f"Loading model from {path}")
        try:
            self.model = joblib.load(path)
            logging.info("Model loaded successfully.")
            # Note: Feature columns and categorical features used for the loaded model
            # are not automatically loaded. Ensure consistency or save/load them separately.
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    DO_TUNING = False # Set to True to run GridSearchCV
    FORECAST_YEAR = 2023
    VALIDATION_YEAR = 2022 # Year used for validation split during tuning/evaluation

    # Preprocessing arguments
    preprocess_config = {
        "rolling_window": 6,
        "lag_periods": [1, 2, 3, 6, 12] # Example lag features
    }

    # LightGBM Model Parameters (Example - Start simple or use tuned params)
    lgbm_model_params = {
        'objective': 'regression_l1', # MAE is often robust to outliers
        'metric': 'mae',
        'n_estimators': 1000, # Can be high if using early stopping
        'learning_rate': 0.05,
        'feature_fraction': 0.8, # Colsample_bytree
        'bagging_fraction': 0.8, # Subsample
        'bagging_freq': 1,
        'lambda_l1': 0.1, # reg_alpha
        'lambda_l2': 0.1, # reg_lambda
        'num_leaves': 31, # Default
        'verbose': -1, # Suppress verbose output
        'n_jobs': 15, # Use all cores
        'seed': 42,
        'boosting_type': 'gbdt',
        'force_row_wise': True,  # To skip testing overhead, force row-wise
        #'force_col_wise': True,  # To force column-wise (usually for memory saving)
    }

    # LightGBM Training Parameters (e.g., for early stopping)
    lgbm_train_params = {
        # Example: Using validation set for early stopping during FINAL training
        # Requires splitting final_train_df further, or passing eval_set manually
        # 'callbacks': [lgb.early_stopping(stopping_rounds=50, verbose=True)],
        # 'eval_metric': 'mae' # Metric for early stopping
    }

    # --- Initialize Forecaster ---
    forecaster = LGBMForecaster(model_params=lgbm_model_params,
                                train_params=lgbm_train_params)

    # --- Load and Preprocess Data ---
    # Store preprocessing args in the preprocessor instance for later use
    forecaster.preprocessor = Preprocess(pd.read_csv(DATA_DIR / INPUT_HISTORY_CSV))
    forecaster.preprocessor._last_preprocess_args = preprocess_config # Store args
    processed_data = forecaster.preprocessor.full_preprocess(**preprocess_config)
    forecaster.preprocessor.save_processed_data() 
    forecaster.feature_columns = forecaster.preprocessor.get_feature_columns()
    forecaster.categorical_features = forecaster.preprocessor.categorical_features # Get actual cat features
    logging.info(f"Data prepared. Shape: {processed_data.shape}")

    # --- Optional: Hyperparameter Tuning ---
    if DO_TUNING:
        # Define parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [200, 500, 1000],
            'learning_rate': [0.02, 0.05, 0.1],
            'num_leaves': [20, 31, 40],
            # 'max_depth': [-1, 10, 20], # Often num_leaves is enough
            'lambda_l1': [0, 0.1, 0.5],
            'lambda_l2': [0, 0.1, 0.5],
            'force_row_wise': [True]
        }
        best_params = forecaster.tune_hyperparameters(data=processed_data,
                                                     param_grid=param_grid,
                                                     cv=3, # Use TimeSeriesSplit ideally
                                                     val_year=VALIDATION_YEAR)
        # The forecaster's model_params are updated internally

    # --- Train Final Model ---
    forecaster.train(data=processed_data,
                     forecast_start_year=FORECAST_YEAR,
                     save_model=True) # Save the trained model

    # --- Generate Predictions ---
    # You could also load a previously saved model here if needed:
    # forecaster.load_model()
    # Ensure feature_columns and categorical_features are set correctly if loading
    predictions_df = forecaster.predict(forecast_year=FORECAST_YEAR)

    # --- Save Predictions ---
    forecaster.save_predictions(predictions_df) # Saves to the default path with TEAM_CODE

    from evaluate import evaluate

    print(evaluate())

    logging.info("--- Script Finished ---")