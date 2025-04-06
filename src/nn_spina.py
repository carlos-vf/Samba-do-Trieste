import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import logging
import sys
from pathlib import Path # Modern path handling

# Assuming preprocessing.py is in the same directory or accessible via PYTHONPATH
try:
    from preprocessing import Preprocess, INPUT_HISTORY_CSV, PROCESSED_DATA_CSV
except ImportError:
    print("Error: Could not import Preprocess class. Make sure preprocessing.py is accessible.")
    sys.exit(1)

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DATA_PATH = DATA_DIR / INPUT_HISTORY_CSV # Use path from preprocessing
PROCESSED_DATA_PATH = DATA_DIR / PROCESSED_DATA_CSV # Use path from preprocessing
OUTPUT_DIR = SCRIPT_DIR.parent / "output" # Define an output directory
MODEL_SAVE_PATH = OUTPUT_DIR / "best_demand_forecast_model.pth"
PREDICTIONS_SAVE_PATH = OUTPUT_DIR / "01_output_prediction_8475.csv" # Follow naming convention

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Hyperparameters & Settings ---
CONFIG = {
    "val_year": 2021,
    "test_year": 2022,
    "target_col": 'Quantity',
    "rolling_window": 6,
    "batch_size": 128,
    "learning_rate": 0.005,
    "num_epochs": 100,
    "patience": 10,
    "seed": 42,
    "num_workers": os.cpu_count() or 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# --- Model Definition ---
class DemandForecastNet(nn.Module):
    def __init__(self, input_dim):
        super(DemandForecastNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)

# --- Custom Loss Function ---
def competition_loss(reference, proposal):
    """
    Calculates the competition's specific loss metric.
    Handles the case where reference is 0 as specified in the PDF.
    Loss = mean(((Reference - Proposal)^2) / Denominator)
    where Denominator = Reference if Reference != 0, else 1 + Reference (which is 1)
    """
    # Ensure tensors are on the same device and float type
    reference = reference.float()
    proposal = proposal.float()

    # Create denominator: reference where reference > 0, 1 otherwise
    denominator = torch.where(reference != 0, reference, torch.ones_like(reference))

    loss_val = torch.mean(((proposal - reference) ** 2) / denominator)
    return loss_val


# --- Data Handling ---
def load_and_prepare_data(config):
    """Loads, preprocesses, splits, scales data and creates DataLoaders."""
    logging.info("--- Starting Data Loading and Preparation ---")

    # Check if processed data exists, otherwise run preprocessing
    if not PROCESSED_DATA_PATH.exists():
        logging.warning(f"Processed data {PROCESSED_DATA_PATH} not found. Running preprocessing...")
        try:
            raw_data_path = DATA_DIR / INPUT_HISTORY_CSV # Assumes INPUT_HISTORY_CSV is just the filename
            if not raw_data_path.exists():
                 raise FileNotFoundError(f"Raw input data {raw_data_path} not found.")

            logging.info(f"Reading raw data from: {raw_data_path}")
            raw_data = pd.read_csv(raw_data_path)
            preprocessor = Preprocess(raw_data)
            preprocessor.parse_dates()
            # Create features
            preprocessor.rolling_stats(window=config["rolling_window"])
            preprocessor.handle_dummies()
            preprocessor.drop_na_features()
            # Save the fully processed data
            preprocessor.save_on_csv(str(PROCESSED_DATA_PATH)) # Use absolute path
            data = preprocessor.get_data()
            logging.info(f"Preprocessing complete. Saved to {PROCESSED_DATA_PATH}")
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}", exc_info=True)
            sys.exit(1)
    else:
        logging.info(f"Loading processed data from {PROCESSED_DATA_PATH}")
        data = pd.read_csv(PROCESSED_DATA_PATH)

    full_processed_data = data.copy()

    # Use the Preprocess class just for splitting
    X_train, y_train, X_val, y_val, X_test, y_test = Preprocess.split_test_train(
        data = data,
        val_year=config["val_year"],
        test_year=config["test_year"],
        target_col=config["target_col"]
    )
    
    feature_columns = X_train.columns.tolist()

    # Convert to numpy arrays (required by scaler)
    X_train_np = X_train.values.astype(np.float32)
    X_val_np = X_val.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)
    y_train_np = y_train.values.astype(np.float32)
    y_val_np = y_val.values.astype(np.float32)
    y_test_np = y_test.values.astype(np.float32)

    # Standardize features (Fit on train, transform train/val/test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_val_scaled = scaler.transform(X_val_np)
    X_test_scaled = scaler.transform(X_test_np)
    logging.info("Features scaled.")

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32).view(-1, 1)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=True if config["device"] == "cuda" else False)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"] * 2, shuffle=False, num_workers=config["num_workers"], pin_memory=True if config["device"] == "cuda" else False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"] * 2, shuffle=False, num_workers=config["num_workers"], pin_memory=True if config["device"] == "cuda" else False)
    logging.info("DataLoaders created.")

    input_dim = X_train_tensor.shape[1]

    # Return necessary items including the scaler, feature columns, and full data
    return train_loader, val_loader, test_loader, scaler, input_dim, y_test, feature_columns, full_processed_data

# --- Trainer Class ---
class Trainer:
    def __init__(self, model, criterion, optimizer, device, scaler, config, model_save_path):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.model_save_path = Path(model_save_path) # Ensure it's a Path object
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.scaler = scaler

    def _train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(batch_y, outputs)
            loss.backward()
            # Gradient clipping
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def _validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(batch_y, outputs)
                running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(val_loader.dataset)
        return epoch_loss

    def train(self, train_loader, val_loader):
        logging.info("--- Starting Training ---")
        start_time = time.time()

        for epoch in range(self.config["num_epochs"]):
            epoch_start_time = time.time()

            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            epoch_duration = time.time() - epoch_start_time
            logging.info(f"Epoch {epoch + 1}/{self.config['num_epochs']} | "
                         f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                         f"Duration: {epoch_duration:.2f}s")

            # --- Early Stopping & Model Checkpointing ---
            if val_loss < self.best_val_loss:
                logging.info(f"Validation loss improved ({self.best_val_loss:.4f} -> {val_loss:.4f}). Saving model...")
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch + 1, val_loss)
            else:
                self.epochs_no_improve += 1
                logging.info(f"Validation loss did not improve. Patience: {self.epochs_no_improve}/{self.config['patience']}")

            if self.epochs_no_improve >= self.config['patience']:
                logging.info("Early stopping triggered.")
                break

        total_training_time = time.time() - start_time
        logging.info(f"--- Training Finished ---")
        logging.info(f"Total Training Time: {total_training_time:.2f}s")
        logging.info(f"Best Validation Loss: {self.best_val_loss:.4f}")

    def _save_checkpoint(self, epoch, val_loss):
        """Saves model checkpoint."""
        try:
            # Ensure the directory exists
            self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': val_loss,
            }, self.model_save_path)
            logging.info(f"Checkpoint saved to {self.model_save_path}")
        except Exception as e:
             logging.error(f"Error saving checkpoint: {e}", exc_info=True)


    def load_best_model(self):
        """Loads the best model checkpoint."""
        if self.model_save_path.exists():
            try:
                logging.info(f"Loading best model from {self.model_save_path}")
                checkpoint = torch.load(self.model_save_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                # Optionally load optimizer state if continuing training
                # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.best_val_loss = checkpoint.get('best_val_loss', float('inf')) # Load saved best loss
                logging.info(f"Best model loaded successfully (Epoch {checkpoint.get('epoch', 'N/A')}, Val Loss: {self.best_val_loss:.4f}).")
            except Exception as e:
                 logging.error(f"Error loading checkpoint: {e}", exc_info=True)
                 sys.exit(1)
        else:
            logging.warning(f"Model checkpoint {self.model_save_path} not found. Using initialized model.")

    def predict(self, test_loader):
        """Generates predictions on the test set using the currently loaded model."""
        self.model.eval()
        all_predictions = []
        logging.info("--- Generating Predictions on Test Set ---")
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                all_predictions.append(outputs.cpu().numpy())

        predictions_np = np.concatenate(all_predictions, axis=0).squeeze()

        # --- Post-processing ---
        # Ensure non-negative integers
        final_predictions = np.round(np.maximum(0, predictions_np)).astype(int)
        logging.info("Predictions generated and post-processed.")

        return final_predictions

# --- Utility Functions ---
def set_seed(seed):
    """Sets random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")

# --- Main Execution ---
def main():
    """Main function to run the forecasting process."""
    set_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])
    logging.info(f"Using device: {device}")

    #? -------------------------------------------------
    #?           1. Load and Prepare Data
    #? -------------------------------------------------
    try:
        train_loader, val_loader, test_loader, scaler, input_dim, y_test, _, _ = load_and_prepare_data(CONFIG)
    except Exception as e:
        logging.error(f"Failed to load or prepare data: {e}", exc_info=True)
        sys.exit(1)

    #? -------------------------------------------------
    #?     2. Initialize Model, Loss, Optimizer
    #? -------------------------------------------------
    model = DemandForecastNet(input_dim).to(device)
    criterion = competition_loss
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    # Optional: Learning rate scheduler
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    #? -------------------------------------------------
    #?           3. Initialize and Run Trainer    
    #? -------------------------------------------------

    trainer = Trainer(model, criterion, optimizer, device, scaler, CONFIG, MODEL_SAVE_PATH)
    #trainer.train(train_loader, val_loader)

    #? -------------------------------------------------
    #?         4. Load Best Model and Predict
    #? -------------------------------------------------
    trainer.load_best_model() # Load the model with the best validation score
    final_predictions = trainer.predict(test_loader)
    
    #? -------------------------------------------------
    #?                  5. Evaluate
    #? -------------------------------------------------
    
    # Convert y_test to tensor for evaluation
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)
    # Convert predictions back to tensor for evaluation
    predictions_tensor = torch.tensor(final_predictions, dtype=torch.float32).view(-1, 1).to(device)

    with torch.no_grad():
         test_loss = criterion(y_test_tensor, predictions_tensor).item()
    logging.info(f"--- Final Evaluation on Test Set ---")
    logging.info(f"Competition Loss: {test_loss:.4f}")
    # Add other metrics if needed (e.g., MSE, MAE)
    mse = mean_squared_error(y_test.values, final_predictions)
    logging.info(f"Mean Squared Error (MSE): {mse:.4f}")

    #? -------------------------------------------------
    #?     6. Now let's predict 2024 (unseen data) 
    #? -------------------------------------------------
    forecast_year = 2023
    
    logging.info(f"--- Starting {forecast_year} Prediction ---")

    # Ensure we have the required components from data loading
    try:
        # Rerun data loading to get all necessary components cleanly
        _, _, _, scaler, _, _, feature_columns, full_processed_data = load_and_prepare_data(CONFIG)
        logging.info(f"Loaded scaler and {len(feature_columns)} feature columns for prediction.")
        # Need original C/P columns for grouping and feature generation
        # Let's reload the raw data and minimally process it again just for C/P list and structure
        raw_data_path = DATA_DIR / INPUT_HISTORY_CSV
        raw_data = pd.read_csv(raw_data_path)
        preprocessor = Preprocess(raw_data)
        preprocessor.parse_dates() # Just get Year, Month_Num, Country, Product
        history_df_minimal = preprocessor.get_data()[['Country', 'Product', 'Year', 'Month', 'Quantity']]

    except NameError:
        logging.error("Scaler, feature_columns, or full_processed_data not defined. Rerun data loading.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error preparing for {forecast_year} prediction: {e}", exc_info=True)
        sys.exit(1)

    # --- Prepare data structure for 2024 ---
    unique_countries = history_df_minimal['Country'].unique()
    unique_products = history_df_minimal['Product'].unique()

    # Create all combinations for 2024
    future_index = pd.MultiIndex.from_product(
        [unique_countries, unique_products, [forecast_year], range(1, 13)],
        names=['Country', 'Product', 'Year', 'Month']
    )
    future_df = pd.DataFrame(index=future_index).reset_index()
    future_df['Quantity'] = np.nan # Placeholder for predicted quantities

    # Combine historical data (needed for lags/rolling) with the future skeleton
    # Use enough history to calculate the features for Jan
    # Using the *minimal* history here avoids carrying over dummy cols etc.
    relevant_history = history_df_minimal[history_df_minimal['Year'] >= forecast_year - 2]
    relevant_history = relevant_history[relevant_history['Year'] < forecast_year] # Exclude future data (if any)
    
    combined_df = pd.concat([relevant_history, future_df], ignore_index=True)
    combined_df.sort_values(by=['Country', 'Product', 'Year', 'Month'], inplace=True)
    combined_df.reset_index(drop=True, inplace=True) # Ensure clean index

    # --- Iteratively predict month by month ---
    model = trainer.model # Get the loaded best model
    model.eval() # Set model to evaluation mode

    logging.info(f"Predicting month by month for {forecast_year}...")
    for month in range(1, 13):
        logging.debug(f"Predicting Month: {month}")

        # --- Feature Engineering for the current month ---
        # Calculate features based on data up to the *previous* month
        # Use the combined_df which gets updated with predictions

        group_cols = ['Country', 'Product']
        # Lags (use shift on the *current* combined_df)
        combined_df['lag_1'] = combined_df.groupby(group_cols)['Quantity'].shift(1)
        combined_df['lag_2'] = combined_df.groupby(group_cols)['Quantity'].shift(2)
        combined_df['lag_3'] = combined_df.groupby(group_cols)['Quantity'].shift(3)

        # Rolling stats (use rolling on the *current* combined_df)
        rolling_window_size = CONFIG['rolling_window']
        combined_df['rolling_mean_6'] = combined_df.groupby(group_cols)['Quantity'].transform(
             lambda x: x.rolling(window=rolling_window_size, min_periods=1).mean().shift(1) # Shift(1) to use data *before* current month
        )
        combined_df['rolling_std_6'] = combined_df.groupby(group_cols)['Quantity'].transform(
             lambda x: x.rolling(window=rolling_window_size, min_periods=1).std().shift(1) # Shift(1)
        )
        
        feature_cols_to_fill = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_6', 'rolling_std_6']
        for col in feature_cols_to_fill:
            # Forward fill within each group
            combined_df[col] = combined_df.groupby(group_cols, group_keys=False)[col].ffill()
            # combined_df[col] = combined_df[col].fillna(0)

        logging.debug("Finished filling NaNs.")

        # --- Prepare features for the *current* month's prediction ---
        current_month_mask = (combined_df['Year'] == forecast_year) & (combined_df['Month'] == month)
        predict_df_month = combined_df[current_month_mask].copy()

        if predict_df_month.empty:
             logging.warning(f"No data found for {forecast_year}-{month}. Skipping.")
             continue

        # Apply one-hot encoding consistent with training data
        predict_df_month = pd.get_dummies(predict_df_month, columns=['Country', 'Product'], drop_first=False)

        # Align columns with the training feature columns
        # Add missing columns (if any) and fill with 0, then reorder
        missing_cols = set(feature_columns) - set(predict_df_month.columns)
        for c in missing_cols:
            predict_df_month[c] = 0
        predict_df_month = predict_df_month[feature_columns] # Ensure same order and columns

        # Check for NaNs in features before scaling (should be handled by fillna above)
        if predict_df_month.isnull().any().any():
             logging.warning(f"NaNs detected in features for {forecast_year}-{month} before scaling. Filling with 0.")
             # print(predict_df_month[predict_df_month.isnull().any(axis=1)]) # Debug which rows/cols
             predict_df_month.fillna(0, inplace=True)

        # Scale features using the *fitted* scaler
        X_pred_scaled = scaler.transform(predict_df_month.values.astype(np.float32))

        # Convert to tensor
        X_pred_tensor = torch.tensor(X_pred_scaled, dtype=torch.float32).to(device)

        # --- Predict ---
        with torch.no_grad():
            outputs = model(X_pred_tensor)
            predictions_np = outputs.cpu().numpy().squeeze()

        # Post-process predictions
        final_predictions = np.round(np.maximum(0, predictions_np)).astype(int)

        # --- Store predictions back into the combined DataFrame ---
        # Use the index from predict_df_month to update the correct rows in combined_df
        combined_df.loc[current_month_mask, 'Quantity'] = final_predictions
        logging.debug(f"Finished prediction for Month: {month}. Example prediction: {final_predictions[:5]}")
        
    combined_df.to_csv(OUTPUT_DIR / "combined_df.csv", index=False)

    # --- Format and Save 2024 Predictions ---
    logging.info(f"Formatting and saving {forecast_year} predictions...")
    forecast_predictions_df = combined_df[combined_df['Year'] == forecast_year].copy()

    # Select required columns
    forecast_predictions_df = forecast_predictions_df[['Country', 'Product', 'Month', 'Quantity']]

    # Convert Month number back to 'MmmYYYY' format
    month_num_to_abbrev = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    forecast_predictions_df['Month'] = forecast_predictions_df['Month'].map(month_num_to_abbrev) + str(forecast_year)

    # Ensure integer quantity
    forecast_predictions_df['Quantity'] = forecast_predictions_df['Quantity'].astype(int)

    # Construct output path
    forecast_predictions_df.to_csv(PREDICTIONS_SAVE_PATH, index=False)
    logging.info(f"{forecast_year} predictions saved to {PREDICTIONS_SAVE_PATH}")
    
    from evaluate import evaluate
    # Call the evaluate function from evaluate.py
    loss = evaluate()
    
    print(f"Competition Loss: {loss}")

    logging.info("--- Script Finished ---")

if __name__ == "__main__":
    main()
