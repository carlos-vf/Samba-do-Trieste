from pathlib import Path

# Absolute path (relative to the parent directory of the script's location)
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR.parent / "data/01_input_history.csv"
PREDICTIONS_SAVE_PATH = SCRIPT_DIR.parent / "output/01_output_prediction_8475.csv"
from preprocessing import Preprocess
import pandas as pd
import numpy as np

def my_score(reference, proposal):
    reference = np.array(reference)
    proposal = np.array(proposal)
    new_reference = np.where(reference == 0, 1, reference)
    return np.mean((reference - proposal) ** 2 / new_reference)

def evaluate():
    """
    Evaluate the predictions against the original data.
    """
    original_data = pd.read_csv(DATA_PATH)
    preprocess_data = Preprocess(original_data)
    preprocess_data.parse_dates()
    preprocessed_df = preprocess_data.get_data()
    original_data_2023 = preprocessed_df[preprocessed_df['Year'] == 2023]

    predictions = pd.read_csv(PREDICTIONS_SAVE_PATH)
    preprocess_pred = Preprocess(predictions)
    preprocess_pred.parse_dates()
    predictions = preprocess_pred.get_data()

    # Merge the two DataFrames on the index
    merged_df = pd.merge(
        original_data_2023,
        predictions,
        on=['Country', 'Product', 'Year', 'Month'],
        how='inner'
    )

    # Extract the true and predicted values
    y_true = merged_df['Quantity_x']
    y_pred = merged_df['Quantity_y']

    loss = my_score(y_true, y_pred)
    
    return loss

if __name__ == "__main__":
    loss = evaluate()
    print(f"Evaluation Loss: {loss}")