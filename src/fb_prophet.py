import pandas as pd
import os
import sys
from pathlib import Path
import numpy as np
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from tqdm import tqdm
from itertools import product
from collections import Counter

# Importa MSE da Scikit-learn
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.exponential_smoothing.ets import ETSModel
# Ignora warning comuni da statsmodels (opzionale ma pulisce l'output)
import warnings
warnings.filterwarnings("ignore")

try:
    # Assumiamo che evaluate() calcoli il loss sul file CSV finale
    from evaluate import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    print("Attenzione: Funzione 'evaluate' non trovata. Evaluation finale non eseguita.")
    EVALUATE_AVAILABLE = False

# --- Configure Logging ---
# Disabilita completamente i logger più ostinati
logging.getLogger('prophet').disabled = True # Anche se non usato, per sicurezza
logging.getLogger('cmdstanpy').disabled = True
logging.getLogger('cmdstanpy.model').disabled = True
logging.getLogger('cmdstanpy.stanfit').disabled = True

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_DIR = SCRIPT_DIR.parent / "output"
INPUT_HISTORY_CSV = '01_input_history.csv'
OUTPUT_PREDICTION_CSV = '01_output_prediction_8475.csv' # Mantenuto il tuo Team Code
NUM_PROCESSES = max(1, cpu_count() - 1)
MIN_OBS_ETS = 25 # Manteniamo soglia minima per ETS
VALIDATION_YEAR = 2022 # Anno da usare per la validazione

# --- Definizione Griglia Parametri ETS ---
# (Riduci/Espandi secondo necessità e tempo disponibile)
param_grid = {
    'error': ['add', 'mul'],
    'trend': ['add', None],         # 'mul' trend è meno comune, proviamo add/None
    'seasonal': ['add', 'mul'],     # Proviamo entrambi per stagionalità
    'damped_trend': [True, False]
}
# Genera tutte le combinazioni possibili
grid_param_list = list(product(*param_grid.values()))
# Converti in lista di dizionari per chiarezza
param_combinations = [dict(zip(param_grid.keys(), p)) for p in grid_param_list]
print(f"Grid Search: Testing {len(param_combinations)} parameter combinations per series using MSE for validation.")


# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Funzione Loss Personalizzata RIMOSSA ---
# (useremo MSE internamente, evaluate() usa quella custom esternamente)


# --- Helper Function for Date Parsing ---
def parse_month_year(mmyyyy):
    # ... (come prima) ...
    month_map = { "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12 }
    try:
        month_abbrev = mmyyyy[:3]; year_str = mmyyyy[3:]; year = int(year_str); month = month_map[month_abbrev]
        return pd.Timestamp(f"{year}-{month}-01") + pd.offsets.MonthEnd(0)
    except: return pd.NaT

# --- Worker Function for Grid Search per Series (using MSE) ---
def process_combination_gridsearch_mse(combo_args, train_df, valid_df, param_combos):
    country, product = combo_args
    # Usiamo .loc per evitare SettingWithCopyWarning, anche se qui non modifichiamo
    # Usiamo dropna() qui per sicurezza, anche se controllato nel main
    series_train = train_df.loc[(train_df['Country'] == country) & (train_df['Product'] == product), 'y'].dropna().sort_index()
    series_valid_actual = valid_df.loc[(valid_df['Country'] == country) & (valid_df['Product'] == product), 'y'].dropna().sort_index()

    # Controlli iniziali sulla validità dei dati per questa serie
    if series_valid_actual.empty or len(series_train) < MIN_OBS_ETS:
        return country, product, float('inf'), None # Non possiamo validare o fittare
    if np.any(np.isnan(series_valid_actual)) or np.any(np.isinf(series_valid_actual)) or np.any(series_valid_actual < 0):
         # print(f"DEBUG WORKER [{country}-{product}]: Invalid values (NaN/Inf/<0) in VALIDATION data. Skipping.", file=sys.stderr)
         return country, product, float('inf'), None # Non possiamo validare
    if np.any(np.isnan(series_train)) or np.any(np.isinf(series_train)) or np.any(series_train < 0):
         # print(f"DEBUG WORKER [{country}-{product}]: Invalid values (NaN/Inf/<0) in TRAINING data. Skipping.", file=sys.stderr)
         return country, product, float('inf'), None # Non possiamo fittare

    best_mse_for_series = float('inf') # Ora minimizziamo MSE
    best_params_for_series = None
    # Contatori per debug (opzionale)
    # ets_fit_exception_count = 0
    # forecast_nan_count = 0

    for params in param_combos:
        current_params = params.copy()
        ets_input_data = series_train.copy()
        needs_positive = (current_params.get('error') == 'mul' or
                          current_params.get('seasonal') == 'mul')
        if needs_positive:
            ets_input_data = ets_input_data.clip(lower=1e-6) # Applica epsilon a 0
            # Se DOPO l'epsilon ci sono ancora valori <=0 (improbabile ma per sicurezza), salta
            if (ets_input_data <= 0).any():
                 continue

        try:
            # --- Fit del modello ETS ---
            model = ETSModel(
                endog=ets_input_data,
                error=current_params['error'],
                trend=current_params['trend'],
                seasonal=current_params['seasonal'],
                damped_trend=current_params['damped_trend'],
                seasonal_periods=12,
            )
            # Nota: fit può lanciare eccezioni per vari motivi (convergenza, dati, etc.)
            fitted_model = model.fit(disp=False, maxiter=2000) # Aumentato maxiter

            # --- Previsione sul periodo di validazione ---
            forecast_valid = fitted_model.forecast(steps=len(series_valid_actual))

            # --- Validazione Previsioni & Calcolo MSE ---
            # Controlla NaN/Inf nelle previsioni PRIMA di MSE
            if np.any(np.isnan(forecast_valid)) or np.any(np.isinf(forecast_valid)):
                # forecast_nan_count += 1
                continue # Salta se la previsione non è numericamente valida

            # Clampa previsioni a >= 0 prima di MSE (quantità non possono essere negative)
            forecast_valid = forecast_valid.clip(lower=0.0)

            # Calcola MSE
            current_mse = mean_squared_error(series_valid_actual, forecast_valid)

             # Controlla se MSE è valido (non NaN/Inf)
            if np.isnan(current_mse) or np.isinf(current_mse):
                 continue # Salta se MSE non è valido

            # Aggiorna se l'MSE è migliore
            if current_mse < best_mse_for_series:
                best_mse_for_series = current_mse
                best_params_for_series = current_params

        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError, IndexError) as e:
            # Cattura eccezioni comuni da statsmodels/numpy durante fit/forecast
            # ets_fit_exception_count += 1
            # print(f"DEBUG WORKER [{country}-{product}]: Handled Exception {type(e).__name__} with params {current_params}. Skipping.", file=sys.stderr)
            continue # Salta questa combinazione di parametri
        except Exception as e: # Cattura altre eccezioni impreviste
            # ets_fit_exception_count += 1
            print(f"DEBUG WORKER [{country}-{product}]: UNHANDLED Exception {type(e).__name__}: {e} with params {current_params}. Skipping.", file=sys.stderr)
            continue # Salta questa combinazione di parametri


    # Stampa riassunto solo se la serie fallisce completamente
    # if best_mse_for_series == float('inf'):
    #      print(f"DEBUG WORKER [{country}-{product}]: FAILED. No valid params found (MSE). ETS Ex: {ets_fit_exception_count}, Forecast NaN: {forecast_nan_count}.", file=sys.stderr)

    return country, product, best_mse_for_series, best_params_for_series


# --- Worker Function for Final Prediction Run (uguale a prima) ---
def process_combination_final(combo_args, full_train_df, last_hist_date, best_global_params):
    # ... (codice identico a prima, nessuna modifica necessaria qui) ...
    country, product = combo_args
    # Usa .loc e dropna per sicurezza
    series_train = full_train_df.loc[(full_train_df['Country'] == country) & (full_train_df['Product'] == product), 'y'].dropna().sort_index()

    future_dates = pd.date_range(start=last_hist_date + pd.DateOffset(months=1), periods=12, freq='ME')
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': 0.0})
    forecast_df['Country'] = country
    forecast_df['Product'] = product

    last_value_fallback = 0.0
    if not series_train.empty: last_value_fallback = series_train.iloc[-1]

    if len(series_train) < MIN_OBS_ETS:
        forecast_df['yhat'] = last_value_fallback
        forecast_df['yhat'] = forecast_df['yhat'].clip(lower=0.0)
        return forecast_df

    params = best_global_params
    # Se non sono stati trovati parametri globali (improbabile ora), usa un default robusto
    if not params:
        params = {'error': 'add', 'trend': 'add', 'seasonal': 'add', 'damped_trend': True}

    ets_input_data = series_train.copy()
    needs_positive = (params.get('error') == 'mul' or params.get('seasonal') == 'mul')
    if needs_positive:
        ets_input_data = ets_input_data.clip(lower=1e-6)
        if (ets_input_data <= 0).any():
            forecast_df['yhat'] = last_value_fallback # Fallback se dati non positivi
            forecast_df['yhat'] = forecast_df['yhat'].clip(lower=0.0)
            return forecast_df
    try:
        model = ETSModel( endog=ets_input_data, error=params['error'], trend=params['trend'], seasonal=params['seasonal'],
                         damped_trend=params['damped_trend'], seasonal_periods=12)
        fitted_model = model.fit(disp=False, maxiter=2000)
        forecast_values = fitted_model.forecast(12)
        forecast_df['yhat'] = forecast_values.values

    except Exception as e:
        # Se anche la run finale fallisce con i parametri migliori, usa il fallback
        forecast_df['yhat'] = last_value_fallback

    # Assicura non-negatività finale
    forecast_df['yhat'] = forecast_df['yhat'].clip(lower=0.0)
    return forecast_df

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Load Data ---
    history_file_path = DATA_DIR / INPUT_HISTORY_CSV
    if not history_file_path.exists(): sys.exit(f"Error: Input file not found at {history_file_path}")
    print(f"Loading data from {history_file_path}...")
    main_start_time = time.time()
    df_history_full = pd.read_csv(history_file_path)
    print(f"Data loaded successfully in {time.time() - main_start_time:.2f}s.")

    # --- Prepare Data & Split ---
    print("Preparing data and splitting for validation...")
    prep_start_time = time.time()
    df_history_full['ds'] = df_history_full['Month'].apply(parse_month_year)
    df_history_full.dropna(subset=['ds'], inplace=True) # Rimuove righe dove parse fallisce

    # Usa tutti i dati pre-2023
    df_history_pre_2023 = df_history_full[df_history_full['ds'] < '2023-01-01'].copy()
    df_history_pre_2023.rename(columns={'Quantity': 'y'}, inplace=True)
    df_history_pre_2023['y'] = pd.to_numeric(df_history_pre_2023['y'], errors='coerce')
    nan_check = df_history_pre_2023['y'].isnull().sum()
    if nan_check > 0:
        print(f"WARNING: Found {nan_check} NaN in 'Quantity'. Dropping them.", file=sys.stderr)
        df_history_pre_2023.dropna(subset=['y'], inplace=True)
    # Assicura non-negatività (importante prima dello split)
    negative_check = (df_history_pre_2023['y'] < 0).sum()
    if negative_check > 0:
        print(f"WARNING: Found {negative_check} negative values in 'Quantity'. Clipping to 0.", file=sys.stderr)
        df_history_pre_2023['y'] = df_history_pre_2023['y'].clip(lower=0.0)
    df_history_pre_2023['y'] = df_history_pre_2023['y'].astype(float)


    # Split: Training (fino a fine 2021), Validazione (2022)
    train_end_date = pd.Timestamp(f'{VALIDATION_YEAR-1}-12-31')
    validation_start_date = pd.Timestamp(f'{VALIDATION_YEAR}-01-01')
    validation_end_date = pd.Timestamp(f'{VALIDATION_YEAR}-12-31')

    df_train_actual = df_history_pre_2023[df_history_pre_2023['ds'] <= train_end_date].copy()
    df_validation = df_history_pre_2023[(df_history_pre_2023['ds'] >= validation_start_date) &
                                         (df_history_pre_2023['ds'] <= validation_end_date)].copy()

    # Controlli finali su NaN/Inf DOPO lo split
    if df_train_actual['y'].isnull().any() or np.isinf(df_train_actual['y']).any(): sys.exit("ERROR: NaN/Inf in Training Data after split.")
    if df_validation['y'].isnull().any() or np.isinf(df_validation['y']).any(): sys.exit("ERROR: NaN/Inf in Validation Data after split.")

    unique_combinations = df_train_actual[['Country', 'Product']].drop_duplicates().values.tolist()
    total_combinations = len(unique_combinations)
    if not df_train_actual.empty:
        last_train_actual_date = df_train_actual['ds'].max()
    else: sys.exit("Error: No valid training data found before validation year.")

    print(f"Data preparation & split complete in {time.time() - prep_start_time:.2f}s.")
    print(f"Training on data up to {last_train_actual_date.strftime('%Y-%m-%d')}, Validating on {VALIDATION_YEAR}.")
    print(f"Found {total_combinations} unique combinations for grid search.")


    # --- Parallel Grid Search (using MSE) ---
    print(f"\nStarting parallel Grid Search (MSE validation) using {NUM_PROCESSES} processes...")
    gridsearch_start_time = time.time()
    worker_gs_func = partial(process_combination_gridsearch_mse, # Usa worker MSE
                             train_df=df_train_actual, # Passa df, indice non serve più settarlo qui
                             valid_df=df_validation,
                             param_combos=param_combinations)
    series_best_results = []
    failed_series_count = 0
    with Pool(processes=NUM_PROCESSES) as pool:
        results_iterator = pool.imap_unordered(worker_gs_func, unique_combinations)
        for result in tqdm(results_iterator, total=total_combinations, desc="Grid Search (MSE)"):
             series_best_results.append(result)
             # result = (country, product, best_mse, best_params)
             if result[2] == float('inf'): # Controlla se il miglior MSE è infinito
                 failed_series_count += 1
    print(f"Parallel Grid Search complete in {time.time() - gridsearch_start_time:.2f}s.")
    print(f"Grid Search Summary: {failed_series_count}/{total_combinations} series failed to find any valid ETS parameters.")


    # --- Aggiorna Risultati & Trova Parametri Globali Migliori (basato su MSE) ---
    valid_mses = [] # Lista per tenere traccia degli MSE validi
    best_params_list = [] # Lista per tenere traccia dei parametri migliori per ogni serie
    for _, _, mse, params in series_best_results:
        # Considera solo risultati dove è stato trovato un MSE finito e dei parametri
        if mse != float('inf') and params is not None:
            valid_mses.append(mse)
            # Converte il dizionario dei parametri in una tupla di item ordinati per renderlo hashable per Counter
            best_params_list.append(tuple(sorted(params.items())))

    # Se nessuna serie ha prodotto un risultato valido
    if not valid_mses:
        print("\nError: Grid search (MSE) failed to find any valid parameters for any series.")
        # Definisci un set di parametri di default conservativo e robusto
        best_global_params = {'error': 'add', 'trend': 'add', 'seasonal': 'add', 'damped_trend': True}
        print(f"Using conservative default parameters: {best_global_params}")
    else:
        # Calcola l'MSE medio di validazione (solo sulle serie che hanno funzionato)
        average_valid_mse = np.mean(valid_mses)
        print(f"\nGrid Search Average Best Validation MSE (on {len(valid_mses)} series): {average_valid_mse:.4f}")
        # Trova la combinazione di parametri che è stata la migliore più frequentemente
        most_common_params_tuple = Counter(best_params_list).most_common(1)[0][0]
        # Riconverti la tupla in dizionario
        best_global_params = dict(most_common_params_tuple)
        print(f"Best Global Parameters (most frequent best): {best_global_params}")


    # --- Run Finale con Parametri Migliori ---
    print(f"\nStarting final prediction run using best global parameters...")
    final_run_start_time = time.time()
    # Usa TUTTI i dati pre-2023 per il training finale
    df_final_train_input = df_history_pre_2023[['Country', 'Product', 'ds', 'y']].copy()
    if not df_final_train_input.empty:
         last_final_train_date = df_final_train_input['ds'].max()
    else: sys.exit("Error: No data for final training run.")

    worker_final_func = partial(process_combination_final,
                                full_train_df=df_final_train_input, # Passa df completo pre-2023
                                last_hist_date=last_final_train_date,
                                best_global_params=best_global_params) # Passa i parametri trovati

    all_final_forecasts = []
    # Riusa le unique_combinations basate sui dati originali pre-2023
    final_unique_combinations = df_final_train_input[['Country', 'Product']].drop_duplicates().values.tolist()
    with Pool(processes=NUM_PROCESSES) as pool:
        results_iterator = pool.imap_unordered(worker_final_func, final_unique_combinations)
        for result_df in tqdm(results_iterator, total=len(final_unique_combinations), desc="Final Prediction"):
             if result_df is not None:
                 all_final_forecasts.append(result_df)
    print(f"Final prediction run complete in {time.time() - final_run_start_time:.2f}s.")


    # --- Combine, Format, Save, Evaluate Finale ---
    if not all_final_forecasts: sys.exit("Error: No final forecasts were generated.")
    print("Combining and formatting final forecasts...")
    format_start_time = time.time()
    final_forecast = pd.concat(all_final_forecasts, ignore_index=True)
    final_forecast['Quantity'] = final_forecast['yhat'].round().astype(int)
    final_forecast['Quantity'] = final_forecast['Quantity'].clip(lower=0) # Assicura non-negatività
    final_forecast['ds'] = pd.to_datetime(final_forecast['ds'])
    # Filtra per assicurarsi che siano solo date del 2023
    final_forecast = final_forecast[final_forecast['ds'].dt.year == 2023].copy()
    final_forecast['Month'] = final_forecast['ds'].dt.strftime('%b%Y').str.capitalize()
    final_forecast.sort_values(by=['Country', 'Product', 'ds'], inplace=True)
    output_df = final_forecast[['Country', 'Product', 'Month', 'Quantity']]
    print(f"Combining and formatting complete in {time.time() - format_start_time:.2f}s.")
    expected_rows = len(final_unique_combinations) * 12 # Usa le combinazioni trovate nei dati finali di training
    if len(output_df) != expected_rows:
        print(f"Warning: Final output rows ({len(output_df)}) mismatch expected ({expected_rows}). Might indicate issues in final run for some series.", file=sys.stderr)

    output_file_path = OUTPUT_DIR / OUTPUT_PREDICTION_CSV
    print(f"Saving final predictions to {output_file_path}...")
    save_start_time = time.time()
    output_df.to_csv(output_file_path, index=False)
    print(f"Predictions saved successfully in {time.time() - save_start_time:.2f}s.")
    total_script_time = time.time() - main_start_time
    print(f"\nScript finished in {total_script_time:.2f}s.")

    # --- Valutazione Finale (con metrica custom esterna) ---
    if EVALUATE_AVAILABLE:
        print("\nRunning final evaluation using evaluate.py (with custom loss)...")
        try:
            loss = evaluate() # Questa chiamata usa la metrica custom
            print(f"Final Evaluation Loss (Custom Metric): {loss}")
        except Exception as e:
            print(f"Error during final evaluation call: {e}")
    else:
        print("\nFinal evaluation skipped.")