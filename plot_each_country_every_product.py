import pandas as pd
import matplotlib.pyplot as plt
import os

# Leggi il file CSV
file_path = '/home/alessiovalle/Scrivania/hackathon/data/01_input_history.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Il file {file_path} non esiste.")

data = pd.read_csv(file_path)

# Controlla che il file abbia le colonne necessarie
required_columns = ['Country', 'Product', 'Month', 'Quantity']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"Il file CSV deve contenere le colonne: {', '.join(required_columns)}")

# Converte la colonna 'Month' in formato datetime
data['Month'] = pd.to_datetime(data['Month'])

# Raggruppa i dati per Country

nazioni = data['Country'].unique()

for Country in nazioni:
    subset = data[data['Country'] == Country]
    plt.figure(figsize=(10, 6))
    
    # Crea un grafico scatter per ogni tipo di prodotto
    for prodotto in subset['Product'].unique():
        prodotto_data = subset[subset['Product'] == prodotto]
        plt.scatter(prodotto_data['Month'], prodotto_data['Quantity'], label=prodotto)
    
    # Configura il grafico
    plt.title(f"Produzione per {Country}")
    plt.xlabel("Data")
    plt.ylabel("Quantità Prodotta")
    plt.legend(title="Tipo di Prodotto")
    plt.grid(True)

    # Salva il grafico come immagine
    output_path = f"/home/alessiovalle/Scrivania/hackathon/country/{Country}_produzione.png"
    plt.savefig(output_path, format='png')
    print(f"Grafico salvato: {output_path}")
    plt.close()