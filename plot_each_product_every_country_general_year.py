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

# Raggruppa i dati per Product
prodotti = data['Product'].unique()
print(f"Prodotti trovati: {len(prodotti)}")
stato = data['Country'].unique()
print(f"stati trovati: {len(stato)}")
input("Premi invio per continuare...")
for prodotto in prodotti:
    subset = data[data['Product'] == prodotto]
    plt.figure(figsize=(10, 6))
    
    # Crea un grafico scatter per ogni nazione
    for nazione in subset['Country'].unique():
        nazione_data = subset[subset['Country'] == nazione]
        plt.scatter(nazione_data['Month'], nazione_data['Quantity'], label=nazione)
    
    # Configura il grafico
    plt.title(f"Produzione per prodotto: {prodotto}")
    plt.xlabel("Mese")
    plt.ylabel("Quantità Prodotta")
    plt.legend(title="Nazione")
    plt.grid(True)
    
    # Salva il grafico come immagine
    output_path = f"/home/alessiovalle/Scrivania/hackathon/{prodotto}_produzione.png"
    plt.show()
    print(f"Grafico salvato: {output_path}")
    plt.close()