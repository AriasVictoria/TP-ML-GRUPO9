import pandas as pd
from pathlib import Path
import logging

# Configuración logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Rutas
DATA_DIR = Path(__file__).resolve().parents[0] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BTC_FILE = RAW_DIR / "BTC-USD_daily.csv"
GOLD_FILE = RAW_DIR / "GOLD.csv"
SP_FILE = RAW_DIR / "SP500.csv"
GOOGLE_FILE = RAW_DIR / "multiTimeline.csv"
FED_FILE = RAW_DIR / "FEDFUNDS.csv"  # Opcional: tasas de interés

OUTPUT_FILE = PROCESSED_DIR / "dataset_completo_mejorado.csv"

def read_csv(file_path, name, rename_close=None):
    logging.info(f"Leyendo {name}...")
    df = pd.read_csv(file_path)
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    if rename_close:
        df.rename(columns={df.columns[1]: rename_close}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.dropna(subset=["Date"], inplace=True)
    return df

btc = read_csv(BTC_FILE, "BTC")
gold = read_csv(GOLD_FILE, "Oro", "Close_gold")
sp500 = read_csv(SP_FILE, "S&P500", "Close_sp")
google = read_csv(GOOGLE_FILE, "Google Trends", "Close_google")
fed = read_csv(FED_FILE, "FED Funds", "FEDFUNDS")

# Merge datasets
df = pd.merge_asof(btc.sort_values('Date'), gold.sort_values('Date'), on='Date', direction='backward')
df = pd.merge_asof(df.sort_values('Date'), sp500.sort_values('Date'), on='Date', direction='backward')
df = pd.merge_asof(df.sort_values('Date'), google.sort_values('Date'), on='Date', direction='backward')
df = pd.merge_asof(df.sort_values('Date'), fed.sort_values('Date'), on='Date', direction='backward')

df.fillna(method='ffill', inplace=True)  # Rellenar datos faltantes

df.to_csv(OUTPUT_FILE, index=False)
logging.info(f"✅ Dataset completo guardado en {OUTPUT_FILE}")
