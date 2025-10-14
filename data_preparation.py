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

OUTPUT_FILE = PROCESSED_DIR / "dataset_completo.csv"

def read_csv_with_date(file_path, name, rename_close=None):
    logging.info(f"Leyendo {name}...")
    df = pd.read_csv(file_path)
    
    # Si hay al menos 2 columnas, la primera será Date, la segunda Close
    if df.shape[1] >= 2:
        df = df.rename(columns={df.columns[0]: "Date"})
        if rename_close:
            df = df.rename(columns={df.columns[1]: rename_close})
    # Si solo hay 1 columna (valores de Google Trends), generamos fechas automáticamente
    elif df.shape[1] == 1:
        df = df.rename(columns={df.columns[0]: rename_close if rename_close else "Value"})
        df["Date"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))
    else:
        raise ValueError(f"{name} no tiene columnas válidas")
    
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Date"])  # <<< ELIMINA FECHAS NULAS
    return df

# Carga de datasets
btc = read_csv_with_date(BTC_FILE, "BTC")
gold = read_csv_with_date(GOLD_FILE, "Oro", rename_close="Close_gold")
sp500 = read_csv_with_date(SP_FILE, "S&P500", rename_close="Close_sp")
google = read_csv_with_date(GOOGLE_FILE, "Google Trends", rename_close="Close_google")

# Merge datasets
df = pd.merge_asof(btc.sort_values('Date'),
                   gold.sort_values('Date'),
                   on='Date', direction='backward')
df = pd.merge_asof(df.sort_values('Date'),
                   sp500.sort_values('Date'),
                   on='Date', direction='backward')
df = pd.merge_asof(df.sort_values('Date'),
                   google.sort_values('Date'),
                   on='Date', direction='backward')

# Forward-fill para Google Trends
if "Close_google" in df.columns:
    df['Close_google'] = df['Close_google'].ffill()

# Guardar dataset final
df.to_csv(OUTPUT_FILE, index=False)
logging.info(f"✅ Dataset completo guardado en {OUTPUT_FILE}")
