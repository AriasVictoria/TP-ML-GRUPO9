import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ruta al archivo CSV
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_FILE = DATA_DIR / "BTC-USD_daily.csv"

def download_btc_csv():
    """
    Descarga datos de BTC-USD si el archivo no existe.
    Guarda el CSV en la carpeta /data/raw.
    """
    if not CSV_FILE.exists():
        logging.info("Descargando histórico de BTC-USD desde Yahoo Finance...")
        df = yf.download("BTC-USD", period="1y", interval="1d")
        if df.empty:
            logging.warning("No se descargaron datos. Verificá la conexión.")
            return None
        df.reset_index(inplace=True)
        df.to_csv(CSV_FILE, index=False)
        logging.info(f"CSV guardado en {CSV_FILE}")
    else:
        logging.info(f"CSV existente: {CSV_FILE}")

def load_raw():
    """
    Carga el CSV de BTC-USD desde /data/raw.
    Retorna un DataFrame con la columna 'Date' parseada.
    """
    download_btc_csv()
    df = pd.read_csv(CSV_FILE, parse_dates=["Date"])
    if "Close" not in df.columns:
        raise ValueError("La columna 'Close' no está en el archivo CSV.")
    return df

def add_features(df):
    df.columns = [col.strip().capitalize() for col in df.columns]

    numeric_cols = ["Open", "High", "Low", "Close", "Adj close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Features alineados con horizon=7
    df["ma_7"] = df["Close"].rolling(7).mean()
    df["std_7"] = df["Close"].rolling(7).std()
    df["ma_14"] = df["Close"].rolling(14).mean()  # opcional, más largo
    df["return_1d"] = df["Close"].pct_change(1)

    print(f"Filas antes de dropna: {len(df)}")
    df = df.dropna()
    print(f"Filas después de dropna: {len(df)}")

    return df

def get_X_y_for_horizon(df, horizon=7, feature_cols=None):
    """
    Prepara matrices X (features) e y (targets) para predicción multistep.

    Parámetros:
    - df: DataFrame con features.
    - horizon: cantidad de días a predecir.
    - feature_cols: lista de columnas a usar como entrada.

    Retorna:
    - X: matriz de features (n_samples x n_features)
    - y: matriz de targets (n_samples x horizon)
    - dates: fechas asociadas a cada muestra
    - feature_cols: lista final de columnas usadas
    """
    df = df.copy().reset_index(drop=True)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ["Date", "Open", "High", "Low", "Close", "Adj Close", "close"]]

    X, y, dates = [], [], []
    for i in range(len(df) - horizon):
        X.append(df.loc[i, feature_cols].values)
        y.append(df.loc[i+1:i+horizon, "Close"].values)
        dates.append(df.loc[i, "Date"])
    return np.array(X), np.array(y), dates, feature_cols
