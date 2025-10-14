import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DATASET_FILE = DATA_PROCESSED_DIR / "dataset_completo.csv"
X_FILE = DATA_PROCESSED_DIR / "X.npy"
y_FILE = DATA_PROCESSED_DIR / "y.npy"
DATES_FILE = DATA_PROCESSED_DIR / "dates.npy"

HORIZON = 7

logging.info(f"Cargando dataset completo desde {DATASET_FILE}")
df = pd.read_csv(DATASET_FILE, parse_dates=["Date"])

# ---- Crear features ----
df["ma_7"] = df["Close"].rolling(7).mean()
df["std_7"] = df["Close"].rolling(7).std()
df["return_1d"] = df["Close"].pct_change(1)

df = df.dropna().reset_index(drop=True)

# ---- Columnas a usar como features ----
feature_cols = [c for c in df.columns if c not in ["Date", "Close", "Close_gold", "Close_sp", "Close_google"]]

X, y, dates = [], [], []

for i in range(len(df) - HORIZON):
    X.append(df.loc[i, feature_cols].values)
    y.append(df.loc[i+1:i+HORIZON, "Close"].values)
    dates.append(df.loc[i, "Date"])

X = np.array(X)
y = np.array(y)
dates = np.array(dates)

np.save(X_FILE, X)
np.save(y_FILE, y)
np.save(DATES_FILE, dates)
logging.info(f"X shape: {X.shape}, y shape: {y.shape}")
logging.info(f"Matrizes guardadas en {DATA_PROCESSED_DIR}")
