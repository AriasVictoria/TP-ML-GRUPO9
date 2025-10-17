import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from datetime import datetime, timedelta

# Configuración logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Rutas 
ROOT_DIR = Path(__file__).resolve().parents[0]
RAW_DIR = ROOT_DIR.parent / "data" / "raw"
PROCESSED_DIR = ROOT_DIR.parent / "data" / "processed"
MODEL_DIR = ROOT_DIR.parent / "models"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BTC_FILE = RAW_DIR / "BTC.csv"
GOLD_FILE = RAW_DIR / "GOLD.csv"
SP_FILE = RAW_DIR / "SP500.csv"
FED_FILE = RAW_DIR / "FEDFUNDS.csv"
DATASET_FILE = PROCESSED_DIR / "dataset_completo_mejorado.csv"
X_FILE = PROCESSED_DIR / "X.npy"
MODEL_FILE = MODEL_DIR / "rf_model.pkl"

# Fechas
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)
logging.info(f"Descargando datos desde {start_date.date()} hasta {end_date.date()}")

# Función para guardar
def save_csv(df, path, name):
    if df is not None and not df.empty:
        df.to_csv(path, index=True)
        logging.info(f"{name} guardado en {path}")
    else:
        logging.warning(f"No se descargaron datos para {name}")

# Descarga de datos 
def download_yf(ticker, path, name):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
        save_csv(df, path, name)
        return df
    except Exception as e:
        logging.warning(f"No se descargaron datos para {name}: {e}")
        return pd.DataFrame()

btc_df = download_yf("BTC-USD", BTC_FILE, "BTC")
gold_df = download_yf("GC=F", GOLD_FILE, "Oro")
sp_df = download_yf("^GSPC", SP_FILE, "S&P500")
fed_df = download_yf("^IRX", FED_FILE, "FEDFUNDS")

# Preparar dataset completo
logging.info("Preparando dataset completo...")

# Renombrar columnas
btc_df = btc_df.rename(columns={'Close':'Close_BTC','Open':'Open_BTC','High':'High_BTC','Low':'Low_BTC','Volume':'Volume_BTC'})
gold_df = gold_df.rename(columns={'Close':'Close_GOLD'})
sp_df = sp_df.rename(columns={'Close':'Close_SP500'})
fed_df = fed_df.rename(columns={'Close':'Close_FED'})

# Merge por fecha
df_merged = pd.merge_asof(btc_df.sort_values('Date'), gold_df.sort_values('Date'), on='Date', direction='backward')
df_merged = pd.merge_asof(df_merged.sort_values('Date'), sp_df.sort_values('Date'), on='Date', direction='backward')
df_merged = pd.merge_asof(df_merged.sort_values('Date'), fed_df.sort_values('Date'), on='Date', direction='backward')

df_merged.fillna(method='ffill', inplace=True)
df_merged.to_csv(DATASET_FILE, index=False)
logging.info(f"Dataset completo guardado en {DATASET_FILE}")

# Generar X.npy
features = ['Close_BTC', 'Close_GOLD', 'Close_SP500', 'Close_FED', 'Open_BTC', 'High_BTC', 'Low_BTC']
X = df_merged[features].values
np.save(X_FILE, X)
logging.info(f"Matriz X guardada en {X_FILE}")

# Cargar modelo y predecir
try:
    model = joblib.load(MODEL_FILE)
    last_X = X[-1].reshape(1, -1)
    pred = model.predict(last_X)

    # Guardar predicciones
    PRED_DIR = PROCESSED_DIR / "predicciones_diarias"
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    start_pred_date = datetime.today() + timedelta(days=1)
    pred_dates = [start_pred_date + timedelta(days=i) for i in range(len(pred.flatten()))]

    pred_df = pd.DataFrame({
        "Fecha": [d.date() for d in pred_dates],
        "Prediccion": pred.flatten()
    })

    run_date = datetime.today().strftime("%Y-%m-%d")
    PRED_FILE_CSV = PRED_DIR / f"predicciones_7dias_{run_date}.csv"
    pred_df.to_csv(PRED_FILE_CSV, index=False)
    logging.info(f"Predicciones guardadas en CSV en {PRED_FILE_CSV}")

    print("\nPredicciones para los próximos 7 días (desde mañana):\n")
    for fecha, valor in zip(pred_df["Fecha"], pred_df["Prediccion"]):
        print(f"{fecha} : {valor:.2f}")

except Exception as e:
    logging.error(f"No se pudo cargar el modelo o hacer la predicción: {e}")
