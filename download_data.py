import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# Rutas
RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)
BTC_FILE = os.path.join(RAW_DIR, "BTC-USD_daily.csv")

# Fechas
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)  # últimos 5 años

# Descargar BTC desde yfinance
btc = yf.download("BTC-USD", start=start_date, end=end_date, interval="1d")
btc.reset_index(inplace=True)

# Guardar CSV
btc.to_csv(BTC_FILE, index=False)
print(f"✅ BTC-USD diario de los últimos 5 años guardado en: {BTC_FILE}")
