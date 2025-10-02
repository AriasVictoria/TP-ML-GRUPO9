import yfinance as yf
import pandas as pd
from pathlib import Path
import datetime
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ruta donde se guardarán los datos
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_btcusd(start="2015-01-01", end=None, interval="1d"):
    """
    Descarga datos OHLCV de BTC-USD desde Yahoo Finance y guarda como CSV.

    Parámetros:
    - start (str): Fecha de inicio en formato 'YYYY-MM-DD'.
    - end (str): Fecha de fin en formato 'YYYY-MM-DD'. Si no se especifica, se usa la fecha actual.
    - interval (str): Intervalo de tiempo ('1d', '1h', etc.).

    Retorna:
    - df (DataFrame): Datos descargados.
    """
    if end is None:
        end = datetime.datetime.today().strftime('%Y-%m-%d')

    ticker = "BTC-USD"
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)

    if df.empty:
        logging.warning("No se descargaron datos. Verificá la conexión o el rango de fechas.")
        return None

    df = df.rename_axis("Date").reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]

    csv_path = DATA_DIR / "BTC-USD_daily.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"Datos guardados en {csv_path} ({len(df)} filas)")
    return df

if __name__ == "__main__":
    fetch_btcusd()
