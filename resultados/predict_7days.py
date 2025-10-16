import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import timedelta
import requests

# -------------------------- Rutas --------------------------
DATA_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
X_FILE = DATA_PROCESSED_DIR / "X.npy"
MODEL_FILE = MODELS_DIR / "rf_model.pkl"
OUTPUT_DIR = DATA_PROCESSED_DIR / "predicciones_diarias"
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------- Función para precio BTC real --------------------------
def obtener_precio_real_btc():
    try:
        response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd")
        response.raise_for_status()
        precio = response.json()["bitcoin"]["usd"]
        return precio
    except Exception as e:
        print("⚠️ No se pudo obtener el precio real de BTC:", e)
        return None

# -------------------------- Cargar datos --------------------------
print("Cargando X...")
X = np.load(X_FILE, allow_pickle=True)
print("Cargando modelo...")
model = joblib.load(MODEL_FILE)

# -------------------------- Última ventana de entrada --------------------------
last_X = X[-1].reshape(1, -1)

# -------------------------- Fecha de inicio --------------------------
hoy = pd.Timestamp.today().normalize()
start_date = hoy + timedelta(days=1)  # siempre predicción desde mañana

# -------------------------- Predicción --------------------------
pred = model.predict(last_X)
if isinstance(pred, (list, np.ndarray)) and np.ndim(pred) > 1:
    pred = pred.flatten()

# -------------------------- Ajustar primera predicción al precio real --------------------------
precio_real = obtener_precio_real_btc()
if precio_real:
    pred[0] = precio_real

# -------------------------- Fechas futuras --------------------------
future_dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]

# -------------------------- Formatear precios --------------------------
def formatear_precio(x):
    return f"{x:,.2f} USD".replace(",", "X").replace(".", ",").replace("X", ".")

pred_formateado = [formatear_precio(p) for p in pred[:7]]

# -------------------------- Crear DataFrame --------------------------
df_pred = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Close": pred_formateado
})

# -------------------------- Guardar CSV diario --------------------------
archivo_salida = OUTPUT_DIR / f"prediccion_{hoy.strftime('%Y-%m-%d')}.csv"
df_pred.to_csv(archivo_salida, index=False, sep=",", header=True)
print(f"✅ Predicciones ajustadas guardadas en {archivo_salida}")
print(df_pred)
