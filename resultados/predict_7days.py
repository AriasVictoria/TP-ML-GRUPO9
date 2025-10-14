import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import timedelta

# --------------------------
# Rutas
# --------------------------
DATA_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

X_FILE = DATA_PROCESSED_DIR / "X.npy"
MODEL_FILE = MODELS_DIR / "rf_model.pkl"
DATASET_FILE = DATA_PROCESSED_DIR / "dataset_completo.csv"
OUTPUT_FILE = DATA_PROCESSED_DIR / "prediccion_7dias.csv"

# --------------------------
# Cargar datos
# --------------------------
print("Cargando X...")
X = np.load(X_FILE, allow_pickle=True)

print("Cargando modelo...")
model = joblib.load(MODEL_FILE)

# --------------------------
# Última ventana de entrada
# --------------------------
last_X = X[-1].reshape(1, -1)

# --------------------------
# Obtener última fecha real del dataset
# --------------------------
df = pd.read_csv(DATASET_FILE, parse_dates=["Date"])
last_date = df["Date"].max()

# --------------------------
# Forzar inicio en 14-10-2025
# --------------------------
start_date = pd.Timestamp("2025-10-14")

# --------------------------
# Predicción para 7 días futuros
# --------------------------
print("Generando predicciones...")

pred = model.predict(last_X)

# Si el modelo devuelve un array 2D (1,7), lo aplanamos
if isinstance(pred, (list, np.ndarray)) and np.ndim(pred) > 1:
    pred = pred.flatten()

future_dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]

# --------------------------
# Crear y guardar DataFrame
# --------------------------
df_pred = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Close": pred[:7]
})

df_pred.to_csv(OUTPUT_FILE, index=False, sep=",", header=True)

print(f"✅ Predicciones guardadas en {OUTPUT_FILE}")
print(df_pred)
