import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

X = np.load(DATA_PROCESSED_DIR / "X.npy", allow_pickle=True)
y = np.load(DATA_PROCESSED_DIR / "y.npy", allow_pickle=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

logging.info(f"Split realizado: X_train={X_train.shape}, X_test={X_test.shape}")

# Crear y entrenar modelo
model = MultiOutputRegressor(RandomForestRegressor(n_jobs=-1, random_state=42))
logging.info("Entrenando Random Forest...")
model.fit(X_train, y_train)

# Evaluar
preds = model.predict(X_test)
mse = np.mean((preds - y_test)**2)
mae = np.mean(np.abs(preds - y_test))
logging.info(f"Evaluación: MSE={mse:.2f}, MAE={mae:.2f}")

# Guardar modelo
joblib.dump(model, MODELS_DIR / "rf_model.pkl")
logging.info(f"Modelo guardado en {MODELS_DIR / 'rf_model.pkl'}")
