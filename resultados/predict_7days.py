import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from features import load_raw, add_features
from datetime import timedelta
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carpeta de salida
OUT_DIR = Path(__file__).resolve().parents[1] / "resultados" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_latest_model(model_name="rf_multioutput"):
    """
    Carga el modelo entrenado y las columnas de features usadas.
    """
    model_path = Path(__file__).resolve().parents[1] / "resultados" / f"{model_name}.joblib"
    meta_path = Path(__file__).resolve().parents[1] / "resultados" / "train_results.joblib"
    model = joblib.load(model_path)
    meta = joblib.load(meta_path)
    feat_cols = meta["feat_cols"]
    return model, feat_cols

def prepare_last_row(df, feat_cols):
    """
    Calcula features hasta el último día disponible y devuelve un array con las mismas columnas de features.
    """
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    df_features = add_features(df_sorted)
    last_feat_row = df_features.iloc[-1]
    X = last_feat_row[feat_cols].values.reshape(1, -1)
    return X, last_feat_row

def predict_multioutput(model, X):
    """
    Predicción directa multihorizonte (7 días) usando MultiOutputRegressor.
    """
    preds = model.predict(X)
    return preds.flatten()

def predict_recursive(model, df, feat_cols, horizon=7):
    """
    Predicción recursiva: predice 1 día, lo agrega a df, recalcula features y repite.
    """
    df_work = df.copy().sort_values("Date").reset_index(drop=True)
    preds = []
    for i in range(horizon):
        df_features = add_features(df_work)
        X = df_features.iloc[-1][feat_cols].values.reshape(1, -1)
        next_pred = model.predict(X)[0][0]
        val = np.asarray(next_pred).ravel()[0] if isinstance(next_pred, (list, np.ndarray)) else next_pred
        preds.append(val)

        # Simula nueva fila con predicción
        new_date = df_work.iloc[-1]["Date"] + timedelta(days=1)
        new_row = {
            "Date": new_date,
            "Open": val, "High": val, "Low": val,
            "Close": val, "Adj Close": val,
            "Volume": df_work.iloc[-1]["Volume"]
        }
        df_work = pd.concat([df_work, pd.DataFrame([new_row])], ignore_index=True)
    return np.array(preds)

def save_prediction_csv(preds, model_name):
    """
    Guarda las predicciones en un CSV con fecha de ejecución.
    """
    today = pd.Timestamp.now(tz="UTC").normalize()
    rows = []
    for i, p in enumerate(preds, start=1):
        target_date = (today + pd.Timedelta(days=i)).date()
        rows.append({
            "model": model_name,
            "pred_date_utc": today.date(),
            "target_date": target_date,
            "pred_close": float(p)
        })
    df_out = pd.DataFrame(rows)
    filename = OUT_DIR / f"pred_{model_name}_{today.date()}.csv"
    df_out.to_csv(filename, index=False)
    logging.info(f"Predicción guardada en {filename}")
    return filename

def main(model_name="rf_multioutput", strategy="multioutput"):
    """
    Ejecuta la predicción de 7 días con la estrategia seleccionada.
    """
    model, feat_cols = load_latest_model(model_name)
    df = load_raw()
    df = add_features(df)

    if strategy == "multioutput":
        X, _ = prepare_last_row(df, feat_cols)
        preds = predict_multioutput(model, X)
    else:
        preds = predict_recursive(model, df, feat_cols, horizon=7)

    save_prediction_csv(preds, f"{model_name}_{strategy}")
    return preds

if __name__ == "__main__":
    preds = main(model_name="rf_multioutput", strategy="multioutput")
    logging.info(f"Predicciones: {preds}")
