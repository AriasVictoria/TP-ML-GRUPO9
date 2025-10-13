import joblib
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
import logging

# Agregar la carpeta raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analisis_exploratorio.features import load_raw, add_features

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carpeta de salida
OUT_DIR = Path(__file__).resolve().parents[1] / "resultados" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_latest_model(model_name="rf_multioutput"):
    model_path = Path(__file__).resolve().parents[1] / "resultados" / f"{model_name}.joblib"
    meta_path = Path(__file__).resolve().parents[1] / "resultados" / "train_results.joblib"
    model = joblib.load(model_path)
    meta = joblib.load(meta_path)
    feat_cols = meta["feat_cols"]
    return model, feat_cols

def prepare_last_row(df, feat_cols):
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    df_features = add_features(df_sorted)
    last_feat_row = df_features.iloc[-1]
    X = last_feat_row[feat_cols].values.reshape(1, -1)
    return X, last_feat_row

def predict_multioutput(model, X):
    preds = model.predict(X)
    return preds.flatten()

def predict_recursive(model, df, feat_cols, horizon=7):
    df_work = df.copy().sort_values("Date").reset_index(drop=True)
    preds = []
    for i in range(horizon):
        df_features = add_features(df_work)

        # Convertir solo las columnas existentes
        cols_to_numeric = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        cols_existing = [c for c in cols_to_numeric if c in df_work.columns]
        for col in cols_existing:
            df_work[col] = pd.to_numeric(df_work[col], errors="coerce")

        X = df_features.iloc[-1][feat_cols].values.reshape(1, -1)
        next_pred = model.predict(X)[0][0]
        val = np.asarray(next_pred).ravel()[0] if isinstance(next_pred, (list, np.ndarray)) else next_pred
        preds.append(val)

        new_date = df_work.iloc[-1]["Date"] + timedelta(days=1)
        new_row = {
            "Date": new_date,
            "Open": val,
            "High": val,
            "Low": val,
            "Close": val,
            "Volume": float(df_work.iloc[-1]["Volume"])
        }
        if "Adj Close" in df_work.columns:
            new_row["Adj Close"] = val

        df_new = pd.DataFrame([new_row], columns=df_work.columns)
        df_work = pd.concat([df_work, df_new], ignore_index=True)

    return np.array(preds)

def save_prediction_csv(preds, model_name):
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

def compare_with_real(preds, df):
    last_known_date = df["Date"].max()
    future_dates = [last_known_date + timedelta(days=i) for i in range(1, 8)]
    df_pred = pd.DataFrame({"Date": future_dates, "Predicted_Close": preds})

    # Emparejar con valores reales si existen
    df_real = df[df["Date"].isin(future_dates)][["Date", "Close"]]
    if df_real.empty:
        logging.warning("No hay valores reales disponibles para comparar todavía.")
        return df_pred

    merged = df_pred.merge(df_real, on="Date", how="left")
    merged["Diff"] = merged["Predicted_Close"] - merged["Close"]
    merged["Abs_Error"] = merged["Diff"].abs()

    logging.info("\n" + merged.to_string(index=False))
    mae = merged["Abs_Error"].mean()
    logging.info(f"Error promedio (MAE): {mae:.2f} USD")
    return merged

def main(model_name="rf_multioutput", strategy="recursive"):
    model, feat_cols = load_latest_model(model_name)
    df = load_raw()
    df = add_features(df)

    if strategy == "multioutput":
        X, _ = prepare_last_row(df, feat_cols)
        preds = predict_multioutput(model, X)
    else:
        preds = predict_recursive(model, df, feat_cols, horizon=7)

    save_prediction_csv(preds, f"{model_name}_{strategy}")
    compare_with_real(preds, df)
    return preds

if __name__ == "__main__":
    preds = main(model_name="rf_multioutput", strategy="recursive")  # probá recursive para 7 días
    logging.info(f"Predicciones: {preds}")
