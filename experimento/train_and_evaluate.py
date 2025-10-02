import sys
from pathlib import Path

# Agrega la carpeta /analisis_exploratorio al path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "analisis_exploratorio"))

# Imports del proyecto
import joblib
import logging
import numpy as np
import pandas as pd
from features import load_raw, add_features, get_X_y_for_horizon
from models import make_models, evaluate_model
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carpeta de salida
OUT_DIR = BASE_DIR / "resultados"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main(horizon=7, test_size=0.2):
    """
    Entrena y evalúa modelos para predicción multihorizonte de BTC.
    Incluye búsqueda de hiperparámetros para RandomForest.
    Guarda modelos y métricas en /resultados.
    """
    # 1. Cargar y procesar datos
    df = load_raw()
    df = add_features(df)

    # 🔍 Diagnóstico
    print("Columnas del DataFrame:", df.columns)
    print("Primeras filas del DataFrame:")
    print(df.head(10))
    print(f"Filas después de add_features: {len(df)}")

    logging.info(f"Filas después de add_features: {len(df)}")

    if len(df) <= horizon:
        raise ValueError(f"No hay suficientes filas ({len(df)}) para horizon={horizon}. Necesitás más datos o reducir horizon.")

    # 2. Preparar X, y
    X, y, dates, feat_cols = get_X_y_for_horizon(df, horizon=horizon)
    logging.info(f"X shape: {X.shape}, y shape: {y.shape}")

    if len(X) == 0:
        raise ValueError("No hay suficientes filas para entrenar. Revisá tu CSV o reducí el horizon.")

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # 3. Ajustar n_splits según tamaño de X_train
    n_splits = max(2, min(3, len(X_train)))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # 4. Búsqueda de hiperparámetros para RandomForest
    logging.info("=== Búsqueda de hiperparámetros para RandomForest ===")
    param_dist = {
        "estimator__n_estimators": [50, 100, 200],
        "estimator__max_depth": [5, 10, 20, None],
        "estimator__min_samples_split": [2, 5, 10]
    }

    search = RandomizedSearchCV(
        MultiOutputRegressor(RandomForestRegressor(random_state=42)),
        param_distributions=param_dist,
        n_iter=5,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_
    logging.info(f"Mejores parámetros encontrados: {search.best_params_}")

    # 5. Diccionario de modelos
    models = make_models()
    models["rf_multioutput"] = best_rf

    # 6. Entrenar y evaluar
    results = {}
    for name, model in models.items():
        logging.info(f"=== Entrenando: {name} ===")
        res = evaluate_model(model, X_train, y_train, X_test, y_test)
        results[name] = res
        joblib.dump(model, OUT_DIR / f"{name}.joblib")
        logging.info(f"{name} MAE por día: {res['mae_per_day']}")
        logging.info(f"{name} RMSE por día: {res['rmse_per_day']}")

        # 7. Guardar resultados
    joblib.dump({"results": results, "feat_cols": feat_cols}, OUT_DIR / "train_results.joblib")
    logging.info("Entrenamiento finalizado. Resultados guardados en /resultados")

    # 8. Crear carpeta de predicciones si no existe
    (OUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)

    # 9. Guardar predicciones del modelo RF
    rf_preds = results["rf_multioutput"]["preds"]
    pred_dates = dates[-len(rf_preds):]  # fechas correspondientes
    df_preds = pd.DataFrame(rf_preds, columns=[f"day_{i+1}" for i in range(horizon)])
    df_preds["date"] = pred_dates
    df_preds.to_csv(OUT_DIR / "predictions" / f"pred_rf_multioutput_{date.today()}.csv", index=False)
    logging.info("Predicciones guardadas en /resultados/predictions")

if __name__ == "__main__":
    main(horizon=7)
