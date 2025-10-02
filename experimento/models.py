import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def make_models():
    """
    Crea un diccionario con tres modelos multisalida:
    - Ridge (lineal regularizado)
    - Random Forest
    - LightGBM

    Todos envueltos en MultiOutputRegressor para predicción multistep.
    """
    models = {
        "ridge_multioutput": MultiOutputRegressor(Ridge()),
        "rf_multioutput": MultiOutputRegressor(RandomForestRegressor(n_jobs=-1, random_state=42)),
        "lgbm_multioutput": MultiOutputRegressor(lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1))
    }
    return models

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Entrena el modelo y calcula métricas MAE y RMSE por día de horizonte.

    Parámetros:
    - model: instancia del modelo
    - X_train, y_train: datos de entrenamiento
    - X_test, y_test: datos de prueba

    Retorna:
    - dict con MAE por día, RMSE por día y predicciones
    """
    logging.info("Entrenando modelo...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Validación de dimensiones
    if preds.shape != y_test.shape:
        raise ValueError("Las dimensiones de predicción y test no coinciden.")

    maes = np.mean(np.abs(preds - y_test), axis=0)
    rmses = np.sqrt(np.mean((preds - y_test)**2, axis=0))

    return {
        "mae_per_day": maes,
        "rmse_per_day": rmses,
        "preds": preds
    }

def time_series_search(model, param_distributions, X, y, n_iter=20):
    """
    Búsqueda aleatoria de hiperparámetros con validación temporal.

    Parámetros:
    - model: modelo base (no envuelto en MultiOutput)
    - param_distributions: diccionario de hiperparámetros
    - X, y: datos
    - n_iter: cantidad de combinaciones a probar

    Retorna:
    - objeto RandomizedSearchCV entrenado
    """
    logging.info("Iniciando búsqueda de hiperparámetros...")
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=42
    )
    search.fit(X, y)
    logging.info(f"Mejores parámetros: {search.best_params_}")
    return search

