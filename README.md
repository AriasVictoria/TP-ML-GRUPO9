Predicción de Precio de Bitcoin (BTC-USD) - Machine Learning 🚀

👥 Integrantes

- Arias Victoria  
- Gretter Alejandro  
- Molina Juan Ignacio

📋 Descripción del Proyecto

Este proyecto implementa modelos de Machine Learning para predecir el precio de cierre de Bitcoin (BTC-USD) para los próximos 7 días. El flujo completo incluye:

- Descarga y preparación de datos

- Análisis exploratorio y creación de features

- Entrenamiento y comparación de modelos

- Generación automática de predicciones diarias

🎯 Objetivo

Predecir el precio de cierre de Bitcoin para un horizonte de 7 días (D+1 hasta D+7) utilizando:

- Datos históricos de precio (OHLCV)

- Variables derivadas de precios de oro, S&P500 y FED Funds Rate

- Modelos multisalida: Ridge, Random Forest, LightGBM

🧪 Metodología

- Estrategia: Multi-step directo (un modelo multisalida por 7 días)

- Validación: Split temporal (80% train / 20% test)

- Métricas: MAE y RMSE por día de horizonte

```markdown
📁 Estructura del Repositorio

TP-ML-GRUPO9/
│
├─ analisis_exploratorio/
│   ├─ features.py                        # Generación de features a partir del dataset procesado
│   └─ data_fetch.py                       # Descarga y manejo de datos crudos
│
├─ data/
│   ├─ raw/                               # Archivos de datos originales
│   │   ├─ BTC-USD_daily.csv
│   │   ├─ GOLD.csv
│   │   ├─ SP500.csv
│   │   └─ FEDFUNDS.csv
│   │
│   └─ processed/                          # Datos preprocesados y matrices para entrenamiento
│       ├─ scaler.pkl                      # Escalador guardado
│       ├─ X.npy                           # Matriz de features
│       ├─ y.npy                           # Matriz de targets
│
├─ experimento/
│   └─ train_and_evaluate.py              # Entrenamiento y evaluación de modelos
│
├─ modelos/
│   ├─ gru_model.h5                        # Modelo GRU entrenado
│   └─ rf_model.pkl                         # Modelo Random Forest entrenado
│
├─ resultados/
│   └─ predict_7days.py                    # Script para generar predicciones para los próximos 7 días
│
├─ data_preparation.py                     # Preparación y limpieza del dataset completo
├─ download_data.py                        # Descarga de datos desde las fuentes originales
├─ memoria_TP_ML_grupo_9.pdf              # Memoria técnica del proyecto
└─ README.md                               # Archivo de documentación del repositorio
```

🔬 Modelos Implementados

| Modelo                  | Descripción                                |
|-------------------------|--------------------------------------------|
| Ridge Regression        | Baseline lineal regularizado               |
| Random Forest Regressor | Robusto ante ruido y no linealidades       |
| LightGBM Regressor      | Mejor desempeño y estabilidad temporal     |

📊 Features Utilizadas

| Feature                    | Descripción                                |
|----------------------------|--------------------------------------------|
| ma_7, std_7                | Media móvil y desviación estándar 7 días   |
| return_1d                  | Retorno diario del precio BTC              |
| Variables derivadas del oro| Precio de cierre del oro                    |
| Variables derivadas del S&P500 | Precio de cierre del índice S&P500     |
| Variables derivadas de FED Funds | Tasa de fondos federales             |

📈 Evaluación de Modelos

| Modelo                  | MAE Promedio | Observación                                   |
|-------------------------|-------------|-----------------------------------------------|
| Ridge Regression        | Alto        | Baseline lineal                               |
| Random Forest Regressor | Medio       | Mejor que Ridge, robusto a no linealidades   |
| LightGBM Regressor      | Bajo        | Mejor desempeño global y estabilidad temporal|

Error esperado: 3–5% del precio diario para horizontes de 1 a 3 días, aumentando hacia el día 7.

💾 Generación de Predicciones Diarias

- Script: resultados/predict_7days.py

- Output: data/processed/predicciones_diarias/predicciones_7dias_YYYY-MM-DD.csv

- Impresión también en terminal

Ejemplo de salida:

Date	Predicted_Close

2025-10-17	103619.52

2025-10-18	107324.86

2025-10-19	105909.22

…	…

🚀 Ejecución Rápida

1- Descargar datos más recientes:

python download_data.py

2- Preparar dataset completo:

python data_preparation.py

3- Generar features y matrices X/y:

python analisis_exploratorio/features.py

4- Entrenar y evaluar modelos:

python experimento/train_and_evaluate.py

5- Generar predicción de los próximos 7 días:

python resultados/predict_7days.py

📦 Dependencias

pip install pandas numpy scikit-learn lightgbm yfinance joblib matplotlib

📝 Conclusiones

- Flujo completo y reproducible de predicción BTC implementado

- Se compararon tres modelos multisalida, LightGBM mostró mejor desempeño

- Predicciones automáticas para 7 días con actualización diaria

- so de múltiples fuentes de información (BTC, GOLD, SP500, FED Funds)