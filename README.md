*-Predicción del Precio de Cierre de Bitcoin (BTC-USD)-*
Grupo 9 – Trabajo Práctico de Machine Learning

*Integrantes:*
Victoria Arias
Alejandro Gretter
Juan Molina

*Objetivo del proyecto*
Desarrollar un sistema de aprendizaje automático (Machine Learning) capaz de predecir el precio de cierre diario del Bitcoin (BTC-USD) para los próximos 7 días, utilizando datos históricos financieros y de interés público.

**El proyecto abarca todo el flujo de trabajo de un modelo de predicción real:**

1- Obtención y preparación de datos.

2- Análisis exploratorio y creación de features.

3- Entrenamiento y comparación de modelos.

4- Generación automática de predicciones diarias.

*Estructura del repositorio*
TP-ML-GRUPO9/
│
├── analisis_exploratorio/
│   └── features.py             
│
├── data/
│   ├── raw/                     
│   └── processed/              
│
├── experimento/
│   └── train_and_evaluate.py    
│
├── modelos/
│   └── models.py               
│
├── resultados/
│   └── predict_7days.py       
│
├── data_preparation.py         
├── download_data.py             
├── memoria_TP_ML_grupo_9.pdf    
└── README.md

*Fuentes de datos*
El sistema utiliza datos históricos provenientes de diversas fuentes:
Fuente	     Variable principal	     Archivo
Yahoo -------- Finance	-----   Precio de Bitcoin (BTC-USD)	BTC-USD_daily.csv
Yahoo Finance--	Oro (GOLD) ----- GOLD.csv
Yahoo Finance -- S&P 500 Index ---- SP500.csv
Google Trends ------ Término “Bitcoin” (popularidad) ----- multiTimeline.csv
FRED ----- Tasa de fondos federales (FED Funds) ---- FEDFUNDS.csv

*Ejecución del flujo completo*
1. Descargar los datos más recientes
python download_data.py

2. Preparar el dataset completo
python data_preparation.py

3. Generar las features y las matrices X/y
python analisis_exploratorio/features.py

4. Entrenar y evaluar los modelos
python experimento/train_and_evaluate.py

5. Generar la predicción de los próximos 7 días
python resultados/predict_7days.py

*Modelos implementados*
* Ridge Regression: modelo lineal regularizado, utilizado como baseline.
* Random Forest Regressor: modelo de árboles de decisión promediados, robusto ante ruido.
* LightGBM Regressor: modelo de boosting de gradiente, elegido como el de mejor desempeño.

Cada modelo fue evaluado mediante métricas MAE (Error Absoluto Medio) y RMSE (Raíz del Error Cuadrático Medio), utilizando un esquema multisalida que predice simultáneamente los 7 valores futuros.

*Predicción automática*
El script predict_7days.py toma la última ventana de entrada del dataset procesado y genera la predicción del precio de cierre para los próximos 7 días, comenzando en la fecha más reciente (por ejemplo, desde el 14/10/2025).

Ejemplo de salida (data/processed/prediccion_7dias.csv):
Date,Predicted_Close
2025-10-14,118841.34
2025-10-15,118373.67
2025-10-16,117597.73
2025-10-17,116018.85
2025-10-18,115992.55
2025-10-19,114615.17
2025-10-20,114774.30

*Evaluación y resultados*
**Métricas empleadas:**
1- MAE (Error Absoluto Medio)
2- RMSE (Raíz del Error Cuadrático Medio)

**Resultados:**
1- Ridge Regression → mayor error (baseline).
2- Random Forest → mejora significativa sobre Ridge.
3- LightGBM → menor error global y mejor estabilidad temporal.

Error esperado: entre 3% y 5% del precio diario para horizontes de 1 a 3 días, aumentando hacia el día 7.

*Conclusiones*
El sistema desarrollado:
1- Cumple con los requerimientos del trabajo práctico.
2- Predice 7 días hacia adelante usando datos históricos reales.
3- Incorpora múltiples fuentes de información.
4- Compara y selecciona el modelo más eficiente (LightGBM multisalida).
5- Genera predicciones reproducibles con scripts automatizados.

La precisión no fue el principal objetivo del trabajo; el foco estuvo en aplicar una metodología completa, reproducible y documentada para un problema real de predicción temporal.

*Requerimientos*

Instalar las dependencias necesarias antes de ejecutar los scripts:
pip install pandas numpy scikit-learn lightgbm yfinance joblib matplotlib
