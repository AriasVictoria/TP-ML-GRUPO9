import pandas as pd
from pathlib import Path

# Carpeta donde están todas las predicciones
PRED_DIR = Path("resultados/predictions/")

# Leer todos los CSV
all_preds = []
for file in PRED_DIR.glob("pred_rf_multioutput_*.csv"):
    df = pd.read_csv(file)
    all_preds.append(df)

# Concatenar todo en un solo DataFrame
df_all = pd.concat(all_preds, ignore_index=True)

# Ordenar por target_date y pred_date_utc
df_all['pred_date_utc'] = pd.to_datetime(df_all['pred_date_utc'])
df_all['target_date'] = pd.to_datetime(df_all['target_date'])
df_all = df_all.sort_values(['target_date', 'pred_date_utc'])

# Calcular diferencia entre predicciones consecutivas para el mismo target_date
df_all['diff_prev'] = df_all.groupby('target_date')['pred_close'].diff()

# Guardar el resultado
df_all.to_csv(PRED_DIR / "comparacion_predicciones.csv", index=False)

print("Comparación lista. Archivo guardado en 'comparacion_predicciones.csv'")
print(df_all.head(15))
