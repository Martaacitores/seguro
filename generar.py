import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ruta a los datos y a los modelos entrenados
csv_files = {
    "reggaeton": "data/L_elevacionreggaeton_converted.csv",
    "pop": "data/L_elevacionpop_converted.csv",
    "clasica": "data/L_rdlclasica_converted.csv"
}

model_files = {
    "reggaeton": "reggaeton_model.pkl",
    "pop": "pop_model.pkl",
    "clasica": "clasica_model.pkl"
}

# Cargar y procesar datos
def load_and_process_data(genre):
    df = pd.read_csv(csv_files[genre])
    
    # Imputar valores NaN en las características con la mediana de cada columna
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # Verificar si la variable objetivo ("A1") contiene NaN después de la imputación
    if df["A1"].isna().sum() > 0:
        # Verificar si existe algún valor en "A1" para la imputación
        if not df["A1"].mode().empty:
            df["A1"] = df["A1"].fillna(df["A1"].mode()[0])
        else:
            df["A1"] = df["A1"].fillna(0)  # Usar un valor predeterminado si no hay moda
    
    # Codificar y escalar columnas numéricas
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cols = encoder.fit_transform(df[["I1", "I2"]])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(["I1", "I2"]))
    
    df = pd.concat([df, encoded_df], axis=1).drop(["I1", "I2"], axis=1)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.drop(["A1"], axis=1))
    y = df["A1"]

    return pd.DataFrame(scaled_data, columns=df.columns.drop("A1")), y

# Generar predicciones y explicaciones
def generate_predictions_and_explanations(genre):
    # Verificar si el archivo de modelo existe
    if not os.path.exists(model_files[genre]):
        print(f"Modelo para {genre} no encontrado. Asegúrate de entrenarlo primero.")
        return
    
    # Cargar datos y modelo
    X, y = load_and_process_data(genre)
    model = joblib.load(model_files[genre])

    # Generar predicciones
    predictions = model.predict(X)
    print(f"Predicciones para {genre}:")
    print(predictions)
    
    # Generar explicaciones SHAP
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Mostrar gráficos de SHAP
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(f"SHAP Summary Plot - {genre}")
    plt.savefig(f"static/{genre}_shap_summary.png")
    plt.close()

    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP Feature Importance - {genre}")
    plt.savefig(f"static/{genre}_shap_importance.png")
    plt.close()

# Ejecutar la generación de predicciones y explicaciones para cada género
for genre in csv_files:
    generate_predictions_and_explanations(genre)
