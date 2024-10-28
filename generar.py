import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos
csv_files = {
    "reggaeton": "data/L_elevacionreggaeton_converted.csv",
    "pop": "data/L_elevacionpop_converted.csv",
    "clasica": "data/L_rdlclasica_converted.csv"
}
dataframes = {name: pd.read_csv(path) for name, path in csv_files.items()}

def generar_grafico(genre):
    if genre in dataframes:
        data = dataframes[genre]
        plt.figure()
        data.plot()  # Ejemplo de generación de gráfico
        plt.title(f"Gráfico de {genre}")
        plt.savefig(f"static/{genre}_grafico.png")
    else:
        print("Género no encontrado")
