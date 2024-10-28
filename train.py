import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Cargar los datos
csv_files = {
    "reggaeton": "data/L_elevacionreggaeton_converted.csv",
    "pop": "data/L_elevacionpop_converted.csv",
    "clasica": "data/L_rdlclasica_converted.csv"
}
dataframes = {name: pd.read_csv(path) for name, path in csv_files.items()}

def entrenar_modelo(genre):
    if genre in dataframes:
        data = dataframes[genre]
        X = data[["I1", "I2", "O1", "O2"]]  # Variables de entrada (ejemplo)
        y = data["A1"]  # Variable objetivo (ejemplo)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"Precisión del modelo para {genre}: {score}")
    else:
        print("Género no encontrado")
