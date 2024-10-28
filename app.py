from flask import Flask, render_template, jsonify
import pandas as pd

app = Flask(__name__)

# Ruta a los archivos CSV
csv_files = {
    "reggaeton": "data/L_elevacionreggaeton_converted.csv",
    "pop": "data/L_elevacionpop_converted.csv",
    "clasica": "data/L_rdlclasica_converted.csv"
}

# Cargar los datos al iniciar la app
dataframes = {name: pd.read_csv(path) for name, path in csv_files.items()}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data/<genre>")
def get_data(genre):
    if genre in dataframes:
        data = dataframes[genre].to_dict(orient="records")
        return jsonify(data)
    else:
        return jsonify({"error": "Género no encontrado"}), 404

if __name__ == "__main__":
    app.run(debug=True)
