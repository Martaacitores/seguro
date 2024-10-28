import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, label_binarize
import numpy as np
from sklearn.utils import class_weight

# Cargar los datos originales y el dataset filtrado
file_path_original = './data/L_elevacionreggaeton_converted.csv'
file_path_filtered = './data/L_elevacionpop_converted.csv'
df_original = pd.read_csv(file_path_original)
df_filtered = pd.read_csv(file_path_filtered)

# Combinar ambos datasets
df_combined = pd.concat([df_original, df_filtered]).drop_duplicates().reset_index(drop=True)

# Codificación one-hot para columnas 'I1' y 'I2'
encoder_I1 = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder_I2 = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

I1_encoded = encoder_I1.fit_transform(df_combined[['I1']])
I2_encoded = encoder_I2.fit_transform(df_combined[['I2']])

# Agregar las columnas codificadas y eliminar las originales
I1_encoded_df = pd.DataFrame(I1_encoded, columns=encoder_I1.get_feature_names_out(['I1']))
I2_encoded_df = pd.DataFrame(I2_encoded, columns=encoder_I2.get_feature_names_out(['I2']))

df_encoded = pd.concat([df_combined, I1_encoded_df, I2_encoded_df], axis=1).drop(['I1', 'I2'], axis=1)

# Escalado de características numéricas
scaler = StandardScaler()
df_encoded[['O1', 'O2']] = scaler.fit_transform(df_encoded[['O1', 'O2']])

# Codificación de la variable objetivo
label_encoder = LabelEncoder()
df_encoded['A1'] = label_encoder.fit_transform(df_encoded['A1'])

# Variables dependientes e independientes
y = df_encoded['A1']
X = df_encoded.drop(['A1'], axis=1)

# Guardar los nombres de las características
feature_names = list(X.columns)
joblib.dump(feature_names, 'feature_names.pkl')

# División en conjunto de entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=100, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=100, stratify=y_temp)

# Balanceo de clases con SMOTE
smote = SMOTE(random_state=100)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Calcular los pesos de clase para manejar el desbalanceo
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_res), y=y_train_res)
class_weights_dict = dict(enumerate(class_weights))

# Hiperparámetros para RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Configuración y búsqueda de los mejores hiperparámetros
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=100, class_weight=class_weights_dict),
    param_distributions=param_dist,
    n_iter=50,
    cv=StratifiedKFold(n_splits=5),
    verbose=2,
    random_state=100,
    n_jobs=-1
)

# Entrenamiento con los mejores hiperparámetros
random_search.fit(X_train_res, y_train_res)
best_params = random_search.best_params_
print("Mejores hiperparámetros encontrados:", best_params)

# Entrenar modelo final con los mejores parámetros
best_model = RandomForestClassifier(random_state=100, **best_params, class_weight=class_weights_dict)
best_model.fit(X_train_res, y_train_res)

# Guardar modelo y encoders
joblib.dump(best_model, 'optimized_model.pkl')
joblib.dump(encoder_I1, 'encoder_I1.pkl')
joblib.dump(encoder_I2, 'encoder_I2.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Evaluación en conjunto de validación
y_pred_val = best_model.predict(X_val)
print("Evaluación en conjunto de validación:")
print("Accuracy:", accuracy_score(y_val, y_pred_val))
print("F1-Score:", f1_score(y_val, y_pred_val, average='weighted'))
print("Classification Report:\n", classification_report(y_val, y_pred_val))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_val))

# Evaluación en conjunto de prueba
y_pred_test = best_model.predict(X_test)
print("Evaluación en conjunto de prueba:")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("F1-Score:", f1_score(y_test, y_pred_test, average='weighted'))
print("Classification Report:\n", classification_report(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))

# Gráficos ROC y SHAP
def generate_evaluation_plots(model, X_test, y_test):
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_binarized.shape[1]
    y_proba = model.predict_proba(X_test)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title(f'ROC Curve - Class {i}')
        plt.legend(loc="lower right")
        plt.savefig(f'static/roc_curve_class_{i}.png')
        plt.close()

generate_evaluation_plots(best_model, X_test, y_test)

# Explicaciones SHAP
explainer = shap.Explainer(best_model, X_train_res)
shap_values = explainer(X_test, check_additivity=False)
joblib.dump(explainer, 'shap_explainer.pkl')
joblib.dump(shap_values, 'shap_values.pkl')
print("SHAP explanations saved successfully")
