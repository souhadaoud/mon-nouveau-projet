from flask import Flask, request, jsonify
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

app = Flask(__name__)

# ======================
# CONFIGURATION
# ======================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.joblib")

# ======================
# CHARGEMENT DES DONNÉES
# ======================
df = pd.read_excel("Data_fournisseurs_final_churn_reel.xlsx")
df = pd.get_dummies(df, drop_first=True)
df = df.dropna()

features = [c for c in df.columns if c != "Churn"]
X = df[features]
y = df["Churn"].astype(int)

# ======================
# SCALER
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================
# DÉFINITION DES MODÈLES
# ======================
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=150, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(probability=True, kernel='rbf', random_state=42)
}

# ======================
# ENTRAÎNEMENT + MLFLOW
# ======================
mlflow.set_experiment("SupplierChurn_Models")

for name, model in models.items():
    model.fit(X_scaled, y)
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))

    with mlflow.start_run(run_name=name):
        mlflow.sklearn.log_model(model, name)
        mlflow.log_metric("accuracy", model.score(X_scaled, y))
        mlflow.log_param("model_type", name)

# Sauvegarde du scaler et des features
joblib.dump(scaler, SCALER_PATH)
joblib.dump(features, FEATURES_PATH)

# ======================
# API FLASK
# ======================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df_input = pd.DataFrame([data])

    features = joblib.load(FEATURES_PATH)
    scaler = joblib.load(SCALER_PATH)

    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=features, fill_value=0)
    df_scaled = scaler.transform(df_input)

    results = {}

    for name in models.keys():
        model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            pred = model.predict(df_scaled)[0]
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(df_scaled)[0][1]
            else:
                prob = 0.0
            results[name] = {"prediction": int(pred), "probability": float(prob)}

    return jsonify(results)


@app.route("/")
def home():
    return "✅ Supplier Churn Prediction API is running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
