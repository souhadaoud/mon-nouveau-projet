from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

app = Flask(__name__)

# Répertoire des modèles
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

# Charger ou entraîner les modèles
if os.path.exists(RF_MODEL_PATH) and os.path.exists(XGB_MODEL_PATH) and os.path.exists(SCALER_PATH):
rf = joblib.load(RF_MODEL_PATH)
xgb = joblib.load(XGB_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_cols = joblib.load(os.path.join(MODEL_DIR, "features.joblib"))
else:
df = pd.read_excel("Data_fournisseurs_final_churn_reel.xlsx")
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)

features = [c for c in df.columns if c != "Churn"]
X = df[features]
y = df["Churn"].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_scaled, y)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=300, random_state=42)
xgb.fit(X_scaled, y)

joblib.dump(rf, RF_MODEL_PATH)
joblib.dump(xgb, XGB_MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(features, os.path.join(MODEL_DIR, "features.joblib"))

@app.route("/predict", methods=["POST"])
def predict():
data = request.json
df = pd.DataFrame([data])
df = pd.get_dummies(df)
df = df.reindex(columns=feature_cols, fill_value=0)
df_scaled = scaler.transform(df)

pred_rf = rf.predict(df_scaled)[0]
pred_xgb = xgb.predict(df_scaled)[0]
prob_rf = rf.predict_proba(df_scaled)[0][1]
prob_xgb = xgb.predict_proba(df_scaled)[0][1]

return jsonify({
"pred_RF": int(pred_rf),
"pred_XGB": int(pred_xgb),
"prob_RF": float(prob_rf),
"prob_XGB": float(prob_xgb)
})

@app.route("/")
def home():
return "✅ API Churn Model is running!"

if __name__ == "__main__":
app.run(host="0.0.0.0", port=10000)


