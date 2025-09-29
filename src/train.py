import os, json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

DATA_PATH = os.path.join("data", "telco-customer-churn", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
assert os.path.exists(DATA_PATH), f"No File! {DATA_PATH}"

# 1) Load
df = pd.read_csv(DATA_PATH)

# 2) Fix dtypes
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# 3) Split X, y
y = df["Churn"].map({"No": 0, "Yes": 1})
X = df.drop(columns=["Churn"])

# 4) Column types
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# 5) Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6) Train 3 models
candidates = {
    "Logistic Regression": Pipeline([("preprocessor", preprocessor),
                                     ("model", LogisticRegression(max_iter=1000))]),
    "Random Forest": Pipeline([("preprocessor", preprocessor),
                               ("model", RandomForestClassifier(n_estimators=200, random_state=42))]),
    "XGBoost": Pipeline([("preprocessor", preprocessor),
                         ("model", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42))]),
}

results = []
for name, pipe in candidates.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    results.append((name, acc, auc))
    print(f"{name:>18s} | ACC={acc:.3f} | AUC={auc:.3f}")

# 7) Pick best by AUC
best = max(results, key=lambda x: x[2])
best_name = best[0]
best_model = candidates[best_name]

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

joblib.dump(best_model, "models/churn_model.pkl")

meta = {"best_model": best_name, "features": list(X.columns),
        "numeric_cols": numeric_cols, "categorical_cols": categorical_cols}
with open("models/model_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

# 8) Export predictions on test set
proba = best_model.predict_proba(X_test)[:, 1]
pred_df = X_test.copy()
pred_df["Churn_actual"] = y_test.values
pred_df["Churn_proba"] = proba
pred_df["Churn_pred"] = (pred_df["Churn_proba"] >= 0.5).astype(int)
pred_df.to_csv("outputs/predictions.csv", index=False)

print("\nSaved models/churn_model.pkl, models/model_meta.json, outputs/predictions.csv")
