import os, json
import pandas as pd
import joblib
import argparse

MODEL_PATH = "models/churn_model.pkl"
META_PATH  = "models/model_meta.json"

def main(input_csv, output_csv):
    assert os.path.exists(MODEL_PATH), "no model models/churn_model.pkl"
    assert os.path.exists(META_PATH), "no file models/model_meta.json"
    assert os.path.exists(input_csv), f"no input {input_csv}"

    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    features = meta["features"]
    df = pd.read_csv(input_csv)

    # fill miss col
    for col in features:
        if col not in df.columns:
            df[col] = "Unknown"

    # reorder columns
    df = df[features].copy()

    # fix Dtype
    num_cols = meta.get("numeric_cols", [])
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = df.copy()
    out["Churn_proba"] = proba
    out["Churn_pred"] = pred
    out.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="new csv path")
    parser.add_argument("--output", default="outputs/predict_new.csv", help="result file")
    args = parser.parse_args()
    main(args.input, args.output)