# Customer Churn Prediction

End-to-end machine learning project to predict customer churn using the Telco Customer Churn dataset.

---

## Project Overview
- Dataset: Telco Customer Churn (~7,000 records, 20+ features)
- Workflow:
  - Data loading and cleaning
  - Exploratory Data Analysis (EDA)
  - Feature engineering (encoding, scaling, handling missing values)
  - Model training: Logistic Regression, Random Forest, XGBoost
  - Model evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
  - Explainability: Feature importance, SHAP values
  - Insights: identified drivers of churn such as short tenure, month-to-month contracts, and high monthly charges
  - Batch inference: automated script to score new customer data

---

## Project Structure
```
churn-prediction/
├─ data/                    # raw data (CSV)
├─ models/                  # saved pipeline & metadata 
├─ notebooks/               # EDA & experiments
├─ outputs/                 # predictions & reports
├─ src/
│ ├─ predict_batch.py       # batch inference script
│ └─ train.py               # retrain end-to-end
└─ app.py                   # Streamlit web app

```
---
## Setup
```powershell
git clone https://github.com/Mickjrp/churn-prediction.git
cd churn-prediction

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Training
```powershell
python src/train.py
```

Artifacts created:
- `models/churn_model.pkl` -> best pipeline model
- `models/model_meta.json` -> metadata (features, column types)
- `outputs/predictions.csv` -> predictions on test set

## Batch Inference
```powershell
python src/predict_batch.py --input data/new_customers.csv --output outputs/predict_new.csv
```

Output: `predict_new.csv` with `Churn_proba` and `Churn_pred`

---

## Key Insights

<img src="image/3.Churn_by_ContractType.png" alt="Churn_by_ContractType" width="400"/>

Customers on **month-to-month contracts** churn at a much higher rate compared to those on one-year or two-year contracts (as shown in Churn by Contract Type).

---



