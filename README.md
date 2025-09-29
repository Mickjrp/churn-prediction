# Customer Churn Prediction

End-to-end machine learning project to predict customer churn using the Telco Customer Churn dataset.

## Project Overview
- Dataset: Telco Customer Churn (~7,000 records, 20+ features)
- Workflow:
  1. Data loading and cleaning
  2. Exploratory Data Analysis (EDA)
  3. Feature engineering (encoding, scaling, handling missing values)
  4. Model training: Logistic Regression, Random Forest, XGBoost
  5. Model evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
  6. Explainability: Feature importance, SHAP values
  7. Insights: identified drivers of churn such as short tenure, month-to-month contracts, and high monthly charges
  8. Batch inference: automated script to score new customer data

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
- `models/churn_model.pkl` → best pipeline model
- `models/model_meta.json` → metadata (features, column types)
- `outputs/predictions.csv` → predictions on test set

## Batch Inference
```powershell
python src/predict_batch.py --input data/new_customers.csv --output outputs/predict_new.csv
```

Output: `predict_new.csv` with `Churn_proba` and `Churn_pred`

## Key Insights
- Customers with month-to-month contracts have the highest churn risk.
- High monthly charges increase churn likelihood.
- Customers with shorter tenure are more likely to churn.
- Insights can guide targeted retention strategies (e.g., offering promotions for high-risk groups).

## Requirements
- pandas, numpy, matplotlib, seaborn  
- scikit-learn, xgboost  
- joblib, shap


