import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
import joblib

GOLD_DIR = Path("data/gold")
MODEL_DIR = Path("models")

def train_anomaly_model():
    f = GOLD_DIR / "medicaid_ndc_year_gold.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    candidate_features = [
        "total_units",
        "total_prescriptions",
        "total_amount_reimbursed",
        "medicaid_amount_reimbursed",
        "avg_amount_per_unit",
        "avg_amount_per_rx",
        "avg_units_yoy_growth",
    ]
    features = [c for c in candidate_features if c in df.columns]
    if not features:
        return
    X = df[features].fillna(0)
    model = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    model.fit(X)
    df["anomaly_score"] = model.decision_function(X)
    df["anomaly_flag"] = pd.Series(model.predict(X)).map({1: 0, -1: 1}).values
    df.to_csv(GOLD_DIR / "medicaid_ndc_year_with_anomalies.csv", index=False)
    joblib.dump(model, MODEL_DIR / "medicaid_isolation_forest.pkl")

def train_price_forecast_model():
    f = GOLD_DIR / "medicare_part_d_ndc_year_gold.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    drug_key = next((c for c in ["brand_name", "generic_name"] if c in df.columns), None)
    if not drug_key:
        return
    df = df.sort_values([drug_key, "year"]).reset_index(drop=True)
    target_col = next((c for c in ["avg_spend_per_unit", "avg_spend_per_claim"] if c in df.columns), None)
    if not target_col:
        return
    df["next_year_spend"] = df.groupby(drug_key)[target_col].shift(-1)
    latest_year = int(df["year"].max())
    forecast_year = latest_year + 1
    train_df = df[df["next_year_spend"].notna()].copy()
    predict_df = df[df["year"] == latest_year].copy()
    if len(train_df) < 2:
        return
    candidate_features = [
        "total_spending",
        "total_dosage_units",
        "total_claims",
        "total_beneficiaries",
        "avg_spend_per_unit",
        "avg_spend_per_claim",
        "avg_spend_per_bene",
        "avg_spending_yoy_growth",
    ]
    features = [c for c in candidate_features if c in df.columns]
    X_train = train_df[features].fillna(0)
    y_train = train_df["next_year_spend"]
    if len(train_df) >= 6:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        with np.warnings.catch_warnings():
            np.warnings.simplefilter("ignore")
            cross_val_score(model, X_train, y_train, cv=LeaveOneOut(), scoring="r2")
        model.fit(X_train, y_train)
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR / "medicare_price_forecast_rf.pkl")
    X_predict = predict_df[features].fillna(0)
    pred = model.predict(X_predict)
    predict_df = predict_df.copy()
    predict_df["base_year"] = latest_year
    predict_df["forecast_year"] = forecast_year
    predict_df["predicted_spend_per_unit"] = pred
    predict_df["pct_change_vs_base"] = (
        (pred - predict_df[target_col].values)
        / predict_df[target_col].replace(0, np.nan).values
        * 100
    )
    out_cols = [
        "base_year",
        "forecast_year",
        drug_key,
        target_col,
        "predicted_spend_per_unit",
        "pct_change_vs_base",
    ]
    out_cols = [c for c in out_cols if c in predict_df.columns]
    predict_df[out_cols].to_csv(GOLD_DIR / "medicare_part_d_forecast.csv", index=False)
