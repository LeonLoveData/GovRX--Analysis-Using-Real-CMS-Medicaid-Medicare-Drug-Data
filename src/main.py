import argparse
import sys
import pandas as pd
from pathlib import Path

from data_process import (
    run_ingestion,
    build_medicaid_features,
    build_medicare_features,
    build_gold_tables,
)

from data_analysis import (
    train_anomaly_model,
    train_price_forecast_model,
)

from data_visualization import generate_plots


TARGET_DRUG = {
    "name": "Zosyn",
    "product_name_keyword": "zosyn",
    "generic_keyword": "piperacillin",
    "ndc_labeler_prefixes": [],
}

RAW_DIR = Path("data/cms")
BRONZE_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")
GOLD_DIR = Path("data/gold")
MODEL_DIR = Path("models")
FIG_DIR = Path("reports/figures")

for d in [BRONZE_DIR, SILVER_DIR, GOLD_DIR, MODEL_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def run_dashboard():
    import streamlit as st

    st.set_page_config(page_title="GovRx Analytics", layout="wide")
    st.title("GovRx Analytics Dashboard")

    medicaid = None
    medicare = None
    risk = None

    anomaly_f = GOLD_DIR / "medicaid_ndc_year_with_anomalies.csv"
    medicare_f = GOLD_DIR / "medicare_part_d_ndc_year_gold.csv"
    risk_f = GOLD_DIR / "ndc_year_risk_gold.csv"

    if anomaly_f.exists():
        medicaid = pd.read_csv(anomaly_f)
    if medicare_f.exists():
        medicare = pd.read_csv(medicare_f)
    if risk_f.exists():
        risk = pd.read_csv(risk_f)

    st.subheader("Overview KPIs")
    col1, col2, col3 = st.columns(3)

    if medicaid is not None:
        col1.metric("Medicaid Units", f"{medicaid['total_units'].sum():,.0f}")
        col1.metric("Anomalies Flagged", f"{int(medicaid['anomaly_flag'].sum()):,}")
    else:
        col1.info("No Medicaid data")

    if medicare is not None:
        col2.metric("Medicare Part D Spending", f"${medicare['total_spending'].sum():,.0f}")
    else:
        col2.info("No Medicare data")

    if risk is not None:
        col3.metric("Avg Risk Score", f"{risk['risk_score'].mean():.3f}")
        col3.metric("Max Risk Score", f"{risk['risk_score'].max():.3f}")
    else:
        col3.info("No risk data")

    st.subheader("Medicaid Units by Year")
    if medicaid is not None:
        trend = medicaid.groupby("year").agg(total_units=("total_units", "sum"))
        st.line_chart(trend)
    else:
        st.info("No Medicaid gold data available.")

    st.subheader("Medicare Part D Spending by Year")
    if medicare is not None:
        trend = medicare.groupby("year").agg(total_spending=("total_spending", "sum"))
        st.line_chart(trend)
    else:
        st.info("No Medicare Part D gold data available.")

    st.subheader("Top 20 High-Risk NDC-Year Combinations")
    if risk is not None:
        st.dataframe(
            risk.sort_values("risk_score", ascending=False).head(20),
            use_container_width=True,
        )
    else:
        st.info("No risk data available.")


def ai_assistant():
    kb = """
    Key concepts:
    - Medicaid utilization = units_reimbursed
    - Rebate per unit    = rebate_amount / units_reimbursed
    - Medicare spending  = total_spending, total_claims, beneficiaries
    - Risk score         = normalized (utilization + spending + yoy_growth)
    - Anomaly flag       = 1 means flagged by IsolationForest
    """

    print("GovRx Assistant Ready. Type 'quit' to exit.")
    print("Commands: 'highest medicaid' | 'highest medicare' | 'risk' | 'anomalies'\n")

    while True:
        q = input("Ask a question: ").strip().lower()
        if q in ("quit", "exit", "q"):
            break

        if "highest medicaid" in q:
            f = GOLD_DIR / "medicaid_ndc_year_gold.csv"
            if f.exists():
                df = pd.read_csv(f)
                print(df.sort_values("total_units", ascending=False).head(10).to_string(index=False))
            else:
                print("Medicaid gold table not found.")

        elif "highest medicare" in q:
            f = GOLD_DIR / "medicare_part_d_ndc_year_gold.csv"
            if f.exists():
                df = pd.read_csv(f)
                print(df.sort_values("total_spending", ascending=False).head(10).to_string(index=False))
            else:
                print("Medicare Part D gold table not found.")

        elif "anomal" in q:
            f = GOLD_DIR / "medicaid_ndc_year_with_anomalies.csv"
            if f.exists():
                df = pd.read_csv(f)
                flagged = df[df["anomaly_flag"] == 1]
                print(f"Total anomalies: {len(flagged):,}")
                print(flagged.sort_values("anomaly_score").head(10).to_string(index=False))
            else:
                print("Anomaly data not found.")

        elif "risk" in q:
            f = GOLD_DIR / "ndc_year_risk_gold.csv"
            if f.exists():
                df = pd.read_csv(f)
                print(df.sort_values("risk_score", ascending=False).head(10).to_string(index=False))
            else:
                print("Risk gold table not found.")

        else:
            print(kb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GovRx Analytics Pipeline")
    parser.add_argument(
        "--mode",
        choices=["pipeline", "dashboard", "assistant"],
        default="pipeline",
    )
    args = parser.parse_args()

    if args.mode == "assistant":
        ai_assistant()

    elif args.mode == "dashboard":
        print("To launch the dashboard, run:")
        print("  streamlit run main.py -- --mode dashboard")

    else:
        print("\n=== GovRx Pipeline ===")
        print("\n[1] Ingestion (Bronze)...")
        run_ingestion(drug=TARGET_DRUG)

        print("\n[2] Feature Engineering (Silver)...")
        build_medicaid_features()
        build_medicare_features()

        print("\n[3] Gold Tables...")
        build_gold_tables()

        print("\n[4] ML Models...")
        train_anomaly_model()
        train_price_forecast_model()

        print("\n[5] Visualizations...")
        generate_plots(TARGET_DRUG)

        print("\n=== Pipeline Complete ===")
        print(f"Bronze : {BRONZE_DIR}")
        print(f"Silver : {SILVER_DIR}")
        print(f"Gold   : {GOLD_DIR}")
        print(f"Models : {MODEL_DIR}")
        print(f"Figures: {FIG_DIR}")
        print("\nTo explore results interactively:")
        print("  streamlit run main.py")
        print("  python main.py --mode assistant")


_running_via_streamlit = (
    "streamlit" in sys.modules
    and "streamlit.web.cli" in sys.modules
)

if _running_via_streamlit:
    run_dashboard()
