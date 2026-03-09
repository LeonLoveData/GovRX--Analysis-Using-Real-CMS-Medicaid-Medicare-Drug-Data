import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

GOLD_DIR = Path("data/gold")
FIG_DIR = Path("reports/figures")

def generate_plots(TARGET_DRUG):
    drug_name = TARGET_DRUG["name"]

    f = GOLD_DIR / "medicaid_ndc_year_gold.csv"
    if f.exists():
        df = pd.read_csv(f)
        trend = df.groupby("year").agg(total_units=("total_units", "sum")).reset_index()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(trend["year"], trend["total_units"], marker="o", color="#2196F3", linewidth=2, markersize=7)
        for _, row in trend.iterrows():
            ax.annotate(
                f"{row['total_units']:,.0f}",
                xy=(row["year"], row["total_units"]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="#1565C0",
            )
        ax.set_title(f"{drug_name} — Medicaid Annual Units Reimbursed (Actual)", fontsize=13)
        ax.set_xlabel("Year")
        ax.set_ylabel("Units Reimbursed")
        ax.set_xticks(trend["year"])
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        out = FIG_DIR / "medicaid_units_by_year.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)

    gold_f = GOLD_DIR / "medicare_part_d_ndc_year_gold.csv"
    forecast_f = GOLD_DIR / "medicare_part_d_forecast.csv"

    if gold_f.exists():
        gold = pd.read_csv(gold_f)
        drug_key = next((c for c in ["brand_name", "generic_name"] if c in gold.columns), None)
        spend_col = next((c for c in ["avg_spend_per_unit", "avg_spend_per_claim"] if c in gold.columns), None)
        if drug_key and spend_col:
            real = gold[[drug_key, "year", spend_col]].copy()
            real = real.sort_values("year").reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(real["year"], real[spend_col], marker="o", color="#2196F3", linewidth=2.5, markersize=9)
            for _, row in real.iterrows():
                ax.annotate(
                    f"{int(row['year'])}\n${row[spend_col]:,.2f}",
                    xy=(row["year"], row[spend_col]),
                    xytext=(0, 14),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color="#1565C0",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#90CAF9", alpha=0.8),
                )
            if forecast_f.exists():
                fc = pd.read_csv(forecast_f)
                if "predicted_spend_per_unit" in fc.columns and "forecast_year" in fc.columns:
                    fc_year = int(fc["forecast_year"].iloc[0])
                    fc_value = float(fc["predicted_spend_per_unit"].iloc[0])
                    base_year = int(fc["base_year"].iloc[0])
                    base_value = float(real[real["year"] == base_year][spend_col].iloc[0])
                    pct = (fc_value - base_value) / base_value * 100
                    ax.plot([base_year, fc_year], [base_value, fc_value], linestyle="--", color="#FF7043", linewidth=2, alpha=0.8)
                    ax.scatter([fc_year], [fc_value], marker="*", s=220, color="#FF5722")
                    arrow_dir = 14 if fc_value >= base_value else -22
                    ax.annotate(
                        f"{fc_year}\n${fc_value:,.2f}\n({pct:+.1f}%)",
                        xy=(fc_year, fc_value),
                        xytext=(0, arrow_dir),
                        textcoords="offset points",
                        ha="center",
                        fontsize=9,
                        color="#BF360C",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#FF7043", alpha=0.9),
                    )
                    all_years = list(real["year"]) + [fc_year]
                    ax.axvspan(base_year - 0.3, fc_year + 0.3, alpha=0.06, color="#FF7043")
            ax.set_title(f"{drug_name} — Medicare Part D Avg Spend per Unit", fontsize=13)
            ax.set_xlabel("Year")
            ax.set_ylabel("Avg Spend per Unit ($)")
            all_x = list(real["year"])
            if forecast_f.exists():
                all_x.append(fc_year)
            ax.set_xticks(all_x)
            ax.set_xticklabels([str(int(y)) for y in all_x])
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            out = FIG_DIR / "medicare_part_d_forecast_plot.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)

    if gold_f.exists():
        gold = pd.read_csv(gold_f)
        if "total_spending" in gold.columns:
            trend = gold.groupby("year").agg(total_spending=("total_spending", "sum")).reset_index()
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(trend["year"].astype(str), trend["total_spending"], color="#42A5F5", edgecolor="white", width=0.5)
            for _, row in trend.iterrows():
                ax.text(str(int(row["year"])), row["total_spending"], f"${row['total_spending']:,.0f}", ha="center", va="bottom", fontsize=9, color="#0D47A1")
            ax.set_title(f"{drug_name} — Medicare Part D Annual Total Spending", fontsize=13)
            ax.set_xlabel("Year")
            ax.set_ylabel("Total Spending ($)")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            out = FIG_DIR / "medicare_spending_by_year.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)

    f = GOLD_DIR / "medicaid_ndc_year_with_anomalies.csv"
    if f.exists():
        df = pd.read_csv(f)
        fig, ax = plt.subplots(figsize=(9, 5))
        normal = df[df["anomaly_flag"] == 0]["anomaly_score"]
        flagged = df[df["anomaly_flag"] == 1]["anomaly_score"]
        ax.hist(normal, bins=30, color="#42A5F5", alpha=0.7)
        ax.hist(flagged, bins=10, color="#EF5350", alpha=0.9)
        ax.set_title(f"{drug_name} — Medicaid Anomaly Score Distribution", fontsize=13)
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        out = FIG_DIR / "anomaly_score_distribution.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
