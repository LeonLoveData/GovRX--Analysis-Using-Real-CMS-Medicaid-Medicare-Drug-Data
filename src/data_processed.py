import pandas as pd
import numpy as np
from pathlib import Path
import re

RAW_DIR = Path("data/cms")
BRONZE_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")
GOLD_DIR = Path("data/gold")

def load_medicaid(drug=None):
    files = sorted(RAW_DIR.glob("medicaid_drug_utilization_*.csv"))
    if not files:
        return
    EXACT_COL_MAP = {
        "ndc": "ndc11",
        "state": "state",
        "year": "year",
        "quarter": "quarter",
        "product_name": "drug_name",
        "units_reimbursed": "units_reimbursed",
        "number_of_prescriptions": "num_prescriptions",
        "total_amount_reimbursed": "total_amount_reimbursed",
        "medicaid_amount_reimbursed": "medicaid_amount_reimbursed",
        "non_medicaid_amount_reimbursed": "non_medicaid_amount_reimbursed",
        "utilization_type": "utilization_type",
        "labeler_code": "labeler_code",
        "product_code": "product_code",
        "package_size": "package_size",
        "suppression_used": "suppression_used",
    }
    NUM_COLS = [
        "units_reimbursed",
        "num_prescriptions",
        "total_amount_reimbursed",
        "medicaid_amount_reimbursed",
        "non_medicaid_amount_reimbursed",
    ]
    CHUNK_SIZE = 50000
    dfs = []
    for f in files:
        chunks = []
        for chunk in pd.read_csv(f, dtype=str, chunksize=CHUNK_SIZE):
            chunk.columns = [c.strip().lower().replace(" ", "_") for c in chunk.columns]
            if drug:
                name_col = "product_name"
                ndc_col = "ndc"
                if name_col in chunk.columns:
                    name_lower = chunk[name_col].str.lower().fillna("")
                    mask = (
                        name_lower.str.contains(drug["product_name_keyword"], na=False)
                        | name_lower.str.contains(drug["generic_keyword"], na=False)
                    )
                else:
                    mask = pd.Series([True] * len(chunk))
                if drug["ndc_labeler_prefixes"] and ndc_col in chunk.columns:
                    ndc_mask = chunk[ndc_col].str[:4].isin(drug["ndc_labeler_prefixes"])
                    mask = mask & ndc_mask
                chunk = chunk[mask]
            if chunk.empty:
                continue
            chunk = chunk.rename(columns=EXACT_COL_MAP)
            keep = [v for v in EXACT_COL_MAP.values() if v in chunk.columns]
            chunk = chunk[keep].copy()
            for col in NUM_COLS:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            chunks.append(chunk)
        if chunks:
            dfs.append(pd.concat(chunks, ignore_index=True))
    if not dfs:
        return
    medicaid = pd.concat(dfs, ignore_index=True)
    month_map = {1: 1, 2: 4, 3: 7, 4: 10}
    if "year" in medicaid.columns and "quarter" in medicaid.columns:
        medicaid["year"] = pd.to_numeric(medicaid["year"], errors="coerce").astype("Int64")
        medicaid["quarter"] = pd.to_numeric(medicaid["quarter"], errors="coerce").astype("Int64")
        medicaid["quarter_begin_date"] = pd.to_datetime(
            medicaid.apply(
                lambda r: f"{r['year']}-{month_map.get(r['quarter'], 1):02d}-01"
                if pd.notna(r["year"]) and pd.notna(r["quarter"])
                else None,
                axis=1,
            ),
            errors="coerce",
        )
    medicaid.to_csv(BRONZE_DIR / "medicaid_bronze.csv", index=False)

def load_medicare_part_b():
    f = RAW_DIR / "medicare_part_b.csv"
    if not f.exists():
        return
    df = pd.read_csv(f, dtype=str)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    col_map = {}
    for col in df.columns:
        if "hcpcs" in col:
            col_map[col] = "hcpcs_code"
        elif col == "year":
            col_map[col] = "year"
        elif "spend" in col:
            col_map[col] = "total_spending"
        elif "unit" in col:
            col_map[col] = "total_units"
        elif "benef" in col:
            col_map[col] = "beneficiaries"
    df = df.rename(columns=col_map)
    keep = [c for c in ["hcpcs_code", "year", "total_spending", "total_units", "beneficiaries"] if c in df.columns]
    df = df[keep].copy()
    for col in ["total_spending", "total_units", "beneficiaries"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    df.to_csv(BRONZE_DIR / "medicare_part_b_bronze.csv", index=False)

def load_medicare_part_d(drug=None):
    f = RAW_DIR / "medicare_part_d.csv"
    if not f.exists():
        return
    df = pd.read_csv(f, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if drug:
        brand_col = next((c for c in df.columns if c.lower() == "brnd_name"), None)
        generic_col = next((c for c in df.columns if c.lower() == "gnrc_name"), None)
        mask = pd.Series([False] * len(df))
        if brand_col:
            mask |= df[brand_col].str.lower().str.contains(drug["product_name_keyword"], na=False)
        if generic_col:
            mask |= df[generic_col].str.lower().str.contains(drug["generic_keyword"], na=False)
        df = df[mask].copy()
        if df.empty:
            return
    id_cols = []
    for c in df.columns:
        cl = c.lower()
        if cl in ("brnd_name", "gnrc_name", "tot_mftr", "mftr_name"):
            id_cols.append(c)
    years = sorted(
        set(
            int(m.group(1))
            for c in df.columns
            for m in [re.search(r"_(\d{4})$", c)]
            if m
        )
    )
    year_dfs = []
    for yr in years:
        yr_cols = {c: c for c in df.columns if c.endswith(f"_{yr}")}
        if not yr_cols:
            continue
        yr_df = df[id_cols + list(yr_cols.keys())].copy()
        rename = {c: re.sub(rf"_{yr}$", "", c).lower() for c in yr_cols}
        yr_df = yr_df.rename(columns=rename)
        yr_df["year"] = yr
        yr_df.columns = [c.lower() for c in yr_df.columns]
        year_dfs.append(yr_df)
    if not year_dfs:
        return
    long_df = pd.concat(year_dfs, ignore_index=True)
    COL_MAP = {
        "brnd_name": "brand_name",
        "gnrc_name": "generic_name",
        "tot_mftr": "total_manufacturers",
        "mftr_name": "manufacturer",
        "tot_spndng": "total_spending",
        "tot_dsg_unts": "total_dosage_units",
        "tot_clms": "total_claims",
        "tot_benes": "total_beneficiaries",
        "avg_spnd_per_dsg_unt_wghtd": "avg_spend_per_unit",
        "avg_spnd_per_clm": "avg_spend_per_claim",
        "avg_spnd_per_bene": "avg_spend_per_bene",
        "outlier_flag": "outlier_flag",
    }
    long_df = long_df.rename(columns=COL_MAP)
    NUM_COLS = [
        "total_spending",
        "total_dosage_units",
        "total_claims",
        "total_beneficiaries",
        "avg_spend_per_unit",
        "avg_spend_per_claim",
        "avg_spend_per_bene",
    ]
    for col in NUM_COLS:
        if col in long_df.columns:
            long_df[col] = pd.to_numeric(long_df[col], errors="coerce")
    long_df["year"] = long_df["year"].astype(int)
    long_df.to_csv(BRONZE_DIR / "medicare_part_d_bronze.csv", index=False)

def run_ingestion(drug=None):
    load_medicaid(drug=drug)
    load_medicare_part_b()
    load_medicare_part_d(drug=drug)

def build_medicaid_features():
    f = BRONZE_DIR / "medicaid_bronze.csv"
    if not f.exists():
        return
    df = pd.read_csv(f, parse_dates=["quarter_begin_date"])
    if "year" not in df.columns or df["year"].isna().all():
        df["year"] = df["quarter_begin_date"].dt.year
    if "quarter" not in df.columns or df["quarter"].isna().all():
        df["quarter"] = df["quarter_begin_date"].dt.quarter
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce")
    amount_col = next(
        (c for c in ["total_amount_reimbursed", "medicaid_amount_reimbursed"] if c in df.columns),
        None,
    )
    agg_dict = {
        "units_reimbursed": ("units_reimbursed", "sum"),
        "num_prescriptions": ("num_prescriptions", "sum"),
    }
    if amount_col:
        agg_dict["total_amount_reimbursed"] = (amount_col, "sum")
    if "medicaid_amount_reimbursed" in df.columns:
        agg_dict["medicaid_amount_reimbursed"] = ("medicaid_amount_reimbursed", "sum")
    if "non_medicaid_amount_reimbursed" in df.columns:
        agg_dict["non_medicaid_amount_reimbursed"] = ("non_medicaid_amount_reimbursed", "sum")
    agg = df.groupby(["ndc11", "state", "year", "quarter"]).agg(**agg_dict).reset_index()
    if "total_amount_reimbursed" in agg.columns:
        agg["amount_per_unit"] = agg["total_amount_reimbursed"] / agg["units_reimbursed"].replace(0, np.nan)
    if "num_prescriptions" in agg.columns and "total_amount_reimbursed" in agg.columns:
        agg["amount_per_rx"] = agg["total_amount_reimbursed"] / agg["num_prescriptions"].replace(0, np.nan)
    agg = agg.sort_values(["ndc11", "state", "year", "quarter"])
    agg["prev_units"] = agg.groupby(["ndc11", "state"])["units_reimbursed"].shift(4)
    agg["units_yoy_growth"] = (agg["units_reimbursed"] - agg["prev_units"]) / agg["prev_units"].replace(0, np.nan)
    agg.to_csv(SILVER_DIR / "medicaid_features.csv", index=False)

def build_medicare_features():
    bfile = BRONZE_DIR / "medicare_part_b_bronze.csv"
    if bfile.exists():
        df = pd.read_csv(bfile)
        df["spend_per_unit"] = df["total_spending"] / df["total_units"].replace(0, np.nan)
        df["spend_per_beneficiary"] = df["total_spending"] / df["beneficiaries"].replace(0, np.nan)
        df.to_csv(SILVER_DIR / "medicare_part_b_features.csv", index=False)
    dfile = BRONZE_DIR / "medicare_part_d_bronze.csv"
    if dfile.exists():
        df = pd.read_csv(dfile)
        if "total_spending" in df.columns and "total_dosage_units" in df.columns:
            df["spend_per_unit_calc"] = df["total_spending"] / df["total_dosage_units"].replace(0, np.nan)
        if "total_spending" in df.columns and "total_claims" in df.columns:
            df["spend_per_claim_calc"] = df["total_spending"] / df["total_claims"].replace(0, np.nan)
        if "total_spending" in df.columns and "total_beneficiaries" in df.columns:
            df["spend_per_bene_calc"] = df["total_spending"] / df["total_beneficiaries"].replace(0, np.nan)
        group_col = next((c for c in ["brand_name", "generic_name"] if c in df.columns), None)
        if group_col and "year" in df.columns:
            df = df.sort_values([group_col, "year"])
            df["prev_spending"] = df.groupby(group_col)["total_spending"].shift(1)
            df["spending_yoy_growth"] = (df["total_spending"] - df["prev_spending"]) / df["prev_spending"].replace(0, np.nan)
        df.to_csv(SILVER_DIR / "medicare_part_d_features.csv", index=False)

def build_gold_tables():
    medicaid_f = SILVER_DIR / "medicaid_features.csv"
    part_d_f = SILVER_DIR / "medicare_part_d_features.csv"
    if medicaid_f.exists():
        df = pd.read_csv(medicaid_f)
        agg_dict = {
            "total_units": ("units_reimbursed", "sum"),
            "total_prescriptions": ("num_prescriptions", "sum"),
            "avg_units_yoy_growth": ("units_yoy_growth", "mean"),
        }
        if "total_amount_reimbursed" in df.columns:
            agg_dict["total_amount_reimbursed"] = ("total_amount_reimbursed", "sum")
        if "medicaid_amount_reimbursed" in df.columns:
            agg_dict["medicaid_amount_reimbursed"] = ("medicaid_amount_reimbursed", "sum")
        if "amount_per_unit" in df.columns:
            agg_dict["avg_amount_per_unit"] = ("amount_per_unit", "mean")
        if "amount_per_rx" in df.columns:
            agg_dict["avg_amount_per_rx"] = ("amount_per_rx", "mean")
        medicaid_year = df.groupby(["ndc11", "year"]).agg(**agg_dict).reset_index()
        medicaid_year.to_csv(GOLD_DIR / "medicaid_ndc_year_gold.csv", index=False)
    if part_d_f.exists():
        df = pd.read_csv(part_d_f)
        group_col = next((c for c in ["brand_name", "generic_name"] if c in df.columns), None)
        if group_col:
            agg_dict = {"total_spending": ("total_spending", "sum")}
            for col, agg_name in [
                ("total_dosage_units", "total_dosage_units"),
                ("total_claims", "total_claims"),
                ("total_beneficiaries", "total_beneficiaries"),
                ("avg_spend_per_unit", "avg_spend_per_unit"),
                ("avg_spend_per_claim", "avg_spend_per_claim"),
                ("avg_spend_per_bene", "avg_spend_per_bene"),
                ("spending_yoy_growth", "avg_spending_yoy_growth"),
            ]:
                if col in df.columns:
                    agg_dict[agg_name] = (col, "mean" if "avg" in col or "growth" in col else "sum")
            medicare_d_year = df.groupby([group_col, "year"]).agg(**agg_dict).reset_index()
            medicare_d_year.to_csv(GOLD_DIR / "medicare_part_d_ndc_year_gold.csv", index=False)
    med_gold = GOLD_DIR / "medicaid_ndc_year_gold.csv"
    mdd_gold = GOLD_DIR / "medicare_part_d_ndc_year_gold.csv"
    if med_gold.exists() and mdd_gold.exists():
        m1 = pd.read_csv(med_gold)
        m2 = pd.read_csv(mdd_gold)
        m1["medicaid_risk_score"] = (
            m1["total_units"] / (m1["total_units"].max() + 1e-6)
            + m1["total_amount_reimbursed"].fillna(0) / (m1["total_amount_reimbursed"].fillna(0).max() + 1e-6)
            + m1["avg_units_yoy_growth"].fillna(0).clip(-1, 1)
        )
        m1.to_csv(GOLD_DIR / "medicaid_risk_gold.csv", index=False)
        spend_col = "total_spending" if "total_spending" in m2.columns else None
        growth_col = "avg_spending_yoy_growth" if "avg_spending_yoy_growth" in m2.columns else None
        if spend_col:
            m2["medicare_risk_score"] = (
                m2[spend_col] / (m2[spend_col].max() + 1e-6)
                + (m2[growth_col].fillna(0).clip(-1, 1) if growth_col else 0)
            )
        m2.to_csv(GOLD_DIR / "medicare_part_d_risk_gold.csv", index=False)
