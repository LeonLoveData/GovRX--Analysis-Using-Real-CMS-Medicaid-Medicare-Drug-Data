# A-Data-Science-Portfolio-Project-Using-Real-CMS-Medicaid-Medicare-Drug-Data
A Data Science Portfolio Project Using Real CMS Medicaid & Medicare Drug Data
📘 Overview
GovRx Analytics Platform is an end‑to‑end data science project built using real U.S. government drug utilization and spending datasets from CMS.
It demonstrates the full lifecycle of a modern analytics system:

Data ingestion & cleaning

Feature engineering

Exploratory analysis

Machine learning

Visualization

Dashboarding

Lightweight AI assistant

The project is designed to reflect the type of work performed by Analytics & Insights, Government Pricing, and Commercial Data Science teams in the healthcare and medtech industry.

🎯 Project Goals
This project showcases the ability to:

Work with large, messy, real‑world healthcare datasets

Build reproducible data pipelines

Engineer features relevant to drug pricing, utilization, and compliance

Apply machine learning to detect anomalies and forecast spending

Build interactive dashboards for business stakeholders

Communicate insights clearly and professionally

🏛️ Datasets Used (Real CMS Data)
All datasets were downloaded directly from CMS:

Medicaid Drug Utilization (SDUD)
medicaid_drug_utilization_2022.csv

medicaid_drug_utilization_2023.csv

medicaid_drug_utilization_2024.csv

Includes:

NDC

State

Units reimbursed

Rebate amounts

Drug name

Quarter

Medicare Drug Spending
medicare_part_b.csv

medicare_part_d.csv

Includes:

Total spending

Units

Beneficiaries

ASP (Part B)

Claims (Part D)

Year

These datasets are widely used in industry for:

Government pricing analytics

Rebate forecasting

Market access strategy

Drug pricing research

🧱 Architecture
Code
```
govrx-analytics/
│
├── data/
│   ├── raw/        # CMS datasets (downloaded)
│   ├── bronze/     # cleaned + standardized
│   ├── silver/     # enriched + joined
│   └── gold/       # analytics-ready tables
│
├── src/
│   ├── ingestion/      # load + clean CMS data
│   ├── transform/      # feature engineering
│   ├── models/         # ML models
│   ├── viz/            # visualizations
│   ├── dashboard/      # Streamlit BI app
│   └── assistant/      # AI data assistant
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── reports/
│   └── figures/
│
└── README.md
```
This structure mirrors a modern data lake (bronze → silver → gold) and demonstrates engineering maturity.

🔧 Data Engineering
Bronze Layer
Standardize column names

Normalize NDC formats

Convert dates & quarters

Remove invalid rows

Deduplicate claims

Silver Layer
Join Medicaid + Medicare datasets

Compute utilization per 1,000 beneficiaries

Compute spending per unit

Compute YoY growth

Compute rebate per unit

Gold Layer
Combined risk score

Price movement index

Utilization anomaly flags

State concentration index

Inflation penalty proxy

📊 Exploratory Data Analysis
Key questions explored:

Which drugs have the highest Medicaid utilization growth?

Which states drive the most utilization?

Which NDCs show unusual spikes in claims or rebates?

Which Medicare drugs show the fastest price inflation?

How do Medicaid and Medicare trends differ for the same NDC?

Visualizations include:

Heatmaps

Trend lines

Boxplots

Scatterplots

Correlation matrices

🤖 Machine Learning
1. Medicaid Utilization Anomaly Detection
Model: Isolation Forest

Goal: Identify unusual spikes in utilization or rebate amounts

Features:

Units reimbursed

Rebate per unit

YoY growth

State concentration

2. Medicare Price Movement Forecast
Model: XGBoost Regressor

Goal: Predict next‑year spending per unit

Features:

Spending per beneficiary

Units

YoY price change

Inflation index

3. High‑Risk Drug Classification
Model: RandomForestClassifier

Goal: Label top 10% highest‑risk NDCs

Target: Combined risk score

📈 Visualizations
The project generates:

Medicaid
Utilization heatmap by state

Quarterly utilization trends

Rebate per unit distribution

Outlier scatterplots

Medicare
Price movement waterfall

Spending per beneficiary trend

Top inflation‑risk drugs

Cross‑Program
Medicaid vs Medicare utilization correlation

Combined risk score ranking

🖥️ Interactive Dashboard (Streamlit)
The dashboard includes:

1. Overview
Total Medicaid units

Total Medicare spending

Top 10 high‑risk drugs

2. Medicaid Analytics
State heatmap

Utilization trends

Anomaly detection

3. Medicare Analytics
Price movement

Spending trends

Inflation analysis

4. Cross‑Program Insights
NDC alignment

Combined risk score explorer

5. AI Assistant
A lightweight assistant that:

Answers questions about drug utilization

Explains pricing metrics

Summarizes trends

Generates SQL queries

🧪 How to Run the Project
1. Install dependencies
Code
pip install -r requirements.txt
2. Build the data lake
Code
python src/ingestion/load_cms_data.py
python src/transform/build_features.py
3. Train models
Code
python src/models/anomaly_detection.py
python src/models/price_forecasting.py
4. Generate visualizations
Code
python src/viz/plots.py
5. Launch dashboard
Code
streamlit run src/dashboard/app.py
