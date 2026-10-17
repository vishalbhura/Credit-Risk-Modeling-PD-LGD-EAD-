"""
dashboard/app.py
================
Streamlit Credit Risk Dashboard — Enterprise Credit Risk Modeling & Validation Framework

4 tabs:
    1. Loan Scoring     — Input form · PD · risk tier · credit score · scorecard waterfall
    2. Expected Loss    — EL = PD × LGD × EAD · portfolio table · EL by risk class
    3. Model Performance — Model comparison · ROC · calibration · decile chart
    4. Explainability   — SHAP global importance · per-loan waterfall · dependence plots

Run from project root:
    streamlit run dashboard/app.py
"""

import os, sys, json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import joblib

# Ensure project root is on path so src/ can be imported
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.preprocessing import apply_woe_encoding

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Model and artifact loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_lr_model():
    with open("models/pd/lr_params.json") as f:
        params = json.load(f)
    woe_results = {}
    for feat, meta in params["woe_results"].items():
        df = pd.DataFrame(meta["records"])
        for c in ["n_good", "n_obs", "n_bad"]:
            if c in df.columns:
                df[c] = df[c].astype("int32")
        for c in ["bad_rate", "pct_good", "pct_bad", "woe", "iv"]:
            if c in df.columns:
                df[c] = df[c].astype(float)
        woe_results[feat] = (df, float(meta.get("iv", 0)), meta.get("bin_edges"))
    return params, woe_results


@st.cache_data
def load_artifacts():
    portfolio = pd.read_csv("artifacts/expected_loss_portfolio.csv")
    shap_imp  = pd.read_csv("artifacts/shap_feature_importance.csv")
    try:
        model_comp = pd.read_csv("artifacts/validation_summary.csv")
    except Exception:
        model_comp = pd.DataFrame()
    return portfolio, shap_imp, model_comp


lr_params, woe_results = load_lr_model()
portfolio_df, shap_df, model_comp_df = load_artifacts()

FINAL_WOE_COLS = lr_params["final_woe_cols"]
LR_CONST       = float(lr_params["const"])
LR_COEFS       = np.array(lr_params["coef_values"], dtype=float)
LR_COEF_NAMES  = lr_params["coef_names"]
OFFSET         = float(lr_params["offset"])
FACTOR         = float(lr_params["factor"])

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
RISK_COLORS = {
    "AA": "#1a9641", "A": "#78c679", "AB": "#d9ef8b", "BB": "#fee08b",
    "B":  "#fdae61", "BC": "#f46d43", "C": "#d73027", "CD": "#a50026",
    "DD": "#7a0023", "F": "#4a0014",
}
GRADES    = list("ABCDEFG")
SUBGRADES = [f"{g}{i}" for g in GRADES for i in range(1, 6)]
LGD_MEAN  = 0.9495   # portfolio mean conditional LGD
CCF_MEAN  = 0.5957   # portfolio mean CCF

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def _make_woe_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for col in FINAL_WOE_COLS:
        feat = col.replace("_woe", "")
        if feat in woe_results:
            woe_df, _, edges = woe_results[feat]
            out[col] = apply_woe_encoding(df, feat, woe_df, bin_edges=edges)
    return pd.DataFrame(out, index=df.index)


def score_application(df: pd.DataFrame):
    X_woe    = _make_woe_matrix(df)
    woe_row  = X_woe[LR_COEF_NAMES].values.astype(float)
    log_odds = LR_CONST + float(woe_row[0] @ LR_COEFS)   # [0] extracts the single row
    prob_good = float(1 / (1 + np.exp(-log_odds)))
    score    = float(np.clip(OFFSET + FACTOR * log_odds, 300, 850))
    contribs = dict(zip(
        [n.replace("_woe", "") for n in LR_COEF_NAMES],
        FACTOR * LR_COEFS * woe_row[0],
    ))
    return prob_good, score, contribs


def pd_to_risk_class(pd_prob: float) -> str:
    for t, cls in [(0.05,"AA"),(0.08,"A"),(0.11,"AB"),(0.14,"BB"),
                   (0.17,"B"),(0.20,"BC"),(0.23,"C"),(0.27,"CD"),(0.33,"DD")]:
        if pd_prob < t:
            return cls
    return "F"


def approval_decision(rc, pd_prob, int_rate, loan_amnt, el_dollars, term):
    if rc in ("AA", "A"):
        return True, "Auto-approved (AA / A)"
    if rc == "F":
        return False, "Auto-declined (F — highest risk)"
    roi = (int_rate/100 * loan_amnt - el_dollars - 0.02*loan_amnt) / loan_amnt / (term/12)
    if roi > 0.0215:
        return True, f"Approved — annualised ROI {roi:.2%} > 2.15% hurdle"
    return False, f"Declined — annualised ROI {roi:.2%} < 2.15% hurdle"


def show_chart(path: str, caption: str = ""):
    if os.path.exists(path):
        st.image(path, caption=caption, width='stretch')
    else:
        st.caption(f"*Chart not yet generated: `{path}`*")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Loan Input Form
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Loan Application")

    sub_grade      = st.selectbox("Sub-grade", SUBGRADES, index=14,
                                   help="A1 = best credit, G5 = highest risk")
    grade          = sub_grade[0]
    loan_amnt      = st.slider("Loan Amount ($)", 1_000, 40_000, 15_000, 500)
    int_rate       = st.slider("Interest Rate (%)", 5.0, 30.99, 13.0, 0.25)
    term           = st.selectbox("Term (months)", [36, 60])
    annual_inc     = st.slider("Annual Income ($)", 15_000, 300_000, 65_000, 1_000)
    dti            = st.slider("DTI Ratio", 0.0, 60.0, 18.0, 0.5)
    fico           = st.slider("FICO Score", 600, 850, 700, 5)
    home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    purpose        = st.selectbox("Loan Purpose", [
        "debt_consolidation", "credit_card", "home_improvement",
        "other", "major_purchase", "medical", "small_business", "car",
    ])
    inq_6mths      = st.slider("Inquiries (last 6 months)", 0, 8, 1)
    acc_open_24    = st.slider("Accounts opened (24 months)", 0, 20, 4)
    verification   = st.selectbox("Income Verification",
                                   ["Not Verified", "Source Verified", "Verified"])
    st.markdown("---")
    st.caption("`streamlit run dashboard/app.py`")

# ─────────────────────────────────────────────────────────────────────────────
# Build input DataFrame and compute score
# ─────────────────────────────────────────────────────────────────────────────
installment = loan_amnt * (int_rate/100/12) / (1 - (1 + int_rate/100/12)**(-term))

input_df = pd.DataFrame([{
    "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate,
    "installment": installment, "grade": grade, "sub_grade": sub_grade,
    "emp_length": 5, "annual_inc": annual_inc,
    "verification_status": verification, "home_ownership": home_ownership,
    "purpose": purpose, "addr_state": "CA", "dti": dti,
    "delinq_2yrs": 0, "fico_range_low": fico - 4, "fico_range_high": fico,
    "inq_last_6mths": inq_6mths, "mths_since_last_delinq": 999,
    "open_acc": 11, "pub_rec": 0, "revol_bal": 8_000, "revol_util": 45.0,
    "total_acc": 22, "mths_since_earliest_cr_line": 150, "acc_now_delinq": 0,
    "initial_list_status": "w", "pub_rec_bankruptcies": 0, "tax_liens": 0,
    "collections_12_mths_ex_med": 0, "tot_cur_bal": 45_000,
    "total_rev_hi_lim": 20_000, "tot_hi_cred_lim": 55_000,
    "total_bal_ex_mort": 38_000, "tot_coll_amt": 0, "bc_util": 50.0,
    "num_actv_rev_tl": 4, "num_il_tl": 8, "num_sats": 12, "mort_acc": 0,
    "acc_open_past_24mths": acc_open_24, "mo_sin_rcnt_tl": 6,
    "mths_since_recent_inq": 4, "mths_since_recent_bc": 12, "pct_tl_nvr_dlq": 90.0,
}])

prob_good, credit_score, contribs = score_application(input_df)
pd_prob    = 1.0 - prob_good
risk_class = pd_to_risk_class(pd_prob)
el_rate    = pd_prob * LGD_MEAN * CCF_MEAN
el_dollars = el_rate * loan_amnt
approved, appr_reason = approval_decision(
    risk_class, pd_prob, int_rate, loan_amnt, el_dollars, term
)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Loan Scoring",
    "💰 Expected Loss",
    "📊 Model Performance",
    "🔍 Explainability",
])

# ── TAB 1 — Loan Scoring ──────────────────────────────────────────────────────
with tab1:
    st.title("Loan Scoring")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Credit Score",  f"{credit_score:.0f}", f"{credit_score-645:+.0f} vs baseline")
    m2.metric("P(Default)",    f"{pd_prob:.1%}")
    m3.metric("Risk Class",    risk_class)
    m4.metric("EL Estimate",   f"${el_dollars:,.0f}", f"{el_rate:.2%} of loan amount")

    if approved:
        st.success(f"✅  **APPROVED** — {appr_reason}  |  Score: {credit_score:.0f}  |  Class: {risk_class}")
    else:
        st.error(f"❌  **DECLINED** — {appr_reason}  |  Score: {credit_score:.0f}  |  Class: {risk_class}")

    st.markdown("---")
    st.subheader("Scorecard Contribution by Feature")
    st.caption("Each bar shows how many credit score points a feature adds (🟢) or subtracts (🔴) from the base score.")

    contrib_s = pd.Series(contribs).sort_values()
    fig, ax   = plt.subplots(figsize=(10, 5))
    ax.barh(contrib_s.index, contrib_s.values,
            color=["#2ca02c" if v > 0 else "#d62728" for v in contrib_s])
    ax.axvline(0, color="black", lw=0.8)
    base_sc = OFFSET + FACTOR * LR_CONST
    ax.set_xlabel("Score Points")
    ax.set_title(f"LR Scorecard  |  Base score: {base_sc:.0f}  →  Total: {credit_score:.0f}")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── TAB 2 — Expected Loss ─────────────────────────────────────────────────────
with tab2:
    st.title("Expected Loss — EL = PD × LGD × EAD")

    st.subheader("This Application")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PD",  f"{pd_prob:.2%}")
    c2.metric("LGD", f"{LGD_MEAN:.2%}", help="Portfolio mean conditional LGD")
    c3.metric("EAD", f"${loan_amnt * CCF_MEAN:,.0f}", help="Loan amount × mean CCF")
    c4.metric("EL",  f"${el_dollars:,.0f}", f"{el_rate:.2%} of funded amount")

    st.info("LGD and EAD use portfolio mean values. Individual estimates require running the full LGD/EAD pipeline.")

    st.markdown("---")
    st.subheader("Portfolio EL by Risk Class — Validation Set 2016")
    if len(portfolio_df) > 0:
        show_cols = [c for c in ["risk_class","N_loans","Avg_PD","Avg_LGD",
                                  "Avg_EL_rate","Total_Funded_M","EL_pct_funded",
                                  "Actual_Default_Rate"] if c in portfolio_df.columns]
        st.dataframe(portfolio_df[show_cols], width='stretch')

        rc_order = ["AA","A","AB","BB","B","BC","C","CD","DD","F"]
        pf2 = portfolio_df.set_index("risk_class").reindex(
            [r for r in rc_order if r in portfolio_df["risk_class"].values]
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(pf2.index, pf2["Avg_EL_rate"] * 100,
               color=[RISK_COLORS.get(r, "#888") for r in pf2.index])
        ax.set_ylabel("Average EL Rate (%)")
        ax.set_title("Average Expected Loss Rate by Risk Class")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── TAB 3 — Model Performance ─────────────────────────────────────────────────
with tab3:
    st.title("Model Performance — Validation Set 2016 vs OOT 2017–2018")

    if len(model_comp_df) > 0:
        perf_cols = [c for c in ["Model","Val AUC","OOT AUC","Val KS","OOT KS",
                                   "Val Gini","OOT Gini","Val Brier","ECE (Val)"]
                     if c in model_comp_df.columns]
        if perf_cols:
            st.dataframe(model_comp_df.reset_index()[perf_cols], width='stretch')

    st.markdown("---")
    ca, cb = st.columns(2)
    with ca:
        st.subheader("ROC Curves — All Models")
        show_chart("reports/05_roc_comparison.png")
    with cb:
        st.subheader("Calibration Curves")
        show_chart("reports/05_calibration_curves.png")
    st.subheader("Default Rate by Score Decile")
    show_chart("reports/05_decile_default_rates.png",
               "Monotonic decrease confirms correct score rank-ordering")

# ── TAB 4 — Explainability ────────────────────────────────────────────────────
with tab4:
    st.title("Explainability — SHAP & Scorecard Contributions")

    cl, cr = st.columns(2)

    with cl:
        st.subheader("Global Feature Importance (XGBoost SHAP)")
        st.caption("Mean |SHAP| — 2,000 stratified validation loans")
        if len(shap_df) > 0:
            top15 = shap_df.head(15)
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.barh(top15["Feature"][::-1], top15["Mean_Abs_SHAP"][::-1], color="#2c7bb6")
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("Top 15 Features — XGBoost SHAP")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with cr:
        st.subheader("This Application — Score Contributions")
        st.caption("SHAP_i = Factor × β_i × WoE_i  (additive relative to baseline)")
        top10 = pd.Series(contribs).sort_values(key=abs, ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(
            top10.index[::-1], top10.values[::-1],
            color=["#2ca02c" if v > 0 else "#d62728" for v in top10.values[::-1]],
        )
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Score Points (+ = lowers risk, − = raises risk)")
        ax.set_title(f"Top 10 Factors  |  Score: {credit_score:.0f}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("SHAP Summary Beeswarm")
    show_chart("reports/08_shap_summary.png")
    st.subheader("SHAP Dependence Plots")
    show_chart("reports/08_shap_dependence_plots.png",
               "x = feature value  |  y = SHAP contribution  |  color = interaction feature")
