"""
src/preprocessing.py
====================
Reusable data cleaning and feature engineering helper functions.

Each function performs exactly one well-defined transformation and is designed
to be called from Jupyter notebooks, keeping notebook cells concise and the
logic testable in isolation.

Methodology and column decisions follow standard credit risk practice
for the Lending Club dataset. Deviations from standard approach are
noted in docstrings.

Functions
---------
drop_high_missing_columns   — Remove columns with >threshold% missing values.
drop_irrelevant_columns     — Drop known non-predictive or leakage-risk columns.
drop_rows_with_key_missing  — Remove the tiny fraction of rows missing in key cols.
clean_term                  — Convert term text ('36 months') → int (36).
clean_emp_length            — Convert employment length text → int years.
create_oot_split            — Chronological 3-way train/val/OOT split.
compute_woe_iv              — Compute Weight of Evidence and Information Value.
apply_woe_encoding          — Map feature values to pre-fitted WoE scores.
convert_dates               — Parse issue_d and earliest_cr_line to datetime.
compute_mths_since_cr_line  — Credit history length in months at loan issue date.
create_target_variables     — Create default, recovery_rate, credit_conv_factor.
optimize_memory             — Downcast dtypes to reduce DataFrame memory usage.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — column lists following standard credit risk methodology
# ---------------------------------------------------------------------------

# Columns to always drop regardless of missing rate.
# Reasons documented inline:
IRRELEVANT_COLUMNS = [
    # Identifiers — unique per row, no predictive value
    "id", "member_id",
    # Free-text / URL — too high cardinality or not useful
    "url", "desc", "title", "emp_title", "zip_code",
    # Future-dated columns — not available at loan origination (data leakage)
    "next_pymnt_d", "last_pymnt_d", "last_credit_pull_d",
    "last_pymnt_amnt", "last_fico_range_high", "last_fico_range_low",
    # Redundant — information already captured in non-_inv counterparts
    "funded_amnt_inv", "total_pymnt_inv", "out_prncp_inv",
    # Near-constant — essentially a single value across the dataset
    "application_type", "pymnt_plan", "policy_code",
    # Administrative — not a borrower characteristic
    "Unnamed: 0",
]

# Columns where NaN rows are dropped outright.
# These columns have <0.01% missing — removal is safe and avoids imputation.
# Reference: standard data cleaning for Lending Club — see notebook 01.
TINY_MISSING_DROP_COLS = [
    "annual_inc", "total_acc", "acc_now_delinq", "pub_rec",
    "open_acc", "delinq_2yrs", "earliest_cr_line", "inq_last_6mths",
]

# Columns used to define the PD target — kept available for LGD/EAD targets
# but excluded from PD feature set later to prevent data leakage.
# These are post-outcome columns (amounts received, recovered, etc.)
LEAKAGE_COLUMNS_FOR_PD = [
    "funded_amnt", "out_prncp", "total_pymnt", "total_rec_prncp",
    "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "loan_status",
]


# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------

def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.70) -> pd.DataFrame:
    """Drop columns where the missing value rate exceeds the threshold.

    A 70% cut-off is used following standard credit scoring methodology —
    imputing columns with very high missingness would introduce substantial bias.

    Args:
        df:        Input DataFrame.
        threshold: Maximum allowed missing proportion (default 0.70).

    Returns:
        DataFrame with high-missing columns removed.
    """
    missing_rate = df.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()

    if cols_to_drop:
        logger.info(
            "Dropping %d columns with >%.0f%% missing values.",
            len(cols_to_drop), threshold * 100,
        )
        df = df.drop(columns=cols_to_drop)
    else:
        logger.info("No columns exceed the %.0f%% missing threshold.", threshold * 100)

    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop known irrelevant, redundant, or high-cardinality columns.

    Only drops columns that actually exist in the DataFrame, so this is
    safe to call regardless of which version of the Lending Club dataset
    is loaded.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with irrelevant columns removed.
    """
    cols_present = [c for c in IRRELEVANT_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_present)
    logger.info("Dropped %d irrelevant columns.", len(cols_present))
    return df


def drop_rows_with_key_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values in key columns.

    Missing values in these columns represent less than 0.01% of the data.
    Dropping them is safe given the large dataset size and avoids
    introducing imputation bias in important features.

    Reference: standard data cleaning — these columns have <0.01% missing;
    dropping is safe given the large dataset size.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with those rows removed, index reset.
    """
    cols_to_check = [c for c in TINY_MISSING_DROP_COLS if c in df.columns]
    before = len(df)
    df = df.dropna(subset=cols_to_check).reset_index(drop=True)
    dropped = before - len(df)
    logger.info(
        "Dropped %d rows with missing values in key columns (%.4f%% of data).",
        dropped, dropped / before * 100,
    )
    return df


def clean_term(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the 'term' column from text to integer (number of months).

    The raw column contains strings like ' 36 months' or ' 60 months'.
    After cleaning, the column holds integers 36 or 60.

    Reference: standard Lending Club preprocessing.

    Args:
        df: DataFrame containing a 'term' column.

    Returns:
        DataFrame with 'term' as integer dtype.
    """
    df = df.copy()
    df["term"] = df["term"].apply(lambda x: int(str(x).strip()[:2]))
    logger.info("'term' column converted to int. Unique values: %s.", df["term"].unique().tolist())
    return df


def clean_emp_length(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'emp_length' from text categories to integer years.

    Processing steps (following Lending Club standard preprocessing):
      1. Fill NaN with '0 years' — assumed to be no professional experience.
      2. Strip the '+' from '10+ years'.
      3. Replace '< 1 year' with '0'.
      4. Strip ' years' and ' year' suffixes.
      5. Cast to int.

    regex=False is used to avoid FutureWarning in pandas ≥2.0.

    Args:
        df: DataFrame containing an 'emp_length' column.

    Returns:
        DataFrame with 'emp_length' as integer dtype (values 0–10).
    """
    df = df.copy()
    df["emp_length"] = df["emp_length"].fillna("0 years")
    df["emp_length"] = (
        df["emp_length"]
        .str.replace("+", "", regex=False)
        .str.replace("< 1 year", "0", regex=False)
        .str.replace(" years", "", regex=False)
        .str.replace(" year", "", regex=False)
        .astype(int)
    )
    logger.info(
        "'emp_length' converted to int. Range: %d – %d years.",
        df["emp_length"].min(), df["emp_length"].max(),
    )
    return df


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date columns from text to datetime.

    Columns converted:
    - 'issue_d'          : month the loan was funded (e.g. 'Jan-2014')
    - 'earliest_cr_line' : borrower's earliest credit line (e.g. 'Apr-1990')

    The Lending Club 2007–2018 dataset uses 4-digit year format ('%b-%Y').
    Invalid values are coerced to NaT.

    Args:
        df: DataFrame with raw string date columns.

    Returns:
        DataFrame with date columns as datetime64.
    """
    df = df.copy()
    for col in ["issue_d", "earliest_cr_line"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%b-%Y", errors="coerce")
            n_nat = df[col].isna().sum()
            if n_nat > 0:
                logger.warning("'%s': %d NaT values after date parsing.", col, n_nat)
            else:
                logger.info("'%s' parsed successfully.", col)
    return df


def compute_mths_since_cr_line(df: pd.DataFrame) -> pd.DataFrame:
    """Compute credit history length in months at the time of loan issue.

    Definition: number of months between the borrower's earliest credit line
    and the loan issue date. This represents credit history length at the
    time of loan application — the correct quantity for an application model.

    This implementation uses per-loan 'issue_d' so that each loan correctly
    reflects its own credit history length at the time of issuance, rather
    than a fixed reference date which would over-state history for older loans.

    After computing the feature, 'earliest_cr_line' is dropped as it is no
    longer needed.

    Args:
        df: DataFrame with 'issue_d' and 'earliest_cr_line' as datetime.

    Returns:
        DataFrame with a new 'mths_since_earliest_cr_line' int column and
        'earliest_cr_line' dropped.
    """
    df = df.copy()
    months = (
        (df["issue_d"].dt.year - df["earliest_cr_line"].dt.year) * 12
        + (df["issue_d"].dt.month - df["earliest_cr_line"].dt.month)
    )
    # Negative values occur when earliest_cr_line parsing yields a
    # 2-digit century ambiguity (e.g. 'Apr-68' → 2068 instead of 1968).
    # Replace with the column maximum to preserve the semantics:
    # "a very long credit history."
    max_months = months[months >= 0].max()
    n_negative = (months < 0).sum()
    if n_negative > 0:
        logger.warning(
            "%d negative credit history values found (century parsing issue). "
            "Replacing with column maximum (%d months).",
            n_negative, max_months,
        )
        months = months.clip(lower=0)
        months[months == 0] = max_months  # Preserve semantics for century-flip rows

    df["mths_since_earliest_cr_line"] = months.astype(int)
    df = df.drop(columns=["earliest_cr_line"])
    logger.info(
        "'mths_since_earliest_cr_line' created. Range: %d – %d months.",
        df["mths_since_earliest_cr_line"].min(),
        df["mths_since_earliest_cr_line"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# Target variable creation
# ---------------------------------------------------------------------------

def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create all three model target variables: PD, LGD, and EAD targets.

    PD target — 'default':
        1 = good borrower (did not default)
        0 = bad borrower  (defaulted)
        Default statuses follow standard credit risk convention:
        'Charged Off', 'Late (31-120 days)',
        'Does not meet the credit policy. Status:Charged Off', 'Default'.
        Loans still current, in grace period, or late 16–30 days are
        treated as non-default because their final outcome is uncertain.

    LGD target — 'recovery_rate':
        recoveries / funded_amnt, clipped to [0, 1].
        LGD = 1 − recovery_rate.

    EAD target — 'credit_conversion_factor' (CCF):
        (funded_amnt − total_rec_prncp) / funded_amnt, clipped to [0, 1].
        EAD = funded_amnt × CCF.

    Args:
        df: Cleaned DataFrame containing 'loan_status', 'recoveries',
            'funded_amnt', and 'total_rec_prncp'.

    Returns:
        DataFrame with three new columns: 'default', 'recovery_rate',
        'credit_conversion_factor'.
    """
    df = df.copy()

    # -- PD target --
    bad_statuses = {
        "Charged Off",
        "Late (31-120 days)",
        "Does not meet the credit policy. Status:Charged Off",
        "Default",
    }
    df["default"] = np.where(df["loan_status"].isin(bad_statuses), 0, 1)
    default_rate = 1 - df["default"].mean()
    logger.info(
        "PD target 'default' created. Default rate: %.2f%% | Good: %d | Bad: %d",
        default_rate * 100,
        df["default"].sum(),
        (df["default"] == 0).sum(),
    )

    # -- LGD target --
    df["recovery_rate"] = (df["recoveries"] / df["funded_amnt"]).clip(0, 1)
    logger.info(
        "LGD target 'recovery_rate' created. Mean: %.4f | Zero-rate rows: %.1f%%",
        df["recovery_rate"].mean(),
        (df["recovery_rate"] == 0).mean() * 100,
    )

    # -- EAD target --
    df["credit_conversion_factor"] = (
        (df["funded_amnt"] - df["total_rec_prncp"]) / df["funded_amnt"]
    ).clip(0, 1)
    logger.info(
        "EAD target 'credit_conversion_factor' created. Mean: %.4f",
        df["credit_conversion_factor"].mean(),
    )

    return df


# ---------------------------------------------------------------------------
# Memory optimization
# ---------------------------------------------------------------------------

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric dtypes and convert object columns to category.

    Reduces DataFrame memory footprint by ~60% by downcasting to smaller
    integer and float types. Object columns are converted to 'category'.
    The result should be saved as parquet to preserve these dtypes.

    Args:
        df: Cleaned DataFrame ready for dtype optimization.

    Returns:
        Memory-optimized DataFrame.
    """
    df = df.copy()
    before_mb = df.memory_usage(deep=True).sum() / 1e6

    # Explicit downcasting for known columns
    to_float32 = [
        "installment", "annual_inc", "dti", "int_rate",
        "out_prncp", "total_pymnt", "total_rec_prncp",
        "total_rec_int", "total_rec_late_fee", "recoveries",
        "collection_recovery_fee", "revol_util",
        "mths_since_last_delinq", "collections_12_mths_ex_med",
        "recovery_rate", "credit_conversion_factor",
    ]
    to_int8 = ["default", "term"]
    to_int16 = [
        "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
        "total_acc", "acc_now_delinq", "emp_length",
        "mths_since_earliest_cr_line",
    ]
    to_int32 = ["loan_amnt", "funded_amnt"]

    for col in to_float32:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    for col in to_int8:
        if col in df.columns:
            df[col] = df[col].astype("int8")
    for col in to_int16:
        if col in df.columns:
            df[col] = df[col].astype("int16")
    for col in to_int32:
        if col in df.columns:
            df[col] = df[col].astype("int32")

    # Convert all remaining object columns to category
    obj_cols = df.select_dtypes("object").columns.tolist()
    df[obj_cols] = df[obj_cols].astype("category")

    after_mb = df.memory_usage(deep=True).sum() / 1e6
    logger.info(
        "Memory optimized: %.1f MB → %.1f MB (%.0f%% reduction).",
        before_mb, after_mb, (1 - after_mb / before_mb) * 100,
    )
    return df


# ---------------------------------------------------------------------------
# OOT Split
# ---------------------------------------------------------------------------

def create_oot_split(
    df: pd.DataFrame,
    date_col: str = "issue_d",
    train_end: str = "2015-12-31",
    val_end: str = "2016-12-31",
):
    """Create a chronological 3-way train / validation / OOT split.

    This must be called BEFORE any WoE fitting or feature preprocessing
    to prevent leakage from future data into the training distribution.

    Split boundaries (inclusive):
    - Train      : issue_d <= train_end            (2007 – Dec 2015)
    - Validation : train_end < issue_d <= val_end  (Jan – Dec 2016)
    - OOT        : issue_d > val_end               (Jan 2017 – Dec 2018)

    Args:
        df:        DataFrame with a parsed datetime date column.
        date_col:  Column name for the loan issue date (default 'issue_d').
        train_end: Last date of the training period (inclusive).
        val_end:   Last date of the validation period (inclusive).

    Returns:
        Tuple (train, val, oot) — three DataFrames with reset index.
    """
    t_end = pd.Timestamp(train_end)
    v_end = pd.Timestamp(val_end)
    dates = df[date_col]
    valid = dates.notna()

    train = df[valid & (dates <= t_end)].reset_index(drop=True)
    val   = df[valid & (dates > t_end) & (dates <= v_end)].reset_index(drop=True)
    oot   = df[valid & (dates > v_end)].reset_index(drop=True)

    logger.info(
        "OOT split complete: Train=%d | Val=%d | OOT=%d | Rows with NaT=%d",
        len(train), len(val), len(oot), (~valid).sum(),
    )
    logger.info(
        "Default rates — Train: %.2f%% | Val: %.2f%% | OOT: %.2f%%",
        (1 - train["default"].mean()) * 100,
        (1 - val["default"].mean()) * 100,
        (1 - oot["default"].mean()) * 100,
    )
    return train, val, oot


# ---------------------------------------------------------------------------
# WoE / IV
# ---------------------------------------------------------------------------

def compute_woe_iv(
    train_df: pd.DataFrame,
    feature: str,
    target: str = "default",
    n_bins: int = 10,
):
    """Compute Weight of Evidence (WoE) and Information Value (IV).

    Convention: target=1 is good (non-default), target=0 is bad.
    WoE = ln(% good in bin / % bad in bin)
    - Positive WoE → bin has proportionally more good loans → lower risk
    - Negative WoE → bin has proportionally more bad loans → higher risk

    IV Interpretation:
    | IV Range     | Predictive Power          |
    |--------------|---------------------------|
    | < 0.02       | Useless                   |
    | 0.02 – 0.10  | Weak                      |
    | 0.10 – 0.30  | Medium (use in scorecard) |
    | 0.30 – 0.50  | Strong                    |
    | > 0.50       | Suspicious (check leakage)|

    WoE is always fitted on training data only. Apply to val/OOT using
    apply_woe_encoding() with the returned bin_edges.

    Args:
        train_df: Training-only DataFrame.
        feature:  Column name to analyse.
        target:   Binary target column (1=good, 0=bad).
        n_bins:   Number of quantile bins for numerical features.

    Returns:
        woe_df:    DataFrame with per-bin statistics and WoE values.
        total_iv:  Total Information Value (float).
        bin_edges: Bin edges list for numerical features; None for categorical.
    """
    df = train_df[[feature, target]].copy()
    total_good = (df[target] == 1).sum()
    total_bad  = (df[target] == 0).sum()
    bin_edges  = None

    is_numeric = (
        pd.api.types.is_numeric_dtype(df[feature])
        and df[feature].nunique(dropna=True) > 10
    )

    if is_numeric:
        non_null = df[feature].dropna()
        _, edges = pd.qcut(non_null, q=n_bins, duplicates="drop", retbins=True)
        edges[0]  -= 1e-6   # ensure the minimum is included
        edges[-1] += 1e-6   # ensure the maximum is included
        bin_edges = edges.tolist()
        df["bin"] = pd.cut(
            df[feature], bins=edges, include_lowest=True
        ).astype(str)
    else:
        df["bin"] = df[feature].astype(str)

    # Missing values are treated as a separate category
    df.loc[df[feature].isna(), "bin"] = "Missing"

    agg = df.groupby("bin")[target].agg(["sum", "count"]).reset_index()
    agg.columns = ["bin", "n_good", "n_obs"]
    agg["n_bad"]    = agg["n_obs"] - agg["n_good"]
    agg["bad_rate"] = (agg["n_bad"] / agg["n_obs"]).round(4)
    agg["pct_good"] = (agg["n_good"] / total_good).clip(lower=1e-9)
    agg["pct_bad"]  = (agg["n_bad"]  / total_bad ).clip(lower=1e-9)
    agg["woe"]      = np.log(agg["pct_good"] / agg["pct_bad"]).round(4)
    agg["iv"]       = ((agg["pct_good"] - agg["pct_bad"]) * agg["woe"]).round(6)

    total_iv = round(float(agg["iv"].sum()), 6)
    woe_df   = agg.sort_values("bad_rate", ascending=False).reset_index(drop=True)

    return woe_df, total_iv, bin_edges


def apply_woe_encoding(
    df: pd.DataFrame,
    feature: str,
    woe_df: pd.DataFrame,
    bin_edges=None,
) -> pd.Series:
    """Apply a pre-fitted WoE encoding to a feature in any split.

    Maps each value to its WoE score from the training-fitted WoE table.
    Unknown categories or out-of-range values receive WoE=0 (neutral).

    Args:
        df:        DataFrame containing the feature column.
        feature:   Column name.
        woe_df:    WoE table returned by compute_woe_iv() (fitted on train).
        bin_edges: Bin edges for numerical features (from compute_woe_iv()).
                   Pass None for categorical features.

    Returns:
        Float Series of WoE-encoded values, same length as df.
    """
    woe_map = dict(zip(woe_df["bin"], woe_df["woe"]))
    series  = df[feature].copy()

    if bin_edges is not None:
        # Numerical: apply the same bin edges used during training
        binned = pd.cut(
            series, bins=bin_edges, include_lowest=True
        ).astype(str)
        binned[series.isna()] = "Missing"
    else:
        # Categorical: direct string mapping
        binned = series.astype(str)
        binned[series.isna()] = "Missing"

    return binned.map(woe_map).fillna(0.0).astype(float)

