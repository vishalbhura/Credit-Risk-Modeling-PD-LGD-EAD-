"""
src/data_loader.py
==================
Handles loading the raw Lending Club CSV, validating the schema,
and sorting the dataset chronologically — the mandatory first step
before any preprocessing or splitting.

Functions
---------
load_raw_data        — Load raw CSV and validate schema.
sort_chronologically — Sort by issue date for correct OOT split ordering.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema: minimum set of columns required in the raw Lending Club dataset.
# The 2007–2018Q4 extract contains 150 columns; only the columns used
# across all modeling phases are listed here for validation purposes.
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    # Loan terms
    "loan_status", "loan_amnt", "funded_amnt", "term", "int_rate",
    "installment", "grade", "sub_grade",
    # Borrower profile
    "emp_length", "home_ownership", "annual_inc", "purpose", "addr_state",
    "dti", "earliest_cr_line",
    # Credit bureau
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "acc_now_delinq",
    # Repayment / recovery (used for LGD and EAD targets only)
    "out_prncp", "total_pymnt", "total_rec_prncp", "total_rec_int",
    "recoveries", "collection_recovery_fee",
    # Date
    "issue_d",
]


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the raw Lending Club CSV file and validate its schema.

    The returned DataFrame is unsorted and unprocessed — it contains all
    original columns exactly as they appear in the source file.

    Args:
        filepath: Path to the raw CSV file
                  (e.g. 'data/raw/accepted_2007_to_2018Q4.csv').

    Returns:
        Raw DataFrame with all original columns.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If required columns are missing from the dataset.
    """
    logger.info("Loading raw data from: %s", filepath)
    df = pd.read_csv(filepath, low_memory=False)
    logger.info("Loaded %d rows × %d columns.", len(df), df.shape[1])

    _validate_schema(df)
    return df


def sort_chronologically(df: pd.DataFrame, date_col: str = "issue_d") -> pd.DataFrame:
    """Sort the dataset by loan issue date in ascending order.

    This must be the first operation on the data — before any split or
    preprocessing. The chronological order is required for a valid
    Out-of-Time (OOT) split that mirrors how credit models are validated
    in practice: trained on past loans, evaluated on future loans.

    The date column is parsed but NOT retained as datetime in the returned
    DataFrame — the parsing step happens properly in convert_dates() inside
    preprocessing.py. This function only sorts.

    Args:
        df:       Raw Lending Club DataFrame.
        date_col: Column to sort by (default: 'issue_d').

    Returns:
        DataFrame sorted by issue date ascending, index reset.
    """
    temp_date = pd.to_datetime(df[date_col], format="%b-%Y", errors="coerce")
    df = df.assign(_sort_key=temp_date).sort_values("_sort_key")
    df = df.drop(columns=["_sort_key"]).reset_index(drop=True)
    logger.info(
        "Dataset sorted chronologically by '%s'. Date range: %s → %s.",
        date_col,
        temp_date.min().strftime("%b-%Y"),
        temp_date.max().strftime("%b-%Y"),
    )
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if any required column is missing.

    Args:
        df: Raw DataFrame to validate.

    Raises:
        ValueError: Lists all missing columns if any are absent.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Schema validation failed — {len(missing)} required column(s) missing:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )
    logger.info("Schema validation passed (%d required columns found).", len(REQUIRED_COLUMNS))
