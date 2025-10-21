
import pandas as pd

DATE_COLS = ["date_activ","date_end","date_renewal"]

def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
    return df

def engineer_dates(df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
    df = df.copy()
    df = _parse_dates(df)
    if reference_date is None:
        reference_date = pd.Timestamp.today().normalize()
    if "date_activ" in df.columns:
        df["days_since_activation"] = (reference_date - df["date_activ"]).dt.days
    if "date_renewal" in df.columns:
        df["days_until_renewal"] = (df["date_renewal"] - reference_date).dt.days
    if "date_end" in df.columns and "date_activ" in df.columns:
        df["tenure_days"] = (df["date_end"] - df["date_activ"]).dt.days
    return df.drop(columns=[c for c in DATE_COLS if c in df.columns])
