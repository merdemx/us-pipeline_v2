import pandas as pd


def last_n_bdays(s: pd.Series, n: int) -> pd.Series:
    """Son n iş gününü döndürür. pandas Series.last() yerine kullanılır."""
    if len(s) == 0:
        return s
    cutoff = s.index[-1] - pd.offsets.BDay(n)
    return s[s.index > cutoff]


def last_n_bdays_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """DataFrame için son n iş günü."""
    if len(df) == 0:
        return df
    cutoff = df.index[-1] - pd.offsets.BDay(n)
    return df[df.index > cutoff]
