import numpy as np
import pandas as pd
from scipy.stats import linregress

def upper_bound_regression(
    df,
    x_col="trill_rate",
    y_col="bandwidth",
    bin_width=2.0,
    x_min=None,
    x_max=None,
    log_y=False
):
    """
    Compute an upper-bound regression using binned maxima (Podos-style).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    x_col : str
        X variable (e.g. trill_rate)
    y_col : str
        Y variable (e.g. bandwidth)
    bin_width : float
        Bin width in X units (Hz)
    x_min : float
        Minimum X to consider
    x_max : float or None
        Maximum X to consider (None = inferred from data)
    log_y : bool
        Whether to log-transform Y before regression

    Returns
    -------
    ub_df : pd.DataFrame
        Upper-bound points (bin_center, x, y)
    reg : dict
        Regression results (slope, intercept, r, p, stderr)
    """

    df = df[[x_col, y_col]].dropna()
    if x_min is None:
        x_min = df[x_col].min()
        print(f"Inferred x_min from data: {x_min:.2f}")
    df = df[df[x_col] >= x_min]

    if x_max is None:
        x_max = df[x_col].max()
        print(f"Inferred x_max from data: {x_max:.2f}")
    df = df[df[x_col] <= x_max]

    # Define bins
    bins = np.arange(x_min, x_max + bin_width, bin_width)
    df["bin"] = pd.cut(df[x_col], bins=bins, include_lowest=True)

    # Select max Y per bin
    ub = (
        df.loc[df.groupby("bin")[y_col].idxmax()]
        .sort_values(x_col)
        .copy()
    )

    if log_y:
        ub["y_reg"] = np.log(ub[y_col] + 1e-9)
    else:
        ub["y_reg"] = ub[y_col]

    # Regression
    res = linregress(ub[x_col], ub["y_reg"])

    reg = {
        "slope": res.slope,
        "intercept": res.intercept,
        "r_value": res.rvalue,
        "p_value": res.pvalue,
        "stderr": res.stderr,
        "n": len(ub)
    }

    return ub, reg


def distance_to_upper_bound(
    df,
    reg,
    x_col="trill_rate",
    y_col="bandwidth",
    signed=False
):
    """
    Compute Euclidean distance from each point to the upper-bound regression.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with trill_rate and bandwidth
    reg : dict
        Regression output from upper_bound_regression()
    x_col : str
        X variable name
    y_col : str
        Y variable name
    signed : bool
        If True, distances are signed (negative = below the bound)

    Returns
    -------
    distances : np.ndarray
        Distance for each row (same order as df)
    """

    a = reg["slope"]
    b = reg["intercept"]

    x = df[x_col].values
    y = df[y_col].values

    # Distance formula
    d = (a * x - y + b) / np.sqrt(a**2 + 1)

    if signed:
        return d
    else:
        return np.abs(d)