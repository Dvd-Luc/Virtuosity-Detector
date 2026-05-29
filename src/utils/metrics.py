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
    df = df[df[x_col] >= x_min]

    if x_max is None:
        x_max = df[x_col].max()
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
    log_y=False,
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
    log_y : bool
        Whether to log-transform Y before computing distances
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

    if log_y:
        y = np.log(y + 1e-9)
    # Distance formula
    d = (a * x - y + b) / np.sqrt(a**2 + 1)

    if signed:
        return d
    else:
        return np.abs(d)
    
def performance_orientation(df, reg, x_col="trill_rate", y_col="bandwidth", log_y=False):
    """
    Decompose performance into BW-oriented vs TR-oriented components.
    
    Returns a DataFrame with:
    - vd_total     : vocal deviation (perpendicular distance, as before)
    - delta_bw     : standardized residual on BW axis (how much BW exceeds prediction)
    - delta_tr     : standardized residual on TR axis (how much TR exceeds prediction)
    - orientation_ratio : delta_bw / (|delta_bw| + |delta_tr|), [-1, 1]
                          > 0  → BW-oriented
                          < 0  → TR-oriented
    - theta        : angle in degrees (atan2), same interpretation
    """
    x = df[x_col].values
    y = df[y_col].values

    if log_y:
            y = np.log(y + 1e-9)

    x_sd, x_mean = df[x_col].std(), df[x_col].mean()
    y_sd, y_mean = df[y_col].std(), df[y_col].mean()

    x_std = (x - x_mean) / x_sd +2
    y_std = (y - y_mean) / y_sd +2

    a = reg["slope"]
    b = reg["intercept"]

    a_std = a * (x_sd / y_sd)
    b_std = (a * x_mean + b - y_mean) / y_sd

    # Residual on BW axis: how far above/below the predicted BW
    y_pred_std = a_std * x_std + b_std
    delta_bw = y_std - y_pred_std  # > 0 = above the line (better BW than expected)

    # Residual on TR axis: how far right/left of where BW would predict TR
    # i.e. invert the line: x = (y - b) / a
    x_pred_std = (y_std - b_std) / a_std
    delta_tr = x_std - x_pred_std  # > 0 = more TR than BW would predict

    # Orientation
    denom = np.abs(y_std) + np.abs(x_std)
    orientation_ratio = np.where(denom > 0, y_std / denom, 0.0)
    theta = np.degrees(np.arctan2(y_std, x_std))

    # Vocal deviation (your existing metric, for reference)
    vd = np.abs(a * x - y + b) / np.sqrt(a**2 + 1)
    vd_std = np.abs(a_std * x_std - y_std + b_std) / np.sqrt(a_std**2 + 1)

    return pd.DataFrame({
        "vd_total": vd,
        "vd_std": vd_std,
        "delta_bw": delta_bw,
        "delta_tr": delta_tr,
        "orientation_ratio": orientation_ratio,
        "theta": theta
    }, index=df.index) 


def bill_centroid(df):
    length = df["Beak.Length_Culmen"]
    depth = df["Beak.Depth"]
    width = df["Beak.Width"]

    centroid = (length * depth * width)**(1/3)
    log_centroid = np.log(centroid)

    return centroid, log_centroid

