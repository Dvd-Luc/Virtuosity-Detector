import os
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
import re
import plotly.express as px
from plotly.subplots import make_subplots

from src.config import load_config_yaml


def plot_virtuosity(df, x_col="trill_rate", y_col="bandwidth", logy=False, upper_bound=None, reg=None):
    xlabel = f'{x_col} (trills/sec)'
    ylabel = f'{y_col} (Hz)'
    title = f'{y_col} vs {x_col}'

    df_plot = df.dropna(subset=[x_col, y_col])
    df_plot = df_plot[df_plot[x_col] >= 2]
    if logy:
        df_plot["log_bandwidth"] = np.log(df_plot[y_col] + 1e-6)
        y_col = "log_bandwidth"
        ylabel = f'log({y_col}) (log(Hz))'
        title = f'log({y_col}) vs {x_col}'

    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df_plot, x=x_col, y=y_col, alpha=0.6)

    if upper_bound is not None and reg is not None:
        # ub_df, reg = upper_bound_regression(df_plot, x_col=x_col, y_col=y_col)
        plt.scatter(upper_bound[x_col], upper_bound[y_col], color="red",s=30, label="Upper bound points")
        x_line = np.linspace(upper_bound["trill_rate"].min(), upper_bound["trill_rate"].max(), 200)
        y_line = reg["intercept"] + reg["slope"] * x_line
        plt.plot(x_line, y_line, "r--", label="Upper-bound regression")
        title += f"\nUpper-bound regression: y = {reg['intercept']:.2f} + {reg['slope']:.2f}*x (R={reg['r_value']:.2f}, p={reg['p_value']:.3e})"
    
    plt.xlabel(f"{xlabel}")
    plt.ylabel(f"{ylabel}")
    plt.title(f"{title}")
    plt.xlim(0, 200)
    plt.grid(True)
    plt.show()

def plot_dist_vs_traits_plotly(df, hue_col, title_suffix):
    traits = [
        ("mass", "Mass"),
        ("Beak.Length_Culmen", "Beak length (culmen)"),
        ("Beak.Width", "Beak width"),
        ("Beak.Depth", "Beak depth"),
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[t[1] for t in traits],
        shared_xaxes=True,
        shared_yaxes=False,   # 🔑 axes Y indépendants
    )

    for i, (col, label) in enumerate(traits):
        row = i // 2 + 1
        col_i = i % 2 + 1

        fig_px = px.scatter(
            df,
            x="dist_to_bound",
            y=col,
            color=hue_col,
            opacity=0.7,
            hover_data={
                "dist_to_bound": True,
                col: True,
                hue_col: True,
                "file_name": True
            }
        )

        for trace in fig_px.data:
            trace.showlegend = (i == 0)  # légende une seule fois
            fig.add_trace(trace, row=row, col=col_i)

        fig.update_yaxes(title_text=label, row=row, col=col_i)
        fig.update_xaxes(title_text="Distance to performance bound", row=row, col=col_i)

    fig.update_layout(
        title=f"Distance to performance bound vs morphology ({title_suffix})",
        height=900,
        width=2000,
        legend_title_text=hue_col,
        template="simple_white"
    )

    fig.show()


def upper_bound_regression(
    df,
    x_col="trill_rate",
    y_col="bandwidth",
    bin_width=2.0,
    x_min=2,
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
    df = df[df[x_col] >= x_min]

    if x_max is None:
        x_max = df[x_col].max()

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
    
def prepare_dataset(df):
    df["file_name"] = df.apply(
        lambda r: f"{r['file_name_radical'].split('.')[0]}_seg{r['segment_id']}.wav",
        axis=1
    )

    df["trill_duration"] = df["t_max"] - df["t_min"]
    df["trill_rate"] = df.apply(
        lambda row: 0 if row["trill_duration"] == 0 else row["count"] / row["trill_duration"],
        axis=1
    )
    df["bandwidth"] = df["f_max"] - df["f_min"]

    df_filtered = df[df["trill_rate"] >= 2]

    ub_df, reg = upper_bound_regression(
        df_filtered,
        x_col="trill_rate",
        y_col="bandwidth",
        bin_width=2.0,
        x_min=2,
        log_y=False
    )

    df_filtered["dist_to_bound"] = distance_to_upper_bound(
        df_filtered,
        reg,
        x_col="trill_rate",
        y_col="bandwidth",
        signed=True
    )

    return df_filtered, ub_df, reg

def load_meta_and_morpho(DATA_DIR, file_timestamps, file_meta, file_morpho):

    df_timestamps = pd.read_csv(os.path.join(DATA_DIR, file_timestamps))
    df_metadata = pd.read_csv(os.path.join(DATA_DIR, file_meta))
    df_morpho = pd.read_csv(os.path.join(DATA_DIR, file_morpho))

    df_timestamps["file_name"] = df_timestamps.apply(
        lambda r: r["file_name"].rsplit(".", 1)[0] + f"_seg{r['syllable_rank']}.wav",
        axis=1
    )

    df_metadata["file_name_radical"] = df_metadata["file_name"].apply(
        lambda x: re.sub(r"_seg\d+\.wav$", ".wav", x)
    )

    df_metadata_sub = df_metadata[
        ['gen', 'family', 'species', 'sub_species', 'common_name', 'recordist', 'date', 'time',
            'country', 'location', 'lat', 'lng', 'bird', 'file_name', 'file_name_radical',
            'gmm_cluster', 'gmm_prob_1', 'gmm_prob_2', 'gmm_prob_4']
    ]

    CLUSTER_MAP = {1: "Slow", 2: "Fast", 4: "Ultrafast"}
    df_metadata_sub['gmm_cluster_label'] = df_metadata_sub['gmm_cluster'].map(CLUSTER_MAP)

    Threshold_PROBA = 0.99
    df_metadata_filtered = df_metadata_sub[
        df_metadata_sub.apply(lambda row: row[f"gmm_prob_{row['gmm_cluster']}"] >= Threshold_PROBA, axis=1)
    ]
    df_metadata_filtered.reset_index(drop=True, inplace=True)

    df_merged = pd.merge(df_timestamps, df_metadata_filtered, on="file_name", how="inner")
    df_merged = pd.merge(df_merged, df_morpho, on="species", how="inner")

    return df_merged

def main():

    config = load_config_yaml(yaml_path="config.yaml")

    pred_annotation_file = "annotations_trills_v2_tests_predictions.csv"
    file_timestamps = "segments_passerines_filtered.csv"
    file_meta = "traits_data_pc_gmm_8components_proba_filtered.csv"
    file_morpho = "data_morpho.csv"

    df_pred = pd.read_csv(os.path.join(config.data_processed_subdir, pred_annotation_file))

    df_pred_filtered, ub_df, reg = prepare_dataset(df_pred)

    df_merged = load_meta_and_morpho(config.data_raw_subdir, file_timestamps, file_meta, file_morpho)
    df_merged = pd.merge(df_pred_filtered, df_merged, on="file_name", how="inner")

    print("\n" + "="*70)
    print("WORKFLOW OPTIONS")
    print("="*70)
    print("1. Plot virtuosity space and upper bound regression")
    print("2. Plot viruosity against morphology metrics")

    choice = input("\nChoice (1-2): ").strip()

    if choice == "1":
        input_log_y = input("Log-transform bandwidth for plotting? (y/n): ").strip().lower()
        log_y = input_log_y == "y"

        
        input_show_bound = input("Show upper bound regression on plot? (y/n): ").strip().lower()
        show_bound = input_show_bound == "y"

        plot_virtuosity(
            df_pred_filtered,
            x_col="trill_rate",
            y_col="bandwidth",
            logy=log_y,
            upper_bound=ub_df if show_bound else None,
            reg=reg if show_bound else None
        )
    
    elif choice == "2":

        plot_dist_vs_traits_plotly(
            df_merged,
            hue_col="family",
            title_suffix="colored by family"
        )

    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()