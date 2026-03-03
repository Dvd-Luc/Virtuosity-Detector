import os
import librosa
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import load_config_yaml
from src.main import visualize_and_confirm_predictions
from src.utils.gui_visualizer import launch_gui

def visualize_best_virtuosity_samples(df, config, x_col="trill_rate", y_col="bandwidth", sorting_metric="dist_to_bound", bin_width=2.0, top_n=5, plot=False):
    df_plot = df.dropna(subset=[sorting_metric])

    x_min = df_plot[x_col].min()
    x_max = df_plot[x_col].max()
    bins = np.arange(x_min, x_max + bin_width, bin_width)
    df_plot["bin"] = pd.cut(df_plot[x_col], bins=bins, include_lowest=True)

    best_samples = (
        df_plot.sort_values(sorting_metric)
        .groupby("bin")
        .head(top_n)
    )

    if plot:
        plot_virtuosity(best_samples, x_col=x_col, y_col=y_col, logy=False, upper_bound=None, reg=None)

    visualize_and_confirm_predictions(config, df_predictions=best_samples)
    
def plot_virtuosity(df, x_col="trill_rate", y_col="bandwidth", logy=False, hue_col=None, upper_bound=None, reg=None):
    df_plot = df.dropna(subset=[x_col, y_col])
    df_plot = df_plot[df_plot[x_col] >= 2]

    if logy:
        df_plot["log_bandwidth"] = np.log(df_plot[y_col] + 1e-6)
        y_col = "log_bandwidth"
        ylabel = f'log({y_col}) (log(Hz))'
        title = f'log({y_col}) vs {x_col}'
    else:
        ylabel = f'{y_col} (Hz)'
        title = f'{y_col} vs {x_col}'


    if hue_col is not None:
        fig = px.scatter(
            df_plot,
            x=x_col,
            y=y_col,
            color=hue_col,
            color_continuous_scale="Viridis",
            opacity=0.6,
            labels={x_col: f"{x_col} (trills/sec)", y_col: ylabel},
            title=title,
        )
    else:
        fig = px.scatter(
            df_plot,
            x=x_col,
            y=y_col,
            opacity=0.6,
            labels={x_col: f"{x_col} (trills/sec)", y_col: ylabel},
            title=title,
        )

    if upper_bound is not None and reg is not None:

        fig.add_trace(
            go.Scatter(
                x=upper_bound[x_col],
                y=upper_bound[y_col],
                mode="markers",
                marker=dict(color="red", size=8),
                name="Upper bound points",
            )
        )

        x_line = np.linspace(upper_bound[x_col].min(), upper_bound[x_col].max(), 200)
        y_line = reg["intercept"] + reg["slope"] * x_line
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color="red", dash="dash"),
                name=f"Upper-bound regression: y = {reg['intercept']:.2f} + {reg['slope']:.2f}*x (R={reg['r_value']:.2f}, p={reg['p_value']:.3e})",
            )
        )

    fig.update_layout(
        xaxis_title=f"{x_col} (trills/sec)",
        yaxis_title=ylabel,
        title=title,
        showlegend=True,
        # grid=True,
    )

    fig.show()

    # plt.figure(figsize=(8,6))
    # if hue_col is not None:
    #     hue_norm = plt.Normalize(df[hue_col].min(), df[hue_col].max())
    #     palette = sns.color_palette("viridis", as_cmap=True)
    #     sns.scatterplot(data=df_plot, x=x_col, y=y_col, hue=hue_col, hue_norm=hue_norm, alpha=0.6, palette=palette)
    # else:
    #     sns.scatterplot(data=df_plot, x=x_col, y=y_col, color="blue", alpha=0.6)

    # if upper_bound is not None and reg is not None:
    #     # ub_df, reg = upper_bound_regression(df_plot, x_col=x_col, y_col=y_col)
    #     plt.scatter(upper_bound[x_col], upper_bound[y_col], color="red",s=30, label="Upper bound points")
    #     x_line = np.linspace(upper_bound["trill_rate"].min(), upper_bound["trill_rate"].max(), 200)
    #     y_line = reg["intercept"] + reg["slope"] * x_line
    #     plt.plot(x_line, y_line, "r--", label="Upper-bound regression")
    #     title += f"\nUpper-bound regression: y = {reg['intercept']:.2f} + {reg['slope']:.2f}*x (R={reg['r_value']:.2f}, p={reg['p_value']:.3e})"
    
    # plt.xlabel(f"{xlabel}")
    # plt.ylabel(f"{ylabel}")
    # plt.title(f"{title}")
    # # plt.xlim(0, 200)
    # plt.grid(True)
    # plt.show()

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
        shared_yaxes=False,
    )

    for i, (col, label) in enumerate(traits):
        row = i // 2 + 1
        col_i = i % 2 + 1

        fig_px = px.scatter(
            df,
            x=col,
            y="dist_to_bound",
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
            trace.showlegend = (i == 0)
            fig.add_trace(trace, row=row, col=col_i)

        fig.update_yaxes(title_text="Distance to performance bound", row=row, col=col_i)
        fig.update_xaxes(title_text=label, row=row, col=col_i)

    fig.update_layout(
        title=f"Morphology ({title_suffix}) vs Distance to Performance Bound",
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
    df["bandwidth_podos"] = df["f_max_podos"] - df["f_min_podos"]

    # df_filtered = df[df["trill_rate"] >= 2]
    df_filtered = df.copy()

    ub_df, reg = upper_bound_regression(
        df_filtered,
        x_col="trill_rate",
        y_col="bandwidth",
        bin_width=2.0,
        x_min=2,
        log_y=False
    )

    ub_df_podos, reg_podos = upper_bound_regression(
        df_filtered,
        x_col="trill_rate",
        y_col="bandwidth_podos",
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

    df_filtered["dist_to_bound_podos"] = distance_to_upper_bound(
        df_filtered,
        reg_podos,
        x_col="trill_rate",
        y_col="bandwidth_podos",
        signed=True
    )

    regression_results = {
        "reg": reg,
        "ub_df": ub_df,
        "reg_podos": reg_podos,
        "ub_df_podos": ub_df_podos
    }

    return df_filtered, regression_results

def load_meta_and_morpho(DATA_DIR, file_timestamps, file_meta, file_morpho):

    df_timestamps = pd.read_csv(os.path.join(DATA_DIR, file_timestamps))
    df_metadata = pd.read_csv(os.path.join(DATA_DIR, file_meta))
    df_morpho = pd.read_csv(os.path.join(DATA_DIR, file_morpho))
    df_morpho["logmass"] = np.log(df_morpho["mass"])

    df_timestamps["file_name"] = df_timestamps.apply(
        lambda r: r["file_name"].rsplit(".", 1)[0] + f"_seg{r['syllable_rank']}.wav",
        axis=1
    )

    # df_metadata["file_name_radical"] = df_metadata["file_name"].apply(
    #     lambda x: re.sub(r"_seg\d+\.wav$", ".wav", x)
    # )

    df_metadata_sub = df_metadata[
        ['gen', 'family', 'species', 'sub_species', 'common_name', 'recordist', 'date', 'time',
            'country', 'location', 'lat', 'lng', 'bird', 'file_name',
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
    file_morpho = "model_traits_morpho_social_data.csv"

    df_pred = pd.read_csv(os.path.join(config.data_processed_subdir, pred_annotation_file))

    df_pred_filtered, regression_results = prepare_dataset(df_pred)
    
    df_merged = load_meta_and_morpho(config.data_raw_subdir, file_timestamps, file_meta, file_morpho)
    df_merged = pd.merge(df_pred_filtered, df_merged, on="file_name", how="inner")

    print("\n" + "="*70)
    print("WORKFLOW OPTIONS")
    print("="*70)
    print("1. Plot virtuosity space and upper bound regression")
    print("2. Plot viruosity against morphology metrics")
    print("3. Visualize top virtuosity samples and confirm predictions")
    print("4. Export final dataset")
    print("5. Launch GUI visualizer")

    choice = input("\nChoice (1-5): ").strip()

    if choice == "1":
        ub_df = regression_results["ub_df"]
        reg = regression_results["reg"]
        show_bound = False

        input_log_y = input("Log-transform bandwidth for plotting? (y/n): ").strip().lower()
        log_y = input_log_y == "y"

        if not log_y:
            input_use_podos = input("Use Podos bandwidth for plotting? (y/n): ").strip().lower()
            use_podos = input_use_podos == "y"
            if use_podos:
                ub_df = regression_results["ub_df_podos"]
                reg = regression_results["reg_podos"]

            input_show_bound = input("Show upper bound regression on plot? (y/n): ").strip().lower()
            show_bound = input_show_bound == "y"

        plot_virtuosity(
            df_merged,
            x_col="trill_rate",
            y_col="bandwidth" if not use_podos else "bandwidth_podos",
            logy=log_y,
            # hue_col="logmass",
            upper_bound=ub_df if show_bound else None,
            reg=reg if show_bound else None
        )
    
    elif choice == "2":

        plot_dist_vs_traits_plotly(
            df_merged,
            hue_col="family",
            title_suffix="colored by family"
        )

    elif choice == "3":
        visualize_best_virtuosity_samples(
            df_merged,
            config,
            sorting_metric="dist_to_bound",
            bin_width=5.0,
            top_n=5,
            plot=True
        )
        
    elif choice == "4":
        df_out = df_merged.copy()
        output_final = os.path.join(config.data_processed_subdir, "final_virtuosity_dataset_v3.csv")
        os.makedirs(os.path.dirname(output_final), exist_ok=True)
        df_out.to_csv(output_final, index=False)

    elif choice == "5":
        launch_gui(config, df_merged, metric_col = "dist_to_bound")

    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()