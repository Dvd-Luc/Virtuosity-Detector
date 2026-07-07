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
from src.utils.metrics import performance_orientation, upper_bound_regression, quantile_upper_bound, distance_to_upper_bound, residuals_to_upper_bound, bill_centroid

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
        plot_virtuosity(best_samples, x_col=x_col, y_col=y_col, logy=False, show_upper_bound=False)

    visualize_and_confirm_predictions(config, df_predictions=best_samples)

def plot_virtuosity(df, x_col="trill_rate", y_col="bandwidth", logy=False, hue_col=None, show_upper_bound=False, upper_bound=None, reg=None):
    df_plot = df.dropna(subset=[x_col, y_col])
    df_plot = df_plot[df_plot[x_col] >= 2]

    if logy:
        df_plot["log_bandwidth"] = np.log(df_plot[y_col] + 1e-6)
        ylabel = f'log({y_col}) (log(Hz))'
        y_col = "log_bandwidth"
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

    # if show_upper_bound :
    #     ub, reg = upper_bound_regression(
    #         df_plot,
    #         x_col=x_col,
    #         y_col=y_col,
    #         log_y=logy)

    #     fig.add_trace(
    #         go.Scatter(
    #             x=ub[x_col],
    #             y=ub[y_col],
    #             mode="markers",
    #             marker=dict(color="red", size=8),
    #             name="Upper bound points",
    #         )
    #     )

    #     x_line = np.linspace(ub[x_col].min(), ub[x_col].max(), 200)
    #     if logy:
    #         y_line = np.exp(reg["intercept"] + reg["slope"] * x_line)
    #     else:
    #         y_line = reg["intercept"] + reg["slope"] * x_line

    #     fig.add_trace(
    #         go.Scatter(
    #             x=x_line,
    #             y=y_line,
    #             mode="lines",
    #             line=dict(color="red", dash="dash"),
    #             name=f"Upper-bound regression: y = {reg['intercept']:.2f} + {reg['slope']:.2f}*x (R={reg['r_value']:.2f}, p={reg['p_value']:.3e})",
    #         )
    #     )

    if show_upper_bound:
        reg = quantile_upper_bound(
            df_plot,
            x_col=x_col,
            y_col=y_col,
            log_y=logy
        )

        x_line = np.linspace(df_plot[x_col].min(), df_plot[x_col].max(), 200)
        if logy:
            y_line = np.exp(reg["intercept"] + reg["slope"] * x_line)
        else:
            y_line = reg["intercept"] + reg["slope"] * x_line

        # Label adapts to whichever method was used
        if "r_value" in reg:
            label = (
                f"Upper-bound regression: y = {reg['intercept']:.2f} + {reg['slope']:.2f}*x"
                f" (R={reg['r_value']:.2f}, p={reg['p_value']:.3e})"
            )
        else:
            label = (
                f"Quantile regression (q={reg['quantile']}): y = {reg['intercept']:.2f} + {reg['slope']:.2f}*x"
                f" (pseudo-R²={reg['pseudo_r2']:.2f}, p={reg['p_value']:.3e})"
            )

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color="red", dash="dash"),
                name=label,
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

    
def prepare_dataset(df):
    # df["file_name"] = df.apply(
    #     lambda r: f"{r['file_name_radical'].split('.')[0]}_seg{r['segment_id']}.wav",
    #     axis=1
    # )

    df["trill_duration"] = df["t_max"] - df["t_min"]
    df["trill_rate"] = df.apply(
        lambda row: 0 if row["trill_duration"] == 0 else row["count"] / row["trill_duration"],
        axis=1
    )
    df["trill_rate_standardized"] = (df["trill_rate"] - df["trill_rate"].mean()) / df["trill_rate"].std()

    df["bandwidth"] = df["f_max"] - df["f_min"]
    df["bandwidth_standardized"] = (df["bandwidth"] - df["bandwidth"].mean()) / df["bandwidth"].std()
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

    quantile_reg = quantile_upper_bound(
        df_filtered,
        x_col="trill_rate",
        y_col="bandwidth",
        quantile=0.90,
        x_min=2,
        log_y=False
    )

    df_filtered["dist_to_quantile_bound"] = distance_to_upper_bound(
        df_filtered,
        quantile_reg,
        x_col="trill_rate",
        y_col="bandwidth",
        signed=True
    )

    df_filtered["residual_to_quantile_bound_y"], df_filtered["residual_to_quantile_bound_x"] = residuals_to_upper_bound(
        df_filtered,
        quantile_reg,
        x_col="trill_rate",
        y_col="bandwidth"
    )

    # --- Per-family upper bound ---

    def get_adaptive_bin_width(group, x_col="trill_rate", n_bins=10, min_bin_width=0.5):
        x_range = group[x_col].max() - group[x_col].min()
        bin_width = x_range / n_bins
        return max(bin_width, min_bin_width)

    family_regs = {}
    family_regs_quantile = {}
    dist_family_list = []
    dist_family_quantiles_list = []
    residuals_family_quantiles_list_x = []
    residuals_family_quantiles_list_y = []

    for family, group in df_filtered.groupby("family"):
        print(f"Processing family {family} with {len(group)} samples...")
        if len(group) < 10:  # skip families with too few observations
            dist_family_list.append(pd.Series(np.nan, index=group.index))
            dist_family_quantiles_list.append(pd.Series(np.nan, index=group.index))
            continue
        try:
            # Orthogonal regression to upper bound
            bin_width_fam = get_adaptive_bin_width(group, x_col="trill_rate", n_bins=10)
            _, reg_fam = upper_bound_regression(
                group, x_col="trill_rate", y_col="bandwidth",
                bin_width=bin_width_fam, x_min=2, log_y=False
            )
            family_regs[family] = reg_fam
            dists = distance_to_upper_bound(
                group, reg_fam, x_col="trill_rate", y_col="bandwidth", signed=True
            )
            dist_family_list.append(pd.Series(dists, index=group.index))

            # Quantile regression to upper bound
            reg_fam_quantile = quantile_upper_bound(
                group, x_col="trill_rate", y_col="bandwidth",
                quantile=0.90, x_min=2, log_y=False
            )
            family_regs_quantile[family] = reg_fam_quantile
            dists_quantile = distance_to_upper_bound(
                group, reg_fam_quantile, x_col="trill_rate", y_col="bandwidth", signed=True
            )
            dist_family_quantiles_list.append(pd.Series(dists_quantile, index=group.index))

            # Residuals to quantile bound
            res_y, res_x = residuals_to_upper_bound(
                group,
                reg_fam_quantile,
                x_col="trill_rate",
                y_col="bandwidth"
            )
            residuals_family_quantiles_list_y.append(pd.Series(res_y, index=group.index))
            residuals_family_quantiles_list_x.append(pd.Series(res_x, index=group.index))

        except Exception as e:
            print(f"Warning: could not fit upper bound for family {family}: {e}")
            dist_family_list.append(pd.Series(np.nan, index=group.index))
            dist_family_quantiles_list.append(pd.Series(np.nan, index=group.index))
            residuals_family_quantiles_list_y.append(pd.Series(np.nan, index=group.index))
            residuals_family_quantiles_list_x.append(pd.Series(np.nan, index=group.index))

    df_filtered["dist_to_bound_family"] = pd.concat(dist_family_list).reindex(df_filtered.index)
    df_filtered["dist_to_quantile_bound_family"] = pd.concat(dist_family_quantiles_list).reindex(df_filtered.index)
    df_filtered["residual_to_quantile_bound_y_family"] = pd.concat(residuals_family_quantiles_list_y).reindex(df_filtered.index)
    df_filtered["residual_to_quantile_bound_x_family"] = pd.concat(residuals_family_quantiles_list_x).reindex(df_filtered.index)

    # --- Vocal deviation metrics (global bound) ---
    vocal_deviation_metrics = performance_orientation(df_filtered, reg, x_col="trill_rate", y_col="bandwidth", log_y=False)
    df_filtered = pd.concat([df_filtered, vocal_deviation_metrics], axis=1)

    regression_results = {
        "reg": reg,
        "ub_df": ub_df,
        "reg_podos": reg_podos,
        "ub_df_podos": ub_df_podos,
        "quantile_reg": quantile_reg,
        "family_regs": family_regs,
        "family_regs_quantile": family_regs_quantile,
    }

    return df_filtered, regression_results

def load_meta_and_morpho(DATA_DIR, file_timestamps, file_meta, file_morpho):

    df_timestamps = pd.read_csv(os.path.join(DATA_DIR, file_timestamps))
    df_metadata = pd.read_csv(os.path.join(DATA_DIR, file_meta))
    df_morpho = pd.read_csv(os.path.join(DATA_DIR, file_morpho))


    df_morpho["logmass"] = np.log(df_morpho["mass"])
    df_morpho["log_beak_length"] = np.log(df_morpho["Beak.Length_Culmen"])
    df_morpho["log_beak_width"] = np.log(df_morpho["Beak.Width"])
    df_morpho["log_beak_depth"] = np.log(df_morpho["Beak.Depth"])

    df_morpho["bill_centroid"], df_morpho["log_bill_centroid"] = bill_centroid(df_morpho)
    df_morpho["log_bill_centroid_over_logmass"] = df_morpho["log_bill_centroid"] / df_morpho["logmass"]

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
    if config.dataset_csv is not None:
        df_merged = pd.read_csv(os.path.join(config.data_processed_subdir, config.dataset_csv))
    else:
        pred_annotation_file = "trills_11032026_predictions.csv"#"annotations_trills_v2_tests_predictions.csv"
        file_timestamps = "segments_passerines_filtered.csv"
        file_meta = "traits_data_pc_gmm_8components_proba_filtered.csv"
        file_morpho = "model_traits_morpho_social_data.csv"

        df_pred = pd.read_csv(os.path.join(config.data_processed_subdir, pred_annotation_file))
        df_pred["file_name"] = df_pred.apply(
            lambda r: f"{r['file_name_radical'].split('.')[0]}_seg{r['segment_id']}.wav",
            axis=1
        )

        df_merged = load_meta_and_morpho(config.data_raw_subdir, file_timestamps, file_meta, file_morpho)
        df_merged = pd.merge(df_pred, df_merged, on="file_name", how="inner")

        df_merged, regression_results = prepare_dataset(df_merged)
        
        

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
        use_podos = False

        input_log_y = input("Log-transform bandwidth for plotting? (y/n): ").strip().lower()
        log_y = input_log_y == "y"

        input_use_podos = input("Use Podos bandwidth for plotting? (y/n): ").strip().lower()
        use_podos = input_use_podos == "y"

        input_show_bound = input("Show upper bound regression on plot? (y/n): ").strip().lower()
        show_bound = input_show_bound == "y"

        plot_virtuosity(
            df_merged,
            x_col="trill_rate",
            y_col="bandwidth" if not use_podos else "bandwidth_podos",
            logy=log_y,
            # hue_col="logmass",
            show_upper_bound=show_bound,
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
        output_final = os.path.join(config.data_processed_subdir, "full_dataset_prediction_quantiles_v2.csv")
        os.makedirs(os.path.dirname(output_final), exist_ok=True)
        df_out.to_csv(output_final, index=False)

    elif choice == "5":
        launch_gui(config, df_merged, metric_col = "dist_to_bound")

    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()