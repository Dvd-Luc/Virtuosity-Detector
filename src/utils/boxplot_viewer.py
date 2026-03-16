"""
boxplot_viewer.py
Generate boxplots from a CSV file for specified columns.

Usage
-----
    python boxplot_viewer.py --cols trill_rate bandwidth confidence
    python boxplot_viewer.py --cols trill_rate bandwidth --group species
    python boxplot_viewer.py --cols trill_rate --group species --theme light
"""

import argparse
import sys
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def pick_file() -> str:
    """Open a file dialog and return the selected CSV path."""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    if not path:
        print("No file selected — exiting.")
        sys.exit(0)
    return path


def make_boxplots(df: pd.DataFrame, cols: list[str],
                  group: str | None, theme: str) -> go.Figure:
    """
    Build a figure with one boxplot per column in *cols*.
    If *group* is provided, boxes are colored by that column.
    """
    dark     = theme == "dark"
    template = "plotly_dark"  if dark else "plotly_white"
    paper_bg = "#1e1e2e"      if dark else "#f5f5f5"
    plot_bg  = "#181825"      if dark else "#ffffff"
    palette  = px.colors.qualitative.Pastel

    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"Warning: columns not found and skipped: {missing}")
        cols = [c for c in cols if c in df.columns]

    if not cols:
        print("No valid columns to plot — exiting.")
        sys.exit(1)

    if group and group not in df.columns:
        print(f"Warning: group column '{group}' not found — ignoring.")
        group = None

    n_cols   = min(len(cols), 3)
    n_rows   = (len(cols) + n_cols - 1) // n_cols
    fig      = make_subplots(rows=n_rows, cols=n_cols,
                             subplot_titles=cols)

    group_vals = sorted(df[group].dropna().unique()) if group else [None]

    for idx, col in enumerate(cols):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1

        for g_idx, grp in enumerate(group_vals):
            subset = df[df[group] == grp] if grp is not None else df
            data   = subset[col].dropna()
            color  = palette[g_idx % len(palette)]

            fig.add_trace(
                go.Box(
                    y=data,
                    name=str(grp) if grp is not None else col,
                    marker_color=color,
                    boxmean="sd",           # shows mean + std dev marker
                    legendgroup=str(grp),
                    showlegend=(idx == 0),  # legend only on first subplot
                    hovertemplate=(
                        f"<b>{col}</b><br>"
                        + (f"{group}: {grp}<br>" if grp else "")
                        + "value: %{y:.4f}<extra></extra>"
                    ),
                ),
                row=row, col=col_pos,
            )

        fig.update_yaxes(title_text=col,
                         title_font=dict(size=13),
                         tickfont=dict(size=11),
                         row=row, col=col_pos)
        if group:
            fig.update_xaxes(title_text=group,
                             tickfont=dict(size=11),
                             row=row, col=col_pos)

    title = "Boxplots — " + ", ".join(cols)
    if group:
        title += f"  ·  grouped by {group}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        template=template,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(size=13),
        boxmode="group",
        legend=dict(
            title=dict(text=group or "", font=dict(size=13)),
            font=dict(size=12),
        ),
        height=600 * n_rows,
    )

    return fig

def make_dashboard(df, group, scatter_color, theme, filename):
    dark     = theme == "dark"
    template = "plotly_dark"  if dark else "plotly_white"
    paper_bg = "#1e1e2e"      if dark else "#f5f5f5"
    plot_bg  = "#181825"      if dark else "#ffffff"
    palette  = px.colors.qualitative.Pastel

    color_vals = sorted(df[scatter_color].dropna().unique()) if scatter_color in df.columns else [None]
    group_vals = sorted(df[group].dropna().unique()) if group else [None]
    color_map  = {v: palette[i % len(palette)] for i, v in enumerate(color_vals)}

    gap   = 0.06   # espace entre subplots
    top   = 1   # y max
    mid   = 0.50   # séparation ligne 1 / ligne 2
    bot   = 0   # y min

    # Domaines x pour 2 colonnes (ligne 1) et 3 colonnes (ligne 2)
    row1_x = [
    [0.0,          2/3 - gap/2],   # scatter : 2/3
    [2/3 + gap/2,  1.0        ],   # histogramme : 1/3
]
    row2_x = [
        [0.0,                         1/3 - gap / 2],
        [1/3 + gap / 2,               2/3 - gap / 2],
        [2/3 + gap / 2,               1.0           ],
    ]
    row1_y = [mid + gap, top]
    row2_y = [bot,       mid - gap]

    fig = go.Figure()

    # ── Row 1: scatter plots ─────────────────────────────────────────────────

    scatter_defs = [
        ("trill_rate",    "bandwidth",    "x",  "y",  0),
    ]

    for x_col, y_col, xref, yref, pos in scatter_defs:
        shown = set()
        for val in color_vals:
            subset = df[df[scatter_color] == val] if val is not None else df
            subset = subset.dropna(subset=[x_col, y_col])
            color  = color_map.get(val, palette[0])
            label  = str(val) if val is not None else scatter_color
            fig.add_trace(go.Scatter(
                x=subset[x_col], y=subset[y_col],
                mode="markers",
                marker=dict(color=color, size=5, opacity=0.6, line=dict(width=0)),
                name=label,
                legendgroup=label,
                showlegend=(label not in shown and pos == 0),
                xaxis=xref, yaxis=yref,
                customdata=subset[["file_name_radical", "segment_id"]].values,
                hovertemplate=(
                    f"{x_col}: %{{x:.3f}}<br>"
                    f"{y_col}: %{{y:.3f}}<br>"
                    "%{customdata[0]} seg%{customdata[1]}"
                    f"<extra>{label}</extra>"
                ),
            ))
            shown.add(label)
        
    # ── Row 1 right: count bar chart by group ───────────────────────────────
    if group and group in df.columns:
        counts = df[group].value_counts().sort_index()
        for g_idx, grp in enumerate(counts.index):
            color = palette[g_idx % len(palette)]
            fig.add_trace(go.Bar(
                x=[str(grp)],
                y=[counts[grp]],
                name=str(grp),
                legendgroup=str(grp),
                showlegend=False,
                marker_color=color,
                xaxis="x2", yaxis="y2",
                hovertemplate=f"{group}: {grp}<br>n=%{{y}}<extra></extra>",
            ))

    # ── Row 2: boxplots ──────────────────────────────────────────────────────

    box_axes = [
        ("trill_rate",    "x3", "y3"),
        ("bandwidth",     "x4", "y4"),
        ("dist_to_bound", "x5", "y5"),
    ]

    for col, xref, yref in box_axes:
        for g_idx, grp in enumerate(group_vals):
            subset = df[df[group] == grp] if grp is not None else df
            color  = palette[g_idx % len(palette)]
            label  = str(grp) if grp is not None else col
            fig.add_trace(go.Box(
                y=subset[col].dropna(),
                name=label,
                marker_color=color,
                boxmean="sd",
                legendgroup=f"box_{label}",
                showlegend=False,
                xaxis=xref, yaxis=yref,
                hovertemplate=f"<b>{col}</b><br>{label}<br>%{{y:.4f}}<extra></extra>",
            ))

    # ── Axes layout ──────────────────────────────────────────────────────────

    axis_style = dict(
        showgrid=True,
        gridcolor="#313244" if dark else "#e8e8e8",
        zerolinecolor="#313244" if dark else "#cccccc",
        tickfont=dict(size=11),
        title_font=dict(size=13),
        # bgcolor=plot_bg,
    )

    # Scatter axes
    for i, (x_col, y_col, _, _, pos) in enumerate(scatter_defs):
        n = "" if i == 0 else str(i + 1)
        fig.update_layout(**{
            f"xaxis{n}": dict(**axis_style,
                              title=x_col,
                              domain=row1_x[pos],
                              anchor=f"y{n}"),
            f"yaxis{n}": dict(**axis_style,
                              title=y_col,
                              domain=row1_y,
                              anchor=f"x{n}"),
        })

        fig.update_layout(
            xaxis2=dict(**axis_style,
                        title=group or "",
                        domain=row1_x[1],
                        anchor="y2",
                        tickangle=-45),
            yaxis2=dict(**axis_style,
                        title="Count",
                        domain=row1_y,
                        anchor="x2"),
        )

    # Box axes
    for i, (col, _, _) in enumerate(box_axes):
        n = str(i + 3)
        fig.update_layout(**{
            f"xaxis{n}": dict(**axis_style,
                              domain=row2_x[i],
                              anchor=f"y{n}",
                              title=group),
            f"yaxis{n}": dict(**axis_style,
                              title=col,
                              domain=row2_y,
                              anchor=f"x{n}"),
        })

    # Annotations comme titres de subplots
    # annotations = [
    #     dict(text="trill_rate vs bandwidth",    x=sum(row1_x[0])/2, y=top + 0.01,
    #          xref="x domain", yref="y domain", showarrow=False, font=dict(size=13)),
    #     dict(text="dist_to_bound vs trill_rate", x=sum(row1_x[1])/2, y=top + 0.01,
    #          xref="paper", yref="paper", showarrow=False, font=dict(size=13)),
    #     dict(text="trill_rate",    x=sum(row2_x[0])/2, y=mid + 0.01,
    #          xref="paper", yref="paper", showarrow=False, font=dict(size=13)),
    #     dict(text="bandwidth",     x=sum(row2_x[1])/2, y=mid + 0.01,
    #          xref="paper", yref="paper", showarrow=False, font=dict(size=13)),
    #     dict(text="dist_to_bound", x=sum(row2_x[2])/2, y=mid + 0.01,
    #          xref="paper", yref="paper", showarrow=False, font=dict(size=13)),
    # ]

    fig.update_layout(
        template=template,
        paper_bgcolor=paper_bg,
        font=dict(size=13, color="#cdd6f4" if dark else "#333"),
        title=dict(
            text=f"Dashboard — colored by {scatter_color}"
                 + (f"  ·  boxes grouped by {group}" if group else "") + f"  ·  {filename}",
            font=dict(size=18),
        ),
        legend=dict(title=dict(text=scatter_color), font=dict(size=12)),
        boxmode="group",
        barmode="group",
        height=950,
        # annotations=annotations,
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate boxplots from a CSV file.")
    parser.add_argument("--cols",  nargs="+", required=True,
                        help="Column names to plot")
    parser.add_argument("--group", default=None,
                        help="Column to group / color boxes by (e.g. species)")
    parser.add_argument("--theme", default="dark", choices=["dark", "light"],
                        help="Plot theme (default: dark)")
    parser.add_argument("--out",   default=None,
                        help="Optional output HTML path (default: open in browser)")
    parser.add_argument("--scatter", default=None,
                    help="Column to color scatter plots by (e.g. species)")
    args = parser.parse_args()

    path = pick_file()
    print(f"Loading: {path}")
    df   = pd.read_csv(path)
    print(f"{len(df)} rows, {len(df.columns)} columns")
    filename = path.split("/")[-1]

    if args.scatter:
        fig = make_dashboard(df, args.group, args.scatter, args.theme, filename)
    else:
        fig = make_boxplots(df, args.cols, args.group, args.theme)

    if args.out:
        fig.write_html(args.out)
        print(f"Saved → {args.out}")
    else:
        fig.show()


if __name__ == "__main__":
    main()