"""
GUI Visualizer for Trill Predictions Spectrograms
3 navigation modes:
  1. Taxonomy filter (species/genus/family) + stats + checkboxes
  2. Metric lists (Top N / threshold) + arrow/dropdown browser
  3. Plotly scatter (bandwidth vs trill rate) + click-to-spectrogram
"""

import os, types, tempfile, webbrowser
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotly.express as px


# ─────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────

def load_spectrogram(row, config):
    audio_path = os.path.join(config.audio_subdir, str(row["file_name_radical"]))
    y, sr = librosa.load(audio_path, sr=None)
    seg_start = float(row.get("seg_start", 0))
    seg_end   = float(row.get("seg_end", librosa.get_duration(y=y, sr=sr)))
    y_seg = y[int(seg_start * sr): int(seg_end * sr)]
    n_fft, hop = config.n_fft, config.hop_length
    S    = np.abs(librosa.stft(y_seg, n_fft=n_fft, hop_length=hop))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    return S_db, times, freqs, row.get("f_min"), row.get("f_max"), row.get("t_min"), row.get("t_max")


def draw_spectrogram(ax, S_db, times, freqs,
                     f_min=None, f_max=None, t_min=None, t_max=None, title="", crop_to_bounds=True):
    BG2 = "#181825"
    ax.clear()
    ax.set_facecolor(BG2)
    ax.pcolormesh(times, freqs, S_db, shading="auto", cmap="magma")
    ax.set_xlabel("Time (s)",       color="#a6adc8")
    ax.set_ylabel("Frequency (Hz)", color="#a6adc8")
    ax.set_title(title, fontsize=9, color="#cdd6f4")
    ax.tick_params(colors="#a6adc8")
    for sp in ax.spines.values():
        sp.set_edgecolor("#313244")
    if crop_to_bounds and f_min is not None and f_max is not None and not (np.isnan(f_min) or np.isnan(f_max)):
        ax.set_ylim(max(0, f_min - 500), f_max + 500)
    if all(v is not None and not np.isnan(float(v)) for v in [t_min, t_max, f_min, f_max]):
        ax.add_patch(plt.Rectangle(
            (t_min, f_min), t_max - t_min, f_max - f_min,
            lw=1.5, edgecolor="cyan", facecolor="none", linestyle="--"))


# ════════════════════════════════════════════════════════
# Main App
# ════════════════════════════════════════════════════════

class SpectroViewer(tk.Tk):
    BG   = "#1e1e2e";  BG2  = "#181825";  SURF = "#313244"
    FG   = "#cdd6f4";  FG2  = "#a6adc8";  ACC  = "#89b4fa";  GRN  = "#a6e3a1"

    def __init__(self, df, config, metric_col="confidence"):
        super().__init__()
        self.df, self.config, self.metric_col = df.copy(), config, metric_col
        self.title("Trill Spectrogram Viewer")
        self.geometry("1250x840")
        self.configure(bg=self.BG)
        self._style(); self._header(); self._notebook()

    # ── Styling ──────────────────────────────────────────────────────────────
    def _style(self):
        s = ttk.Style(self); s.theme_use("clam")
        s.configure("TNotebook", background=self.BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=self.SURF, foreground=self.FG,
                    padding=[14, 7], font=("Helvetica", 10))
        s.map("TNotebook.Tab",
              background=[("selected", self.ACC)],
              foreground=[("selected", self.BG2)])
        s.configure("TCombobox", fieldbackground=self.SURF, background=self.SURF,
                    foreground=self.FG, arrowcolor=self.FG)

    # ── Header ───────────────────────────────────────────────────────────────
    def _header(self):
        h = tk.Frame(self, bg=self.BG2, pady=8); h.pack(fill="x")
        tk.Label(h, text="🎵 Trill Spectrogram Viewer",
                 font=("Helvetica", 15, "bold"), fg=self.FG, bg=self.BG2).pack(side="left", padx=16)
        tk.Label(h, text="Metric:", fg=self.FG2, bg=self.BG2).pack(side="left", padx=(24, 4))
        self.metric_var = tk.StringVar(value=self.metric_col)
        num_cols = [c for c in self.df.select_dtypes(include=np.number).columns]
        cb = ttk.Combobox(h, textvariable=self.metric_var, values=num_cols, width=20)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda _: self._metric_changed())

        tk.Label(h, text="Spectro view:", fg=self.FG2, bg=self.BG2).pack(side="left", padx=(24, 4))
        self.spectro_mode = tk.StringVar(value="crop")
        for val, txt in [("crop", "Cropped ± 500 Hz"), ("full", "Full segment")]:
            tk.Radiobutton(h, text=txt, variable=self.spectro_mode, value=val,
                        fg=self.FG, bg=self.BG2, selectcolor=self.SURF,
                        activebackground=self.BG2).pack(side="left", padx=4)
        
        tk.Label(h, text=f"{len(self.df)} samples", fg=self.GRN, bg=self.BG2,
                 font=("Helvetica", 9)).pack(side="right", padx=16)

    def _metric_changed(self):
        self.metric_col = self.metric_var.get()
        self._opt1_update_stats()

    # ── Notebook ─────────────────────────────────────────────────────────────
    def _notebook(self):
        self.nb = ttk.Notebook(self); self.nb.pack(fill="both", expand=True, padx=10, pady=8)
        self.t1 = tk.Frame(self.nb, bg=self.BG)
        self.t2 = tk.Frame(self.nb, bg=self.BG)
        self.t3 = tk.Frame(self.nb, bg=self.BG)
        self.nb.add(self.t1, text="Taxonomy Filter")
        self.nb.add(self.t2, text="Metric Lists")
        self.nb.add(self.t3, text="Scatter Plot")
        self._tab1(); self._tab2(); self._tab3()

    # ════════════════════════════════════════════
    # TAB 1
    # ════════════════════════════════════════════
    def _tab1(self):
        top = tk.Frame(self.t1, bg=self.BG); top.pack(fill="x", padx=14, pady=10)
        tk.Label(top, text="Filter by:", fg=self.FG2, bg=self.BG).pack(side="left")
        self.tax_level = tk.StringVar(value="species")
        for lvl in ["species", "gen", "family"]:
            tk.Radiobutton(top, text=lvl, variable=self.tax_level, value=lvl,
                           command=self._opt1_load_values,
                           fg=self.FG, bg=self.BG, selectcolor=self.SURF,
                           activebackground=self.BG).pack(side="left", padx=8)
        tk.Label(top, text="Value:", fg=self.FG2, bg=self.BG).pack(side="left", padx=(20, 4))
        self.tax_val = tk.StringVar()
        self.tax_cb  = ttk.Combobox(top, textvariable=self.tax_val, width=28); self.tax_cb.pack(side="left")
        self.tax_cb.bind("<<ComboboxSelected>>", lambda _: self._opt1_update_stats())

        # Stats panel
        lf = tk.LabelFrame(self.t1, text=" Statistics ", fg=self.ACC, bg=self.BG,
                            font=("Helvetica", 10, "bold"))
        lf.pack(fill="x", padx=14, pady=4)
        inner = tk.Frame(lf, bg=self.BG); inner.pack(pady=8)
        self.sl = {}; self.sv = {}
        for i, stat in enumerate(["count", "min", "max", "mean", "median"]):
            f = tk.Frame(inner, bg=self.BG, padx=22); f.grid(row=0, column=i)
            tk.Label(f, text=stat.upper(), fg=self.ACC, bg=self.BG,
                     font=("Helvetica", 9, "bold")).pack()
            lbl = tk.Label(f, text="—", fg=self.FG, bg=self.BG, font=("Helvetica", 13, "bold"))
            lbl.pack(); self.sl[stat] = lbl
            if stat != "count":
                v = tk.BooleanVar(); self.sv[stat] = v
                tk.Checkbutton(f, text="view spectro", variable=v, fg=self.FG2, bg=self.BG,
                               selectcolor=self.SURF, activebackground=self.BG,
                               command=lambda s=stat: self._opt1_view(s)).pack()

        self.fig1, self.ax1 = plt.subplots(figsize=(10, 3.4), facecolor=self.BG2)
        self.ax1.set_facecolor(self.BG2)
        c = FigureCanvasTkAgg(self.fig1, master=self.t1); c.get_tk_widget().pack(fill="both", expand=True, padx=14, pady=6)
        self.c1 = c
        self._opt1_load_values()

    def _opt1_load_values(self):
        col = self.tax_level.get()
        if col in self.df.columns:
            vals = sorted(self.df[col].dropna().unique().tolist())
            self.tax_cb["values"] = vals
            if vals: self.tax_val.set(vals[0]); self._opt1_update_stats()

    def _opt1_sub(self):
        return self.df[self.df[self.tax_level.get()] == self.tax_val.get()]

    def _opt1_update_stats(self):
        sub = self._opt1_sub(); m = self.metric_col
        for v in self.sv.values(): v.set(False)
        if m not in sub.columns or sub.empty:
            for s in self.sl: self.sl[s].config(text="—"); return
        vals = sub[m].dropna()
        self.sl["count"].config(text=str(len(vals)))
        self.sl["min"].config(text=f"{vals.min():.4f}")
        self.sl["max"].config(text=f"{vals.max():.4f}")
        self.sl["mean"].config(text=f"{vals.mean():.4f}")
        self.sl["median"].config(text=f"{vals.median():.4f}")

    def _opt1_view(self, stat):
        for s, v in self.sv.items():
            if s != stat: v.set(False)
        if not self.sv[stat].get(): return
        sub = self._opt1_sub(); m = self.metric_col
        vals = sub[m].dropna()
        idx = {"min": vals.idxmin, "max": vals.idxmax,
               "mean": lambda: (vals - vals.mean()).abs().idxmin(),
               "median": lambda: (vals - vals.median()).abs().idxmin()}[stat]()
        self._show(self.ax1, self.c1, sub.loc[idx])

    # ════════════════════════════════════════════
    # TAB 2
    # ════════════════════════════════════════════
    def _tab2(self):
        ctrl = tk.Frame(self.t2, bg=self.BG); ctrl.pack(fill="x", padx=14, pady=10)
        tk.Label(ctrl, text="Criterion:", fg=self.FG2, bg=self.BG).pack(side="left")
        self.crit = tk.StringVar(value="top_n")
        for v, t in [("top_n", "Top N highest"), ("threshold", "Threshold filter")]:
            tk.Radiobutton(ctrl, text=t, variable=self.crit, value=v,
                           command=self._t2_toggle,
                           fg=self.FG, bg=self.BG, selectcolor=self.SURF,
                           activebackground=self.BG).pack(side="left", padx=8)
        # Top-N
        self.tnf = tk.Frame(ctrl, bg=self.BG); self.tnf.pack(side="left", padx=10)
        tk.Label(self.tnf, text="N:", fg=self.FG2, bg=self.BG).pack(side="left")
        self.nvar = tk.IntVar(value=10)
        tk.Spinbox(self.tnf, from_=1, to=1000, textvariable=self.nvar, width=5,
                   bg=self.SURF, fg=self.FG, insertbackground="white",
                   buttonbackground=self.SURF).pack(side="left")
        # Threshold
        self.thf = tk.Frame(ctrl, bg=self.BG)
        self.opv = tk.StringVar(value=">")
        ttk.Combobox(self.thf, textvariable=self.opv, values=[">","<",">=","<="], width=4).pack(side="left")
        self.tv = tk.DoubleVar(value=0.5)
        tk.Entry(self.thf, textvariable=self.tv, width=8,
                 bg=self.SURF, fg=self.FG, insertbackground="white").pack(side="left", padx=4)
        tk.Button(ctrl, text="Build List ▶", bg=self.ACC, fg=self.BG2,
                  font=("Helvetica", 9, "bold"), relief="flat", padx=10,
                  command=self._t2_build).pack(side="left", padx=14)
        self.li_lbl = tk.Label(ctrl, text="No list yet", fg=self.FG2, bg=self.BG)
        self.li_lbl.pack(side="left")
        # Nav
        nav = tk.Frame(self.t2, bg=self.BG2, pady=6); nav.pack(fill="x", padx=14)
        bcfg = dict(bg=self.SURF, fg=self.FG, relief="flat", padx=14, pady=4,
                    font=("Helvetica", 12, "bold"), activebackground=self.ACC, activeforeground=self.BG2)
        tk.Button(nav, text="◀", command=self._t2_prev, **bcfg).pack(side="left", padx=4)
        tk.Button(nav, text="▶", command=self._t2_next, **bcfg).pack(side="left", padx=4)
        tk.Label(nav, text="Jump:", fg=self.FG2, bg=self.BG2).pack(side="left", padx=(16, 4))
        self.jv = tk.StringVar()
        self.jcb = ttk.Combobox(nav, textvariable=self.jv, width=50); self.jcb.pack(side="left")
        self.jcb.bind("<<ComboboxSelected>>", lambda _: self._t2_jump())
        self.nav_lbl = tk.Label(nav, text="", fg=self.ACC, bg=self.BG2, font=("Helvetica", 9))
        self.nav_lbl.pack(side="left", padx=10)
        self.fig2, self.ax2 = plt.subplots(figsize=(10, 3.4), facecolor=self.BG2)
        self.ax2.set_facecolor(self.BG2)
        c = FigureCanvasTkAgg(self.fig2, master=self.t2); c.get_tk_widget().pack(fill="both", expand=True, padx=14, pady=6)
        self.c2 = c; self.lst = []; self.lidx = 0
        self._t2_toggle()

    def _t2_toggle(self):
        if self.crit.get() == "top_n":
            self.tnf.pack(side="left", padx=10); self.thf.pack_forget()
        else:
            self.thf.pack(side="left", padx=4); self.tnf.pack_forget()

    def _t2_build(self):
        m = self.metric_col; df = self.df.dropna(subset=[m])
        if self.crit.get() == "top_n":
            sub = df.nlargest(self.nvar.get(), m)
        else:
            op = self.opv.get(); thr = self.tv.get()
            sub = df[{">": df[m]>thr,"<": df[m]<thr,">=": df[m]>=thr,"<=": df[m]<=thr}[op]]
        self.lst  = [r for _, r in sub.iterrows()]; self.lidx = 0
        labels = [f"{r.get('file_name_radical','?')}_seg{r.get('segment_id','?')}"
                  for r in self.lst]
        self.jcb["values"] = labels
        self.li_lbl.config(text=f"{len(self.lst)} items")
        if self.lst: self._t2_show()

    def _t2_show(self):
        if not self.lst: return
        row = self.lst[self.lidx]
        self.nav_lbl.config(text=f"[{self.lidx+1}/{len(self.lst)}]")
        self.jcb.current(self.lidx)
        self._show(self.ax2, self.c2, row, prefix=f"[{self.lidx+1}/{len(self.lst)}]  ")

    def _t2_prev(self):
        if self.lst and self.lidx > 0: self.lidx -= 1; self._t2_show()
    def _t2_next(self):
        if self.lst and self.lidx < len(self.lst)-1: self.lidx += 1; self._t2_show()
    def _t2_jump(self):
        i = self.jcb.current()
        if i >= 0: self.lidx = i; self._t2_show()

    # ════════════════════════════════════════════
    # TAB 3
    # ════════════════════════════════════════════
    def _tab3(self):
        ctrl = tk.Frame(self.t3, bg=self.BG); ctrl.pack(fill="x", padx=14, pady=10)
        num_cols = [c for c in self.df.select_dtypes(include=np.number).columns]
        cat_cols = [""] + [c for c in self.df.select_dtypes(include=object).columns]

        for lbl, attr, default_fn, w in [
            ("X axis:",   "sx", lambda: next((c for c in num_cols if "rate" in c.lower() or "trill" in c.lower()), num_cols[0] if num_cols else ""), 18),
            ("Y axis:",   "sy", lambda: next((c for c in num_cols if "band" in c.lower() or "bw"   in c.lower()), num_cols[1] if len(num_cols)>1 else num_cols[0] if num_cols else ""), 18),
            ("Color by:", "sc", lambda: next((c for c in ["species","genus","family"] if c in self.df.columns), ""), 16),
            ("Size by:",  "ss", lambda: "", 16),
        ]:
            tk.Label(ctrl, text=lbl, fg=self.FG2, bg=self.BG).pack(side="left", padx=(8, 2))
            v = tk.StringVar(value=default_fn())
            setattr(self, f"_{attr}", v)
            vals = num_cols if attr in ("sx","sy","ss") else cat_cols
            if attr == "ss": vals = [""] + num_cols
            ttk.Combobox(ctrl, textvariable=v, values=vals, width=w).pack(side="left")

        tk.Button(ctrl, text="Open Plotly →", bg=self.GRN, fg=self.BG2,
                font=("Helvetica", 9, "bold"), relief="flat", padx=10,
                command=self._t3_plotly).pack(side="left", padx=14)

        tk.Label(self.t3,
                text="Click 'Open Plotly →' → hover a point to get its index → enter it below.",
                fg=self.FG2, bg=self.BG, font=("Helvetica", 9)).pack(anchor="w", padx=16)

        pick = tk.Frame(self.t3, bg=self.BG); pick.pack(fill="x", padx=14, pady=6)
        tk.Label(pick, text="Index (from hover):", fg=self.FG2, bg=self.BG).pack(side="left")
        self.pick_idx = tk.IntVar(value=0)
        tk.Entry(pick, textvariable=self.pick_idx, width=7,
                bg=self.SURF, fg=self.FG, insertbackground="white").pack(side="left", padx=4)
        tk.Button(pick, text="Show", bg=self.ACC, fg=self.BG2, relief="flat", padx=10,
                command=self._t3_show).pack(side="left", padx=4)

        self.fig3, self.ax3 = plt.subplots(figsize=(10, 3.4), facecolor=self.BG2)
        self.ax3.set_facecolor(self.BG2)
        c = FigureCanvasTkAgg(self.fig3, master=self.t3); c.get_tk_widget().pack(fill="both", expand=True, padx=14, pady=6)
        self.c3 = c

    def _t3_plotly(self):
        x, y = self._sx.get(), self._sy.get()
        color = self._sc.get() or None
        size  = self._ss.get() or None
        if x not in self.df.columns or y not in self.df.columns:
            messagebox.showerror("Error", f"Columns not found: {x}, {y}"); return
        df = self.df.dropna(subset=[x, y]).copy()
        df["_idx"] = np.arange(len(df))          
        df["_label"] = df["file_name_radical"].astype(str) + "_seg" + df["segment_id"].astype(str)
        hover = [c for c in ["_label", "species", "genus", "family", self.metric_col]
                if c in df.columns and c not in [x, y]]
        fig = px.scatter(df, x=x, y=y, color=color, size=size,
                        hover_name="_idx",       
                        hover_data=hover,
                        template="plotly_dark", title=f"{y}  vs  {x}",
                        labels={x: x.replace("_"," ").title(), y: y.replace("_"," ").title()},
                        opacity=0.75)
        fig.update_traces(marker=dict(size=9 if not size else None, line=dict(width=0)))
        fig.update_layout(paper_bgcolor="#1e1e2e", plot_bgcolor="#181825",
                        hoverlabel=dict(bgcolor="#313244", font_color="#cdd6f4"))
        self._plotly_df = df.reset_index(drop=True)
        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        fig.write_html(tmp.name); webbrowser.open(f"file://{tmp.name}")

    def _t3_show(self):
        idx = self.pick_idx.get()
        # Use the filtered df (dropna) stored during the last Open Plotly, which has a stable positional index
        source = getattr(self, "_plotly_df", self.df)
        if 0 <= idx < len(source):
            self._show(self.ax3, self.c3, source.iloc[idx])
        else:
            # messagebox.showwarning("Index invalide", f"Index {idx} hors bornes (0–{len(source)-1})")
            messagebox.showwarning("Invalid Index", f"Index {idx} out of bounds (0–{len(source)-1})")

    # ── Shared spectro display ────────────────────────────────────────────────
    def _show(self, ax, canvas, row, prefix=""):
        try:
            S_db, times, freqs, f_min, f_max, t_min, t_max = load_spectrogram(row, self.config)
            mv = row.get(self.metric_col, float("nan"))
            title = f"{prefix}{row.get('file_name_radical','?')}  seg {row.get('segment_id','?')}  |  {self.metric_col}={mv:.4f}"
            crop=(self.spectro_mode.get() == "crop")
            draw_spectrogram(ax, S_db, times, freqs, f_min, f_max, t_min, t_max, title=title, crop_to_bounds=crop)
            canvas.draw()
        except Exception as e:
            messagebox.showerror("Spectrogram Error", str(e))


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

def launch_gui(config, df_predictions=None, metric_col="confidence"):
    if df_predictions is None:
        path = config.output_csv.replace(".csv", "_predictions.csv")
        df_predictions = pd.read_csv(path)
    SpectroViewer(df_predictions, config, metric_col=metric_col).mainloop()