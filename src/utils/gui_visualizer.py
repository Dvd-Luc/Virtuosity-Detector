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
import tkinter.filedialog as fd
import numpy as np
import pandas as pd
import librosa
import matplotlib
import sounddevice as sd
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotly.express as px
from scipy.stats import linregress
from src.utils.metrics import upper_bound_regression


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
        self._current_row = None
        self.title("Trill Spectrogram Viewer")
        self.geometry("1250x840")
        self.configure(bg=self.BG)
        self._style(); self._header(); self._notebook()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _on_close(self):
        sd.stop()
        self.destroy()
        import sys; sys.exit(0)

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
            
        # tk.Label(h, text="Play:", fg=self.FG2, bg=self.BG2).pack(side="left", padx=(24, 4))
        # tk.Button(h, text="▶ Segment", bg=self.SURF, fg=self.FG, relief="flat", padx=8,
        #         command=lambda: self._play_audio(self._current_row, mode="segment")
        #         ).pack(side="left", padx=2)
        # tk.Button(h, text="▶ Prediction ±10%", bg=self.SURF, fg=self.FG, relief="flat", padx=8,
        #         command=lambda: self._play_audio(self._current_row, mode="prediction")
        #         ).pack(side="left", padx=2)
        # tk.Button(h, text="⏹", bg=self.SURF, fg=self.FG, relief="flat", padx=6,
        #         command=sd.stop).pack(side="left", padx=2)
        # self._current_row = None 

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
        self.tax_cb  = ttk.Combobox(top, textvariable=self.tax_val, width=40); self.tax_cb.pack(side="left")
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

        play1 = tk.Frame(self.t1, bg=self.BG); play1.pack(pady=4)
        tk.Button(play1, text="▶ Segment", bg=self.SURF, fg=self.FG, relief="flat", padx=10,
                command=lambda: self._play_audio(self._current_row, mode="segment")
                ).pack(side="left", padx=6)
        tk.Button(play1, text="▶ Prediction ±10%", bg=self.SURF, fg=self.FG, relief="flat", padx=10,
                command=lambda: self._play_audio(self._current_row, mode="prediction")
                ).pack(side="left", padx=6)
        # tk.Button(play1, text="⏹ Stop", bg=self.SURF, fg=self.FG, relief="flat", padx=10,
        #         command=sd.stop).pack(side="left", padx=6)
        
        self._opt1_load_values()

    def _opt1_load_values(self):
        col = self.tax_level.get()
        if col not in self.df.columns:
            self.tax_cb["values"] = []; return

        has_pred  = set(self.df.dropna(subset=["t_min"])[col].dropna().unique())
        all_vals  = sorted(self.df[col].dropna().unique().tolist())

        display   = [v if v in has_pred else f"{v}  ⚠ no prediction" for v in all_vals]
        self._tax_display_map = dict(zip(display, all_vals))
        self.tax_cb["values"] = display
        if display:
            self.tax_val.set(display[0])
            self._opt1_update_stats()

    def _opt1_sub(self):
        col = self.tax_level.get()
        raw_val = self._tax_display_map.get(self.tax_val.get(), self.tax_val.get())
        return self.df[self.df[col] == raw_val]

    def _opt1_update_stats(self):
        sub = self._opt1_sub(); m = self.metric_col
        for v in self.sv.values(): v.set(False)

        sub_pred = sub.dropna(subset=["t_min"])

        if sub_pred.empty:
            for s in self.sl: self.sl[s].config(text="—")
            # Display a random spectrogram from the group if no predictions are available, to at least show something relevant
            if not sub.empty:
                rand_row = sub.sample(1).iloc[0]
                self._show(self.ax1, self.c1, rand_row,
                        prefix="[no prediction — random segment]  ")
            return

        # Stats for the metric column, only on rows with predictions
        vals = sub_pred[m].dropna()
        self.sl["count"].config(text=str(len(vals)))
        self.sl["min"].config(text=f"{vals.min():.4f}")
        self.sl["max"].config(text=f"{vals.max():.4f}")
        self.sl["mean"].config(text=f"{vals.mean():.4f}")
        self.sl["median"].config(text=f"{vals.median():.4f}")

    def _opt1_view(self, stat):
        for s, v in self.sv.items():
            if s != stat: v.set(False)
        if not self.sv[stat].get(): return
        sub      = self._opt1_sub()
        sub_pred = sub.dropna(subset=["t_min"])
        if sub_pred.empty: return
        vals = sub_pred[self.metric_col].dropna()
        idx = {"min":    vals.idxmin,
            "max":    vals.idxmax,
            "mean":   lambda: (vals - vals.mean()).abs().idxmin(),
            "median": lambda: (vals - vals.median()).abs().idxmin()}[stat]()
        self._show(self.ax1, self.c1, sub_pred.loc[idx])

    # ════════════════════════════════════════════
    # TAB 2
    # ════════════════════════════════════════════
    def _tab2(self):
        ctrl = tk.Frame(self.t2, bg=self.BG); ctrl.pack(fill="x", padx=14, pady=10)
        tk.Label(ctrl, text="Criterion:", fg=self.FG2, bg=self.BG).pack(side="left")
        self.crit = tk.StringVar(value="top_n_high")
        for v, t in [("top_n_high", "Top N highest"), ("top_n_low", "Top N lowest"), ("threshold", "Threshold filter")]:
            tk.Radiobutton(ctrl, text=t, variable=self.crit, value=v,
                           command=self._t2_toggle,
                           fg=self.FG, bg=self.BG, selectcolor=self.SURF,
                           activebackground=self.BG).pack(side="left", padx=8)

        # Parameters frame (Top-N or Threshold depending on criterion)   
        param_frame = tk.Frame(ctrl, bg=self.BG)
        param_frame.pack(side="left", padx=10)

        # Top-N
        self.tnf = tk.Frame(param_frame, bg=self.BG); self.tnf.pack(side="left", padx=10)
        tk.Label(self.tnf, text="N:", fg=self.FG2, bg=self.BG).pack(side="left")
        self.nvar = tk.IntVar(value=10)
        tk.Spinbox(self.tnf, from_=1, to=1000, textvariable=self.nvar, width=5,
                   bg=self.SURF, fg=self.FG, insertbackground="white",
                   buttonbackground=self.SURF).pack(side="left")
        # Threshold
        self.thf = tk.Frame(param_frame, bg=self.BG)
        self.opv = tk.StringVar(value=">")
        ttk.Combobox(self.thf, textvariable=self.opv, values=[">","<",">=","<="], width=4).pack(side="left")
        self.tv = tk.DoubleVar(value=0.5)
        tk.Entry(self.thf, textvariable=self.tv, width=8,
                 bg=self.SURF, fg=self.FG, insertbackground="white").pack(side="left", padx=4)
        
        tk.Button(ctrl, text="Build List ▶", bg=self.ACC, fg=self.BG2,
                  font=("Helvetica", 9, "bold"), relief="flat", padx=10,
                  command=self._t2_build).pack(side="left", padx=14)
        
        tk.Button(ctrl, text="📂 Load CSV", bg=self.SURF, fg=self.FG, relief="flat", padx=8,
          command=lambda: self._load_csv_to_df(target="tab2")).pack(side="left", padx=4)
        
        self.li_lbl = tk.Label(ctrl, text="No list yet", fg=self.FG2, bg=self.BG)
        self.li_lbl.pack(side="left")

        tk.Button(ctrl, text="⬇ Export full list", bg=self.GRN, fg=self.BG2, relief="flat", padx=8,
          command=self._t2_export_list).pack(side="left", padx=4)

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
        tk.Button(nav, text="＋ Export row", bg=self.GRN, fg=self.BG2, relief="flat", padx=8,
          command=lambda: self._export_to_csv(self.lst[self.lidx]) if self.lst else None
          ).pack(side="left", padx=8)
        self.fig2, self.ax2 = plt.subplots(figsize=(10, 3.4), facecolor=self.BG2)
        self.ax2.set_facecolor(self.BG2)
        c = FigureCanvasTkAgg(self.fig2, master=self.t2); c.get_tk_widget().pack(fill="both", expand=True, padx=14, pady=6)
        self.c2 = c; self.lst = []; self.lidx = 0
        self._t2_toggle()

        play2 = tk.Frame(self.t2, bg=self.BG); play2.pack(pady=4)
        tk.Button(play2, text="▶ Segment", bg=self.SURF, fg=self.FG, relief="flat", padx=10,
                command=lambda: self._play_audio(self._current_row, mode="segment")
                ).pack(side="left", padx=6)
        tk.Button(play2, text="▶ Prediction ±10%", bg=self.SURF, fg=self.FG, relief="flat", padx=10,
                command=lambda: self._play_audio(self._current_row, mode="prediction")
                ).pack(side="left", padx=6)
        # tk.Button(play2, text="⏹ Stop", bg=self.SURF, fg=self.FG, relief="flat", padx=10,
        #         command=sd.stop).pack(side="left", padx=6)

    def _t2_toggle(self):
        if self.crit.get() == "top_n_high":
            self.tnf.pack(side="left", padx=10); self.thf.pack_forget()
        elif self.crit.get() == "top_n_low":
            self.tnf.pack(side="left", padx=10); self.thf.pack_forget()
        else:
            self.thf.pack(side="left", padx=4); self.tnf.pack_forget()

    def _t2_build(self):
        m = self.metric_col; df = self.df.dropna(subset=[m])
        if self.crit.get() == "top_n_high":
            sub = df.nlargest(self.nvar.get(), m)
        elif self.crit.get() == "top_n_low":
            sub = df.nsmallest(self.nvar.get(), m)
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

    def _t2_export_list(self):
        if not self.lst:
            messagebox.showwarning("Empty list", "No list to export — build or load a list first.")
            return

        path = fd.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Export full list"
        )
        if not path: return

        pd.DataFrame(self.lst).to_csv(path, index=False)
        messagebox.showinfo("Exported", f"{len(self.lst)} rows saved → {os.path.basename(path)}")

    # ════════════════════════════════════════════
    # TAB 3
    # ════════════════════════════════════════════
    def _tab3(self):
        ctrl = tk.Frame(self.t3, bg=self.BG); ctrl.pack(fill="x", padx=14, pady=10)
        num_cols = [c for c in self.df.select_dtypes(include=np.number).columns]
        cat_cols = [""] + [c for c in self.df.select_dtypes(include=object).columns]
        self._t3_df = self.df.dropna(subset=num_cols) if num_cols else self.df.copy()

        for lbl, attr, default_fn, w in [
            ("X axis:",   "sx", lambda: next((c for c in num_cols if "rate" in c.lower() or "trill" in c.lower()), num_cols[0] if num_cols else ""), 18),
            ("Y axis:",   "sy", lambda: next((c for c in num_cols if "band" in c.lower() or "bw"   in c.lower()), num_cols[1] if len(num_cols)>1 else num_cols[0] if num_cols else ""), 18),
            ("Color by:", "sc", lambda: "", 16),
            ("Size by:",  "ss", lambda: "", 16),
        ]:
            tk.Label(ctrl, text=lbl, fg=self.FG2, bg=self.BG).pack(side="left", padx=(8, 2))
            v = tk.StringVar(value=default_fn())
            setattr(self, f"_{attr}", v)
            # vals = num_cols if attr in ("sx","sy","ss") else cat_cols # Only categorical for color, size can be numeric or categorical
            if attr == "sc":
                vals = [""] + list(self.df.columns) 
            else:
                vals = num_cols if attr in ("sx","sy") else [""] + num_cols

            if attr == "ss": vals = [""] + num_cols
            ttk.Combobox(ctrl, textvariable=v, values=vals, width=w).pack(side="left")

        tk.Button(ctrl, text="Open Plotly →", bg=self.GRN, fg=self.BG2,
                font=("Helvetica", 9, "bold"), relief="flat", padx=10,
                command=self._t3_plotly).pack(side="left", padx=14)
        
        self.ub_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl, text="Upper bound regression", variable=self.ub_var,
                    fg=self.FG, bg=self.BG, selectcolor=self.SURF,
                    activebackground=self.BG).pack(side="left", padx=8)

        default_bin = 2.0
        tk.Label(ctrl, text="bin_width:", fg=self.FG2, bg=self.BG).pack(side="left", padx=(8,2))
        self.ub_bin = tk.DoubleVar(value=default_bin)
        tk.Entry(ctrl, textvariable=self.ub_bin, width=5,
                bg=self.SURF, fg=self.FG, insertbackground="white").pack(side="left")

        self.ub_logy = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl, text="log Y", variable=self.ub_logy,
                    fg=self.FG, bg=self.BG, selectcolor=self.SURF,
                    activebackground=self.BG).pack(side="left", padx=4)
        
        self.theme_var = tk.StringVar(value="dark")
        tk.Label(ctrl, text="Theme:", fg=self.FG2, bg=self.BG).pack(side="left", padx=(16,2))
        ttk.Combobox(ctrl, textvariable=self.theme_var, values=["dark", "light"], width=8).pack(side="left")

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
        
        tk.Button(pick, text="＋ Export row", bg=self.GRN, fg=self.BG2, relief="flat", padx=8,
          command=lambda: self._export_to_csv(
              getattr(self, "_plotly_df", self.df).iloc[self.pick_idx.get()]
          )).pack(side="left", padx=6)
        
        tk.Button(pick, text="📂 Load subset CSV", bg=self.SURF, fg=self.FG, relief="flat", padx=8,
          command=lambda: self._load_csv_to_df(target="tab3")).pack(side="left", padx=6)
        self.t3_subset_lbl = tk.Label(pick, text="Full dataset", fg=self.FG2, bg=self.BG)
        self.t3_subset_lbl.pack(side="left", padx=8)


        self.fig3, self.ax3 = plt.subplots(figsize=(10, 3.4), facecolor=self.BG2)
        self.ax3.set_facecolor(self.BG2)
        c = FigureCanvasTkAgg(self.fig3, master=self.t3); c.get_tk_widget().pack(fill="both", expand=True, padx=14, pady=6)
        self.c3 = c

        play3 = tk.Frame(self.t3, bg=self.BG); play3.pack(pady=4)
        tk.Button(play3, text="▶ Segment", bg=self.SURF, fg=self.FG, relief="flat", padx=10,
                command=lambda: self._play_audio(self._current_row, mode="segment")
                ).pack(side="left", padx=6)
        tk.Button(play3, text="▶ Prediction ±10%", bg=self.SURF, fg=self.FG, relief="flat", padx=10,
                command=lambda: self._play_audio(self._current_row, mode="prediction")
                ).pack(side="left", padx=6)
        # tk.Button(play3, text="⏹ Stop", bg=self.SURF, fg=self.FG, relief="flat", padx=10,
        #         command=sd.stop).pack(side="left", padx=6)

    def _t3_plotly(self):
        x, y = self._sx.get(), self._sy.get()
        color = self._sc.get() or None
        size  = self._ss.get() or None

        color_continuous = False

        if color and color in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[color]):
                color_continuous = True


        if x not in self.df.columns or y not in self.df.columns:
            messagebox.showerror("Error", f"Columns not found: {x}, {y}"); return
        source = self._t3_df if self._t3_df is not None else self.df
        df = source.dropna(subset=[x, y]).copy()
        
        df["_idx"] = np.arange(len(df))          
        df["_label"] = df["file_name_radical"].astype(str) + "_seg" + df["segment_id"].astype(str)
        hover = [c for c in ["_label", "species", "genus", "family", self.metric_col]
                if c in df.columns and c not in [x, y]]
        
        
        if self.theme_var.get() == "dark":
            fig = px.scatter(df, 
                        x=x, 
                        y=y,
                        color=color, 
                        color_continuous_scale="Viridis" if color_continuous else None,
                        color_discrete_sequence=px.colors.qualitative.Pastel if not color_continuous else None,
                        size=size,
                        hover_name="_idx",       
                        hover_data=hover,
                        template="plotly_dark", title=f"{y}  vs  {x}",
                        labels={x: x.replace("_"," ").title(), y: y.replace("_"," ").title()},
                        opacity=0.5,
                        log_y=self.ub_logy.get())
            fig.update_traces(marker=dict(size=6 if not size else None, line=dict(width=0)))
            fig.update_layout(paper_bgcolor="#1e1e2e", plot_bgcolor="#181825",
                            hoverlabel=dict(bgcolor="#313244", font_color="#cdd6f4"))
        
        else :
            fig = px.scatter(df,
                    x=x,
                    y=y,
                    color=color,
                    color_continuous_scale="Viridis" if color_continuous else None,
                    color_discrete_sequence=px.colors.qualitative.Set2 if not color_continuous else None,
                    size=size,
                    hover_name="_idx",
                    hover_data=hover,
                    template="plotly_white", title=f"{y}  vs  {x}",
                    labels={x: x.replace("_"," ").title(), y: y.replace("_"," ").title()},
                    opacity=0.7,
                    log_y=self.ub_logy.get())
            fig.update_traces(marker=dict(size=6 if not size else None,
                                        line=dict(width=0.5, color="#ffffff")))
            fig.update_layout(
                paper_bgcolor="#d9d2d2",   # gris très clair pour le fond extérieur
                plot_bgcolor="#d9d2d2",    # blanc pur pour la zone du graphe
                font=dict(color="#333333"),
                title_font=dict(color="#1a1a2e"),
                hoverlabel=dict(bgcolor="#e8e8f0", font_color="#1a1a2e",
                                bordercolor="#aaaacc"),
            )
            fig.update_xaxes(gridcolor="#AAA5A5", zerolinecolor="#585353")
            fig.update_yaxes(gridcolor="#AAA5A5", zerolinecolor="#585353")
        
        # Upper bound regression (max per bin + linear fit)
        if self.ub_var.get():
            ub_df, reg = upper_bound_regression(
                df, x_col=x, y_col=y,
                bin_width=self.ub_bin.get(),
                log_y=self.ub_logy.get()
            )
            # Points des maxima par bin
            fig.add_scatter(x=ub_df[x], y=ub_df[y],
                            mode="markers", name="UB points",
                            marker=dict(color="#f38ba8", size=10, symbol="diamond"),
                            hoverinfo="skip",
                            hovertemplate=None,
                            showlegend=True)
            # Droite de régression
            x_line = np.linspace(df[x].min(), df[x].max(), 200)
            if self.ub_logy.get():
                y_line = np.exp(reg["slope"] * x_line + reg["intercept"])
            else:
                y_line = reg["slope"] * x_line + reg["intercept"]
            label = (f"UB fit  r={reg['r_value']:.3f}  "
                    f"p={reg['p_value']:.3e}  "
                    f"n={reg['n']}")
            fig.add_scatter(x=x_line, y=y_line,
                            mode="lines", name=label,
                            line=dict(color="#f38ba8", width=2, dash="dash"))
        self._plotly_df = df.reset_index(drop=True)
        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        html = fig.to_html(include_plotlyjs="cdn")

        # Injection du JS de copie au clic
        copy_js = """
        <script>
        document.addEventListener("DOMContentLoaded", function() {
            var plot = document.querySelector(".plotly-graph-div");
            plot.on("plotly_click", function(data) {
                var idx = data.points[0].hovertext;   // hovertext = hover_name = _idx
                navigator.clipboard.writeText(String(idx)).then(function() {
                    // Feedback visuel : petit toast
                    var toast = document.createElement("div");
                    toast.textContent = "Index " + idx + " copied!";
                    toast.style.cssText = "position:fixed;bottom:30px;left:50%;transform:translateX(-50%);"
                        + "background:#89b4fa;color:#1e1e2e;padding:8px 20px;border-radius:8px;"
                        + "font-family:Helvetica;font-size:14px;font-weight:bold;z-index:9999;"
                        + "opacity:1;transition:opacity 1s;";
                    document.body.appendChild(toast);
                    setTimeout(function(){ toast.style.opacity="0"; }, 1500);
                    setTimeout(function(){ toast.remove(); }, 2500);
                });
            });
        });
        </script>
        """

        html = html + copy_js

        with open(tmp.name, "w", encoding="utf-8") as f:
            f.write(html)
        webbrowser.open(f"file://{tmp.name}")

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
        self._current_row = row  # Store current row for potential future use (e.g., playback)
        try:
            S_db, times, freqs, f_min, f_max, t_min, t_max = load_spectrogram(row, self.config)
            mv = row.get(self.metric_col, float("nan"))
            title = f"{prefix}{row.get('file_name_radical','?')}  seg {row.get('segment_id','?')}  |  {self.metric_col}={mv:.4f}"
            crop=(self.spectro_mode.get() == "crop")
            draw_spectrogram(ax, S_db, times, freqs, f_min, f_max, t_min, t_max, title=title, crop_to_bounds=crop)
            canvas.draw()
        except Exception as e:
            messagebox.showerror("Spectrogram Error", str(e))
    
    def _play_audio(self, row, mode="segment"):
        """mode = 'segment' (full) or 'prediction' (box ± 10%)"""
        try:
            audio_path = os.path.join(self.config.audio_subdir, str(row["file_name_radical"]))
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)

            seg_start = row.get("seg_start", 0)
            seg_end   = row.get("seg_end", duration)

            if mode == "prediction":
                t_min = float(row.get("t_min", 0)) + seg_start
                t_max = float(row.get("t_max", duration)) + seg_start
                margin = (t_max - t_min) * 0.10
                start = max(0, t_min - margin)
                end   = min(duration, t_max + margin)
            else:
                start = seg_start
                end   = seg_end

            y_play = y[int(start * sr): int(end * sr)]
            sd.stop()
            sd.play(y_play, sr)
        except Exception as e:
            messagebox.showerror("Playback Error", str(e))

    def _export_to_csv(self, row):
        if not hasattr(self, "_export_path") or not self._export_path:
            path = fd.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Choose or create export CSV"
            )
            if not path: return
            self._export_path = path

        row_df = pd.DataFrame([row])

        if os.path.exists(self._export_path):
            existing = pd.read_csv(self._export_path)
            # Déduplication sur file_name_radical + segment_id
            dup = (
                (existing["file_name_radical"] == row["file_name_radical"]) &
                (existing["segment_id"]        == row["segment_id"])
            ).any()
            if dup:
                messagebox.showinfo("Duplicate", f"{row['file_name_radical']} seg {row['segment_id']} already exists.")
                return
            combined = pd.concat([existing, row_df], ignore_index=True)
        else:
            combined = row_df

        combined.to_csv(self._export_path, index=False)
        messagebox.showinfo("Exported", f"Added → {os.path.basename(self._export_path)}  ({len(combined)} lines)")

    # def _t2_load_csv(self):
    #     path = fd.askopenfilename(filetypes=[("CSV files", "*.csv")], title="Load list CSV")
    #     if not path: return

    #     loaded = pd.read_csv(path)
    #     # Joining on file_name_radical + segment_id to ensure we only show rows that have corresponding spectrograms in our main dataframe
    #     merged = pd.merge(
    #         loaded[["file_name_radical", "segment_id"]],
    #         self.df,
    #         on=["file_name_radical", "segment_id"],
    #         how="inner"
    #     )
    #     if merged.empty:
    #         messagebox.showwarning("No Match",
    #                             "No rows in the CSV match the loaded dataframe.")
    #         return

    #     self.lst  = [r for _, r in merged.iterrows()]
    #     self.lidx = 0
    #     m = self.metric_col
    #     labels = [
    #         f"{r.get('file_name_radical','?')}_seg{r.get('segment_id','?')}  "
    #         f"[{m}={r.get(m, float('nan')):.4f}]" # Display metric value in label for easier identification in the list
    #         for r in self.lst
    #     ]
    #     self.jcb["values"] = labels
    #     self.li_lbl.config(text=f"{len(self.lst)} items (from CSV)")
    #     self._t2_show()

    def _load_csv_to_df(self, target="tab2"):

        path = fd.askopenfilename(filetypes=[("CSV files", "*.csv")], title="Load list CSV")
        if not path: return

        loaded = pd.read_csv(path)
        merged = pd.merge(
            loaded[["file_name_radical", "segment_id"]],
            self.df,
            on=["file_name_radical", "segment_id"],
            how="inner"
        ).reset_index(drop=True)

        if merged.empty:
            messagebox.showwarning("No Match", "No rows in the CSV match the loaded dataframe.")
            return

        n     = len(merged)
        fname = os.path.basename(path)

        if target == "tab2":
            self.lst  = [r for _, r in merged.iterrows()]
            self.lidx = 0
            m = self.metric_col
            labels = [
                f"{r.get('file_name_radical','?')}_seg{r.get('segment_id','?')}  "
                f"[{m}={r.get(m, float('nan')):.4f}]"
                for r in self.lst
            ]
            self.jcb["values"] = labels
            self.li_lbl.config(text=f"{n} items (from CSV: {fname})")
            self._t2_show()

        elif target == "tab3":
            self._t3_df = merged
            self.t3_subset_lbl.config(
                text=f"Subset: {n} / {len(self.df)} samples  ({fname})",
                fg=self.GRN
            )

# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

def launch_gui(config, df_predictions=None, metric_col="confidence"):
    if df_predictions is None:
        path = config.output_csv.replace(".csv", "_predictions.csv")
        df_predictions = pd.read_csv(path)
    SpectroViewer(df_predictions, config, metric_col=metric_col).mainloop()