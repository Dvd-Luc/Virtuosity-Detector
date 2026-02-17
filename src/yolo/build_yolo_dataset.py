import librosa
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.utils.spectrograms import *
from src.utils.segmentation import tf_to_yolo
from src.config import load_config_yaml

# =========================
# CONFIG
# =========================

YAML_PATH = "config.yaml"

# =========================
# MAIN
# =========================

cfg = load_config_yaml(YAML_PATH)
IMG_DIR = Path(cfg.yolo_dataset_subdir) / "images"
LBL_DIR = Path(cfg.yolo_dataset_subdir) / "labels"

IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(cfg.output_csv)

for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Segment spectrograms production",
        unit="segment"
    ):

    audio_path = Path(cfg.audio_dir) / row["file_name_radical"]
    if not audio_path.exists():
        continue

    y, sr = librosa.load(audio_path, sr=None, mono=True)

    y_seg, seg_start, seg_end = extract_centered_segment(
        y,
        sr,
        row["time_start"],
        row["time_end"],
        cfg.win_len
    )

    stem = f"{audio_path.stem}_seg{row['segment_id']}"

    img_path = IMG_DIR / f"{stem}.png"
    lbl_path = LBL_DIR / f"{stem}.txt"

    save_spectrogram(y_seg, sr, img_path, cfg)

    t0 = row["trill_t_start"]
    t1 = row["trill_t_end"]

    if row["trill_min_freq"] == 0 and row["trill_max_freq"] == 0:
        open(lbl_path, "w").close()
        continue

    xc, yc, w, h = tf_to_yolo(
        t0, t1,
        row["trill_min_freq"], row["trill_max_freq"],
        cfg.win_len,
        sr / 2
    )

    if w <= 0 or h <= 0:
        open(lbl_path, "w").close()
        continue

    with open(lbl_path, "w") as f:
        f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

print("✅ Dataset YOLO généré")