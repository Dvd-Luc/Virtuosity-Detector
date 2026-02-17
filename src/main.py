
import os
import cv2
import pandas as pd
import numpy as np
import librosa
import re
from ultralytics import YOLO
import albumentations as A
from tqdm import tqdm

from src.config import load_config_yaml

from src.utils.annotate import annotate
from src.utils.counting import *
from src.utils.segmentation import *
from src.utils.spectrograms import *

def constrained_stratified_sample(df, group_col, n_samples, max_per_group=2,
                                   group_exclusion_list=None, diversity_col=None, random_state=42):
    
    """
    Perform a constrained stratified sampling over groups in a DataFrame.

    The function iterates over groups defined by `group_col` and samples rows
    while enforcing:
    - a maximum number of samples per group,
    - a global maximum number of samples,
    - optional exclusion of specific groups,
    - optional intra-group diversity based on `diversity_col`.

    When `diversity_col` is provided, up to two samples with different values
    of this column are selected per group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_col : str
        Column defining the groups.
    n_samples : int
        Total number of samples to select.
    max_per_group : int, optional
        Maximum number of samples per group (default: 2).
    group_exclusion_list : list or None, optional
        Groups to exclude from sampling.
    diversity_col : str or None, optional
        Column used to enforce diversity within groups.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Shuffled DataFrame containing the selected samples.
    """
    rng = np.random.default_rng(random_state)
    selected_rows = []
    total_selected = 0

    if group_exclusion_list is not None:
        df = df[~df[group_col].isin(group_exclusion_list)]

    groups = df[group_col].unique().tolist()
    rng.shuffle(groups)

    for group in groups:
        if total_selected >= n_samples:
            break

        df_g = df[df[group_col] == group]
        if df_g.empty:
            continue

        if diversity_col is None:
            if len(df_g) < 2:
                continue
            n_take = min(max_per_group, len(df_g), n_samples - total_selected)
            sampled = df_g.sample(n=n_take, random_state=random_state)
            selected_rows.append(sampled)
            total_selected += len(sampled)
            continue

        values = df_g[diversity_col].dropna().unique().tolist()
        if len(values) < 2:
            continue

        rng.shuffle(values)
        v1, v2 = values[:2]

        row1 = df_g[df_g[diversity_col] == v1].sample(n=1, random_state=random_state)
        row2 = df_g[df_g[diversity_col] == v2].sample(n=1, random_state=random_state)
        sampled = pd.concat([row1, row2])
        
        remaining = n_samples - total_selected
        sampled = sampled.iloc[:remaining]

        selected_rows.append(sampled)
        total_selected += len(sampled)

    if not selected_rows:
        return df.iloc[0:0].copy()

    return pd.concat(selected_rows).sample(frac=1, random_state=random_state).reset_index(drop=True)

def manual_annotation_session(df_audio_files, annotator, config):
    """
    Run a manual annotation session on a set of audio segments.

    The function iterates over a DataFrame describing audio segments,
    extracts a centered window around each segment, and collects
    human-provided annotations. Annotations that are incomplete or
    inconsistent are skipped. Valid annotations are appended to an
    output CSV file.

    Parameters
    ----------
    df_audio_files : pd.DataFrame
        DataFrame describing audio segments to annotate.
    annotator : str
        Identifier of the human annotator.
    config : object
        Configuration object containing paths and annotation parameters.

    Returns
    -------
    None
    """

    rows = []

    for i, (idx, row) in enumerate(df_audio_files.iterrows(), start=1):
        f = row["file_name_radical"]
        print(f"\n[{i}/{len(df_audio_files)}] Annotating {f} segment {row['syllable_rank']}")
        
        y, sr = librosa.load(os.path.join(config.audio_subdir, f), sr=None)
        start = row["time_start"]
        end = row["time_end"]
        
        y_segment, start, end = extract_centered_segment(y, sr, start, end, config.win_len)

        ann = annotate(y_segment, sr, config, prediction=None)

        if ann["no_trill"]:
            if (ann["count"] != 0 or ann["t_min"] is None or ann["t_max"] is None):
                print("  ⚠️  Inconsistent no-trill, skipping")
                continue
        else:
            if (ann["t_min"] is None or ann["t_max"] is None or ann["f_min"] is None or 
                ann["f_max"] is None or ann["count"] is None or ann["count"] < 1):
                print("  ⊘ Skipped (incomplete)")
                continue

        rows.append({
            "file_name_radical": f,
            "species": row["species"],
            "segment_id": row["syllable_rank"],
            "annotator": annotator,
            "annotation_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "time_start": start,
            "time_end": end,
            "trill_t_start": ann["t_min"],
            "trill_t_end": ann["t_max"],
            "trill_min_freq": ann["f_min"],
            "trill_max_freq": ann["f_max"],
            "num_trills": ann["count"],
            "has_trill": not ann["no_trill"],
            "trill_duration": (ann["t_max"] - ann["t_min"]) if ann["t_min"] is not None and ann["t_max"] is not None else None,
        })

    if rows:
        df = pd.DataFrame(rows)

        if os.path.exists(config.output_csv):
            df_existing = pd.read_csv(config.output_csv)
            df_out = pd.concat([df_existing, df], ignore_index=True)
        else:
            df_out = df

        os.makedirs(os.path.dirname(config.output_csv), exist_ok=True)
        df_out.to_csv(config.output_csv, index=False)
        print(f"\n✓ Saved {len(df)} new annotations")


def assisted_annotation_session(df_audio_files, annotator, config, debug=False):
    """
    Run an assisted annotation session using pre-trained models.

    The function performs manual annotation aided by two models:
    one for temporal/frequency segmentation and one for trill counting.
    Existing annotations are skipped to avoid duplication. Model
    predictions are provided to the annotator but final validation
    remains manual.

    Parameters
    ----------
    df_audio_files : pd.DataFrame
        DataFrame describing audio segments to annotate.
    annotator : str
        Identifier of the human annotator.
    config : object
        Configuration object containing paths and model parameters.
    debug : bool, optional
        If True, enable debug mode for model prediction (default: False).

    Returns
    -------
    None
    """
    df_existing = pd.read_csv(config.output_csv)

    seg_model = YOLO(os.path.join(config.models_subdir, config.selected_detection_model))   
        
    rows = []
    for i, (idx, row) in enumerate(df_audio_files.iterrows(), start=1):
        f = row["file_name_radical"]
        seg_id = row["syllable_rank"]
        if ((df_existing['file_name_radical'] == f) & (df_existing['segment_id'] == seg_id)).any():
            continue

        print(f"\n[{i}/{len(df_audio_files)}] Assisted annotation for {f}_seg{seg_id}")
        
        y_segment, sr, start, end, prediction = predict_box_and_trill_rate(row, config, seg_model, debug=debug)

        ann = annotate(y_segment, sr, config, prediction=prediction)

        # Validation (même logique)
        if ann["no_trill"]:
            if (ann["count"] != 0 or ann["t_min"] is None or ann["t_max"] is None):
                print("  ⚠️  Inconsistent no-trill, skipping")
                continue
        else:
            if (ann["t_min"] is None or ann["t_max"] is None or ann["f_min"] is None or 
                ann["f_max"] is None or ann["count"] is None or ann["count"] < 1):
                print("  ⊘ Skipped (incomplete)")
                continue

        rows.append({
            "file_name_radical": f,
            "species": row["species"],
            "segment_id": row["syllable_rank"],
            "annotator": annotator,
            "annotation_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "time_start": start,
            "time_end": end,
            "trill_t_start": ann["t_min"],
            "trill_t_end": ann["t_max"],
            "trill_min_freq": ann["f_min"],
            "trill_max_freq": ann["f_max"],
            "num_trills": ann["count"],
            "has_trill": not ann["no_trill"],
            "trill_duration": (ann["t_max"] - ann["t_min"]) if ann["t_min"] is not None and ann["t_max"] is not None else None,
        })

    if rows:
        df_new = pd.DataFrame(rows)
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
        os.makedirs(os.path.dirname(config.output_csv), exist_ok=True)
        df_out.to_csv(config.output_csv, index=False)
        print(f"\n✓ Added {len(df_new)} new annotations")

def trill_detection(config, debug=False):
    """
    Run YOLO-based trill detection on spectrogram images.

    The function loads a pre-trained YOLO segmentation model and applies it
    to a directory of spectrograms. Detection outputs (bounding boxes,
    confidence scores, and text files) are saved to disk for later use.

    Parameters
    ----------
    config : object
        Configuration object containing model paths and inference settings.
    debug : bool, optional
        Reserved for debugging purposes (currently unused).

    Returns
    -------
    None

    Notes 
    -----
    Spectrogram images should be pre-generated and stored 
    in the directory specified by `config.raw_spectrograms_subdir`.

    Generation is made using the `build_yolo_dataset.py` script, which extracts segments from
    """

    print("Trill detection YOLO model")
    segmentation_model = YOLO(os.path.join(config.models_subdir, config.selected_detection_model))
    abs_path = os.path.abspath(config.trill_detection_subdir)

    segmentation_model.predict(
        source=config.raw_spectrograms_subdir,
        save=True,
        save_txt=True,
        save_conf=True,
        conf=0.25,
        project=abs_path,
        # name="",
        exist_ok=True, 
        device=config.device,
        imgsz=640,
        max_det = 1
    )

def crop_and_save_trills(spec_image_path, yolo_txt_path, output_dir, target_size=(128,64)):
    """
    Crop trill regions from a spectrogram image using YOLO detections.

    The function reads a spectrogram image and its corresponding YOLO
    annotation file, converts normalized bounding boxes to pixel
    coordinates, extracts the detected regions, resizes them to a
    fixed target size, and saves them to disk.

    Parameters
    ----------
    spec_image_path : str
        Path to the spectrogram image (e.g. 640×640 PNG/JPG).
    yolo_txt_path : str
        Path to the corresponding YOLO annotation file
        (class_id, x_center, y_center, width, height; normalized).
    output_dir : str
        Directory where cropped images are saved.
    target_size : tuple of int, optional
        Output image size (height, width), default is (128, 64).

    Returns
    -------
    list of str
        Paths to the saved cropped images.
    """

    os.makedirs(output_dir, exist_ok=True)
    
    # Lire l'image
    spec = cv2.imread(spec_image_path, cv2.IMREAD_GRAYSCALE)  # [H,W]
    H, W = spec.shape
    
    # Lire le fichier YOLO
    with open(yolo_txt_path, "r") as f:
        lines = f.readlines()
    
    crops = []
    for i, line in enumerate(lines):
        if line.strip() == "":
            continue
        
        parts = line.strip().split()
        cls_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])  # normalisés [0,1]
        
        # Convertir en pixels
        x1 = int((xc - w/2) * W)
        y1 = int((yc - h/2) * H)
        x2 = int((xc + w/2) * W)
        y2 = int((yc + h/2) * H)
        
        # Clamp pour rester dans l'image
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(W,x2), min(H,y2)
        
        if x2 <= x1 or y2 <= y1:
            continue  # ignore les boîtes invalides
        
        # Crop + resize
        crop = spec[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Nom de fichier
        base_name = os.path.splitext(os.path.basename(spec_image_path))[0]
        crop_filename = os.path.join(output_dir, f"{base_name}.png")
        
        cv2.imwrite(crop_filename, crop_resized)
        crops.append(crop_filename)
    
    return crops

def shape_segments(detected_trill_dir, output_dir, target_size=(128,64)):
    """
    Extract and normalize detected trill segments from YOLO predictions.

    The function scans a YOLO prediction directory, matches spectrogram
    images with their corresponding label files, and extracts detected
    trill regions. Cropped segments are resized to a fixed target size
    and saved to the output directory.

    Parameters
    ----------
    detected_trill_dir : str
        Path to the YOLO output directory containing spectrogram images
        and a 'labels/' subdirectory.
    output_dir : str
        Directory where processed trill segments are saved.
    target_size : tuple of int, optional
        Output image size (height, width), default is (128, 64).

    Returns
    -------
    None
    """

    os.makedirs(output_dir, exist_ok=True)
    detected_trill_dir = os.path.join(detected_trill_dir, "predict/")
    
    labels_dir = os.path.join(detected_trill_dir, "labels/")
    if not os.path.exists(labels_dir):
        print(f"Warning: labels directory {labels_dir} does not exist.")
        return
    
    for file in os.listdir(detected_trill_dir):
        if file.endswith((".png", ".jpg", ".jpeg")):
            spec_image_path = os.path.join(detected_trill_dir, file)
            
            # Cherche le .txt correspondant dans labels/
            base_name = os.path.splitext(file)[0]
            yolo_txt_path = os.path.join(labels_dir, base_name + ".txt")
            
            if not os.path.exists(yolo_txt_path):
                continue  # pas de détection
            
            print(f"Processing {spec_image_path}")
            crop_and_save_trills(spec_image_path, yolo_txt_path, output_dir, target_size)

def predict_box_and_trill_rate(row_audio, config, seg_model, debug=False):
    """
    Predict trill location and estimate trill count for a given audio segment.

    The function extracts a centered audio segment, generates a temporary
    spectrogram, and applies a YOLO-based segmentation model to detect a
    trill region. If a detection is found, temporal and frequency bounds
    are converted to signal coordinates and a trill rate is estimated to
    derive the total number of trills.

    Parameters
    ----------
    row_audio : pd.Series
        Row describing the audio segment to process.
    config : object
        Configuration object containing paths and signal parameters.
    seg_model : object
        Pre-loaded YOLO segmentation model.
    debug : bool, optional
        If True, enable debug mode during trill rate estimation.

    Returns
    -------
    y_segment : np.ndarray
        Extracted audio segment.
    sr : int
        Sampling rate.
    start : float
        Start time of the extracted segment (seconds).
    end : float
        End time of the extracted segment (seconds).
    prediction : dict or None
        Predicted trill information (time/frequency bounds, confidence,
        estimated count), or None if no trill is detected.
    """

    audio_filename = row_audio["file_name_radical"]   
    y, sr = librosa.load(os.path.join(config.audio_subdir, audio_filename), sr=None)
    start = row_audio["time_start"]
    end = row_audio["time_end"]
    
    y_segment, start, end = extract_centered_segment(y, sr, start, end, config.win_len)

    out_path = os.path.join(config.temp_subdir, f"temp_spectro.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    save_spectrogram(y_segment, sr, out_path, config)

    # save_temp_spectro(y_segment,
    #                 sr,
    #                 out_path=out_path, 
    #                 n_fft=config.n_fft, 
    #                 hop_length=config.hop_length
    #                 )
    
    pred = seg_model.predict(
        source=out_path,
        save_conf=True,
        conf=0.25,
        device=config.device,
        imgsz=640,
        max_det = 1,
        verbose=False,
    )

    pred = pred[0]  # prendre la première (et unique) prédiction du batch   
    
    if pred.boxes is None or len(pred.boxes) == 0:
        # print("  ⊘ No trill detected by model, skipping")
        prediction = None
    else:
        box = pred.boxes.xyxyn[0].cpu().numpy()  # [x1, y1, x2, y2] normalisés
        conf = float(pred.boxes.conf[0])

        x1, y1, x2, y2 = box
        t_min, t_max, f_min, f_max = yolo_to_tf_xyxy(x1, 
                                                                    x2, 
                                                                    y1, 
                                                                    y2, 
                                                                    config.win_len, 
                                                                    sr/2)
        prediction = {
            "t_min": float(t_min),
            "t_max": float(t_max),
            "f_min": float(f_min),
            "f_max": float(f_max),
            "confidence": conf,
            "count": 0
        }

        trill_t_start = start + prediction["t_min"]
        trill_t_end = start + prediction["t_max"]
        trill_duration = trill_t_end - trill_t_start

        sample_start = int(trill_t_start * sr)
        sample_end = int(trill_t_end * sr)

        y_segment_cropped = y[int(sample_start):int(sample_end)]

        trill_rate = estimate_trill_rate(y_segment_cropped, sr, hop_length=config.hop_length, debug=debug)
        count = int(np.round(trill_rate * trill_duration))
        prediction["count"] = count

    return y_segment, sr, start, end, prediction

def predict_whole_dataset(df_audio_files, config, stop_debug = False):
    """
    Run trill detection and counting on an entire dataset.

    The function applies a YOLO-based segmentation model to each audio
    segment in the dataset, predicts trill time/frequency bounds, and
    estimates the number of trills per segment. Results are saved to
    a CSV file for downstream analysis.

    Parameters
    ----------
    df_audio_files : pd.DataFrame
        DataFrame describing the audio segments to process.
    config : object
        Configuration object containing paths and model parameters.
    stop_debug : bool, optional
        If True, stop after a small number of segments for debugging.

    Returns
    -------
    None
    """
    seg_model = YOLO(os.path.join(config.models_subdir, config.selected_detection_model))

    results = []
    for idx, row in tqdm(
        df_audio_files.iterrows(),
        total=len(df_audio_files),
        desc="🔍 YOLO trill detection",
        unit="segment"
    ):
        if stop_debug and idx >= 10:
            print("Debug mode: stopping after 10 segments")
            break

        _, _,seg_start, seg_end, prediction = predict_box_and_trill_rate(row, config, seg_model, debug=False)
        if prediction is not None:
            results.append({
                "file_name_radical": row["file_name_radical"],
                "segment_id": row["syllable_rank"],
                "seg_start": seg_start,
                "seg_end": seg_end,
                "t_min": prediction["t_min"],
                "t_max": prediction["t_max"],
                "f_min": prediction["f_min"],
                "f_max": prediction["f_max"],
                "confidence": prediction["confidence"],
                "count": prediction["count"]
            })
        else:
            results.append({
                "file_name_radical": row["file_name_radical"],
                "segment_id": row["syllable_rank"],
                "seg_start": seg_start,
                "seg_end": seg_end,
                "t_min": None,
                "t_max": None,
                "f_min": None,
                "f_max": None,
                "confidence": None,
                "count": 0
            })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(config.output_csv.replace(".csv", "_predictions.csv")), index=False)
    print(f"\n✓ Saved predictions for {len(df_results)} segments")

def visualize_and_confirm_predictions(config):
    """
    Visualize model predictions and allow manual confirmation.

    The function loads previously generated trill predictions, displays
    each predicted segment with its time and frequency bounds, and
    launches an interactive annotation interface to visually inspect
    and confirm the predictions.

    Parameters
    ----------
    config : object
        Configuration object containing paths and annotation parameters.

    Returns
    -------
    None
    """

    df_predictions = pd.read_csv(os.path.join(config.output_csv.replace(".csv", "_predictions.csv")))
    for idx, row in df_predictions.iterrows():
        if pd.isna(row["t_min"]) or pd.isna(row["t_max"]) or pd.isna(row["f_min"]) or pd.isna(row["f_max"]):
            continue
        
        prediction = {
            "t_min": row["t_min"],
            "t_max": row["t_max"],
            "f_min": row["f_min"],
            "f_max": row["f_max"],
            "confidence": row["confidence"],
            "count": row["count"]
        }
        seg_start, seg_end = row["seg_start"], row["seg_end"]

        print(f"\nVisualizing prediction for {row['file_name_radical']}_seg{row['segment_id']}")
        print(f"Prediction: T_min={prediction['t_min']:.2f}s, T_max={prediction['t_max']:.2f}s, F_min={prediction['f_min']:.1f}Hz, F_max={prediction['f_max']:.1f}Hz, Count={prediction['count']}")

        y, sr = librosa.load(os.path.join(config.audio_subdir, row["file_name_radical"]), sr=None)
        y_segment = y[int(seg_start * sr):int(seg_end * sr)]
        print(f"seg_start={seg_start:.2f}s, seg_end={seg_end:.2f}s, segment_duration={seg_end - seg_start:.2f}s")
        annotate(y_segment, sr, config, prediction=prediction)

def prepare_dataset(config, threshold_proba=0.99, clusters_to_include=["Fast", "Ultrafast"]):
    """
    Prepare the dataset by loading, filtering, and merging data.

    The function loads timestamp, metadata, and morphological data from
    CSV files, applies filtering based on GMM probabilities and specified
    clusters, and merges the datasets to create a final DataFrame ready
    for annotation or model training.

    Parameters
    ----------
    config : object
        Configuration object containing paths and parameters.
    threshold_proba : float, optional
        Minimum GMM probability to include a sample (default: 0.99).
    clusters_to_include : list of str, optional
        List of GMM cluster labels to include (default: ["Fast", "Ultrafast"]).

    Returns
    -------
    pd.DataFrame
        Merged and filtered DataFrame containing audio segment information,
        metadata, and morphological traits for the selected samples.
    """

    df_timestamps = pd.read_csv(os.path.join(config.data_raw_subdir, "segments_passerines_filtered.csv"))
    df_metadata = pd.read_csv(os.path.join(config.data_raw_subdir, "traits_data_pc_gmm_8components_proba_filtered.csv"))
    df_morpho = pd.read_csv(os.path.join(config.data_raw_subdir, "data_morpho.csv"))

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

    df_metadata_filtered = df_metadata_sub[
        (df_metadata_sub.apply(lambda row: row[f"gmm_prob_{row['gmm_cluster']}"] >= threshold_proba, axis=1)) &
        (df_metadata_sub['gmm_cluster_label'].isin(clusters_to_include))
    ]
    df_metadata_filtered.reset_index(drop=True, inplace=True)

    df_merged = pd.merge(df_timestamps, df_metadata_filtered, on="file_name", how="inner")
    df_merged = pd.merge(df_merged, df_morpho, on="species", how="inner")

    return df_merged

def get_remaining_samples(df_merged, df_done):
    """
    Return the subset of merged dataset that has not been annotated yet.

    This function compares the merged dataset of all segments (`df_merged`)
    with the already annotated samples (`df_done`) and returns only the
    rows that are not present in the annotations.

    Parameters
    ----------
    df_merged : pd.DataFrame
        DataFrame containing all audio segments with metadata.
    df_done : pd.DataFrame
        DataFrame containing already annotated segments with columns
        'file_name_radical' and 'segment_id'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only segments that still need annotation.
    """

    df_remaining = df_merged.merge(
        df_done[["file_name_radical", "segment_id"]],
        left_on=["file_name_radical", "syllable_rank"],
        right_on=["file_name_radical", "segment_id"],
        how="left",
        indicator=True
    )
    df_remaining = df_remaining[df_remaining["_merge"] == "left_only"]
    df_remaining = df_remaining.drop(columns=["_merge", "segment_id"])
    return df_remaining


def main():

    config = load_config_yaml(yaml_path="config.yaml")

    ANNOTATOR = input("Annotator name: ").strip()

    df_merged = prepare_dataset(config, threshold_proba=config.clusters_probability_threshold, clusters_to_include=config.trill_clusters)

    # Échantillonnage stratifié
    group_exclusion_list = []

    if os.path.exists(config.output_csv):
        df_done = pd.read_csv(config.output_csv)
        print(f"Found {len(df_done)} already annotated samples")
    else:
        df_done = pd.DataFrame()
        print("No existing annotation file found")

    if not df_done.empty:
        df_remaining = get_remaining_samples(df_merged, df_done)
        group_exclusion_list = df_done["species"].unique().tolist()
    else:
        df_remaining = df_merged.copy()

    CLUSTERS = config.trill_clusters
    df_remaining = df_remaining[df_remaining["gmm_cluster_label"].isin(CLUSTERS)]

    N_PER_SESSION = 2

    df_sample = constrained_stratified_sample(
        df_remaining,
        group_col="species",
        group_exclusion_list=group_exclusion_list,
        diversity_col="file_name_radical",
        max_per_group=2,
        n_samples=min(N_PER_SESSION, len(df_remaining))
    )

    # ════════════════════════════════════════════════════════
    # WORKFLOW
    # ════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("WORKFLOW OPTIONS")
    print("="*70)
    print("1. Manual annotation session")
    print("2. Assisted annotation session")
    print("3. Trill detection on test spectrograms (YOLO model) and shaping spectrograms for counting")
    print("4. Shaping detected trill segments for counting (if already done detection)")
    print("5. Predict whole dataset")
    print("6. Visualize and confirm predictions")
    choice = input("\nChoice (1-6): ").strip()

    if choice == "1":
        manual_annotation_session(df_sample, ANNOTATOR, config)
    
    elif choice == "2":
        assisted_annotation_session(df_sample, ANNOTATOR, config, debug=False)
    
    elif choice == "3":
        trill_detection(config)
        print("SHAPING DETECTED TRILL SEGMENTS...")
        shape_segments(config.trill_detection_subdir, config.trill_spectrograms_subdir)  # H, W

    elif choice == "4":
        shape_segments(config.trill_detection_subdir, config.trill_spectrograms_subdir)  # H, W

    elif choice == "5":
        predict_whole_dataset(df_remaining, config, stop_debug=False)

    elif choice == "6":
        visualize_and_confirm_predictions(config)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()