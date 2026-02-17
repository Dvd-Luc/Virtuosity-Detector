from dataclasses import dataclass, field
from typing import Optional
import torch
import os
import yaml

@dataclass
class TrillConfig:
    # ROOT DIR
    root_dir: str = "./data"

    # Subdirectories
    data_raw_subdir: str = "dataset"
    data_processed_subdir: str = "dataset/processed"
    audio_subdir: Optional[str] = "audio"
    raw_spectrograms_subdir: str = "spectrograms/test"
    trill_detection_subdir: str = "runs/trill_detection"
    trill_spectrograms_subdir: str = "counting_inputs"
    temp_subdir: str = "temp"
    models_subdir: str = "models"
    yolo_dataset_subdir: str = "yolo_dataset"

    # File names
    output_csv: str = "counting_annotations.csv"
    test_csv: str = "test_annotations.csv"
    selected_detection_model: str = "best.pt"

    
    # Type de modèles
    count_use_log: bool = True
    
    # Audio
    sr: int = 44100
    n_fft: int = 256
    hop_length: int = 64
    win_len: float = 2.0

    # Spectrogram parameters
    img_size: int = 640
    cmap: str = "gray"
    
    # Training
    batch_size: int = 16
    epochs_seg: int = 20
    patience : int = 8
    epochs_count: int = 15
    
    val_split: float = 0.15
    
    # Device for training (CPU or GPU)
    # GPU only needed for training, not necessary if you already have trained models and just want to run inference

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Trill Types
    trill_clusters: list = field(default_factory=lambda: ["Fast", "UltraFast"])
    clusters_probability_threshold: float = 0.99

    # ===== Internes (non init) =====
    _seg_model_path: str = field(init=False)

    def __post_init__(self):
        self._build_paths()

    def _build_paths(self):
        # Dossiers
        self.data_raw_subdir = os.path.join(self.root_dir, self.data_raw_subdir)
        self.data_processed_subdir = os.path.join(self.root_dir, self.data_processed_subdir)
        self.yolo_dataset_subdir = os.path.join(self.root_dir, self.yolo_dataset_subdir)
        
        if self.audio_subdir is None:
            self.audio_dir = None
        elif os.path.isabs(self.audio_subdir):
            self.audio_dir = self.audio_subdir
        else:
            self.audio_dir = os.path.join(self.root_dir, self.audio_subdir)
        self.raw_spectrograms_subdir = os.path.join(self.root_dir, self.raw_spectrograms_subdir)
        self.trill_detection_subdir = os.path.join(self.root_dir, self.trill_detection_subdir)
        self.trill_spectrograms_subdir = os.path.join(self.trill_detection_subdir, self.trill_spectrograms_subdir)
        self.models_subdir = os.path.join(self.root_dir, self.models_subdir)
        self.temp_subdir = os.path.join(self.root_dir, self.temp_subdir)
        # Fichiers
        self.output_csv = os.path.join(self.data_processed_subdir, self.output_csv)
        self.test_csv = os.path.join(self.data_processed_subdir, self.test_csv)

        # Modèles
        self._seg_model_path = os.path.join(self.models_subdir, self.selected_detection_model)
       
    
def load_config_yaml(yaml_path="config.yaml"):
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return TrillConfig(**config_dict)
