from collections import defaultdict
from pathlib import Path
import random
import shutil
import yaml
from src.config import load_config_yaml


def extract_species(stem: str):
    parts = stem.split("_")
    return "_".join(parts[:2])  # Genus_species


def copy_files(species_set, split_name, species_to_files):
    for species in species_set:
        for img in species_to_files[species]:
            shutil.copy(
                img,
                OUT / f"images/{split_name}/{img.name}"
            )
            shutil.copy(
                LBL / f"{img.stem}.txt",
                OUT / f"labels/{split_name}/{img.stem}.txt"
            )

YAML_PATH = "config.yaml"
cfg = load_config_yaml(YAML_PATH)
ROOT = Path(cfg.yolo_dataset_subdir)
IMG = ROOT / "images"
LBL = ROOT / "labels"

OUT = Path(ROOT) / "split"

for p in ["images/train", "images/val", "labels/train", "labels/val"]:
    (OUT / p).mkdir(parents=True, exist_ok=True)

species_to_files = defaultdict(list)

for img_file in IMG.glob("*.png"):
    species = extract_species(img_file.stem)
    species_to_files[species].append(img_file)

species_list = list(species_to_files.keys())
random.shuffle(species_list)

split = int((1-cfg.val_split) * len(species_list))
train_species = set(species_list[:split])
val_species = set(species_list[split:])

copy_files(train_species, "train", species_to_files)
copy_files(val_species, "val", species_to_files)

data_yaml = {
    "path": str(OUT),
    "train": "images/train",
    "val": "images/val",
    "nc": 1,
    "names": ["trill"]
}

with open(OUT / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f)