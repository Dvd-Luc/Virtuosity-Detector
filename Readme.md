# Virtuosity Detector

Automatic detection of trills and complex vocalizations in passerine birds from audio recordings, using object detection on spectrograms.

---

## What does this project do?

**Virtuosity Detector** analyzes audio recordings of passerine birds and automatically detects trill sequences. The processing pipeline:

1. Converts audio recordings into **spectrograms** 
2. Applies an **automatic detection model** (YOLOv8) trained to recognize trills
3. Produces **annotations** and **counts** of vocal events (notes)
4. Enables cross-referencing of results with morphological and biological trait data

---

## Requirements

Before getting started, you will need:

- Data files (.csv) are needed for running this project

> A NVIDIA GPU is recommended for model training, but is **not required** to run the detector on new recordings.

---

## Getting started

### Step 1 — Install Python

If Python is not yet installed on your machine:

- **Windows / macOS**: download the installer from [python.org](https://www.python.org/downloads/). Make sure to check the **"Add Python to PATH"** box during installation.
- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt update && sudo apt install python3.10 python3.10-venv python3-pip
  ```

To verify that Python is correctly installed, open a terminal and type:
```bash
python --version
# You should see something like: Python 3.10.x
```

---

### Step 2 — Get the project

**Option A — Without command line**: click the green **"Code"** button on GitHub, then **"Download ZIP"**. Extract the archive to the folder of your choice.

**Option B — Via terminal (recommended)**:
```bash
git clone https://github.com/your-username/virtuosity-detector.git
cd virtuosity-detector
```

---

### Step 3 — Create an isolated environment

A virtual environment installs the project's dependencies without affecting the rest of your Python system. This is strongly recommended.

```bash
# Create the environment (only once)
python -m venv .venv

# Activate it — macOS / Linux
source .venv/bin/activate

# Activate it — Windows
.venv\Scripts\activate
```

---

### Step 4 — Install the project and its dependencies

```bash
pip install -e .
```

This command automatically installs all required libraries (audio processing, object detection, data manipulation, etc.). Installation may take a few minutes.

---

### Step 5 — Set up your local configuration

```bash
cp config.example.yaml config.yaml
```

Open `config.yaml` with any text editor and fill in your local paths and other parameters (model):

> ⚠️ This file is specific to your machine and will never be shared on GitHub.

---

## Usage

```bash
# Run the detection pipeline
python src/main.py

# Run the virtuosity analysis
python src/tools/virtuosity_main.py
```
---

## Data

Place your data files in `data/raw/`. The expected files are:

| File | Description |
|---|---|
| `segments_passerines_filtered.csv` | Filtered audio segments |
| `data_morpho.csv` | Individual morphological measurements |
| `traits_data_pc_gmm_8components_proba_filtered.csv` | Biological traits (PCA / GMM) |

---

## Project Structure

```
.
├── config.yaml                 # Local configuration (paths parameters)
├── data/
│   └── raw/                    # Input CSV annotations and trait data (not versioned)
│   ├── processed/              # Pred + annotations CSV files 
│
└── src/
    ├── main.py                 # Main entry point
    ├── config.py               # Configuration loader
    │
    ├── models/                 # YOLO Models folder
    │   ├── best_old.pt
    │   ├── Yolov8s_med_640.pt        
    │
    ├── utils/
    │   ├── annotate.py         # Annotation utilities
    │   ├── counting.py         # Vocal event counting
    │   ├── segmentation.py     # Audio segmentation
    │   └── spectrograms.py     # Spectrogram generation
    ├── tools/
    │   ├── csv_cleaner.ipynb   # Remove specific elements using
    │   ├── metric_tests.ipynb  # Test to chechk metrics relevance
    │   |
    │   ├── segments_manipulation.ipynb # Export pred wav files
    │   |
    │   └── virtuosity_main.py  # High-level analysis pipeline
    └── yolo/
        ├── build_yolo_dataset.py # Dataset preparation
        ├── prepare_yolo.py       # YOLO format conversion
        ├── yolo_training.py      # Model training script
        └── colab_training_script.ipynb # Google Colab Script
```

---

## Troubleshooting

**`ModuleNotFoundError` on startup**
→ Make sure your virtual environment is active (`source .venv/bin/activate`) and re-run `pip install -e .`

**`FileNotFoundError` on a CSV or `.pt` file**
→ Check the paths defined in your `config.yaml`

**Error related to `torch` / CUDA**
→ If you do not have a NVIDIA GPU, install the CPU-only version of PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```