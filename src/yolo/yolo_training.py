import albumentations as A
from ultralytics import YOLO
import os

def get_spectrogram_augmentations(intensity="medium"):
    """
    Augmentations optimisées pour spectrogrammes de trills.
    
    Parameters:
    -----------
    intensity : str
        "light", "medium" ou "strong"
    """
    
    if intensity == "light":
        return A.Compose([
            # Gamma (meilleur que Brightness pour spectro)
            A.RandomGamma(gamma_limit=(90, 110), p=0.3),
            
            # CLAHE (améliore contraste local)
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            
            # Bruit très léger
            A.GaussNoise(var_limit=(1.0, 8.0), mean=0, p=0.2),
            
            # Blur léger
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            
            # SpecAugment très léger (petits trous)
            A.CoarseDropout(
                max_holes=2,
                max_height=30,
                max_width=10,
                min_holes=1,
                min_height=10,
                min_width=5,
                fill_value=0,
                p=0.25
            ),
        ])
    
    elif intensity == "medium":
        return A.Compose([
            # === 1️⃣ Dynamique / Contraste ===
            A.RandomGamma(
                gamma_limit=(85, 115),  # Comme concurrent
                p=0.4
            ),
            
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=0.3
            ),
            
            # === 2️⃣ Bruit (légèrement plus fort que concurrent) ===
            A.GaussNoise(
                var_limit=(1.0, 20.0),  # Compromis entre vous et concurrent
                mean=0,
                p=0.35
            ),
            
            # === 3️⃣ Perte de résolution ===
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.25
            ),
            
            # === 4️⃣ SpecAugment hybride (trous + bandes) ===
            # Petits trous (style concurrent)
            A.CoarseDropout(
                max_holes=3,
                max_height=40,   # Masking fréquentiel
                max_width=15,    # Masking temporel léger
                min_holes=1,
                min_height=15,
                min_width=5,
                fill_value=0,
                p=0.35
            ),
            
            # Bandes fréquentielles (ma proposition)
            A.CoarseDropout(
                max_holes=1,
                max_height=50,   # Bande fréquentielle
                max_width=640,   # Toute la largeur
                min_holes=1,
                min_height=20,
                min_width=640,
                fill_value=0,
                p=0.2  # Moins fréquent
            ),
        ])
    
    else:  # "strong"
        return A.Compose([
            # === 1️⃣ Dynamique / Contraste ===
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.5
            ),
            
            A.CLAHE(
                clip_limit=3.0,  # Plus agressif
                tile_grid_size=(8, 8),
                p=0.4
            ),
            
            # === 2️⃣ Bruit ===
            A.GaussNoise(
                var_limit=(5.0, 30.0),
                mean=0,
                p=0.45
            ),
            
            # === 3️⃣ Perte de résolution ===
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            
            # === 4️⃣ SpecAugment fort ===
            # Petits trous multiples
            A.CoarseDropout(
                max_holes=5,
                max_height=50,
                max_width=20,
                min_holes=2,
                min_height=15,
                min_width=5,
                fill_value=0,
                p=0.4
            ),
            
            # Bandes fréquentielles
            A.CoarseDropout(
                max_holes=2,
                max_height=60,
                max_width=640,
                min_holes=1,
                min_height=25,
                min_width=640,
                fill_value=0,
                p=0.3
            ),
            
            # Bandes temporelles
            A.CoarseDropout(
                max_holes=1,
                max_height=640,
                max_width=80,
                min_holes=1,
                min_height=640,
                min_width=30,
                fill_value=0,
                p=0.25
            ),
        ])

# ====================
# ENTRAÎNEMENT RECOMMANDÉ
# ====================

RootDir = "path/to/project/root/"

model = YOLO("yolov8n.pt")

# Commencez avec "medium"
custom_transforms = get_spectrogram_augmentations(intensity="medium")

model.train(
    data=os.path.join(RootDir, "yolo_final/t4/data.yaml"),
    imgsz=640,
    epochs=150,
    batch=8,  # Augmentez à 16 si vous avez assez de RAM
    device="cpu",
    workers=2,
    cache=True,  # ✅ Activé pour 600 images
    amp=True,
    
    # === AUGMENTATIONS ULTRALYTICS (géométriques) ===
    # Comme le concurrent n'en propose pas, gardons-les légères
    translate=0.15,     # ±15% translation (léger)
    scale=0.15,         # ±15% zoom
    fliplr=0.0,         # Pas de flip horizontal
    flipud=0.0,         # Pas de flip vertical
    mosaic=0.0,         # Désactivé
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,          # Gamma gère déjà la luminosité
    degrees=0.0,        # Pas de rotation
    perspective=0.0,    # Pas de perspective
    erasing=0.0,        # CoarseDropout gère déjà le masking
    
    # === AUGMENTATIONS ALBUMENTATIONS ===
    augmentations=custom_transforms,
    
    # === AUTRES PARAMÈTRES ===
    project=os.path.join(RootDir, "runs_yolo"),
    name="trill_yolov8n_hybrid_aug_v1",
    patience=25,
    save=True,
    save_period=10,
    
    # Optimizer settings (optionnel)
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
)


"""
from ultralytics import YOLO
import albumentations as A


custom_transforms = [A.GaussNoise(var_limit=(10.0, 50.0), p=0.5)]
model = YOLO("yolov8n.pt")

model.train(
data="yolo_final/data.yaml",
imgsz=640,
epochs=100,
batch=4, 
device="cuda",
workers=2,
cache=False,
amp=True,
mosaic=False,
translate=0.05,
scale=0.1,
augmentations=custom_transforms
)
"""
