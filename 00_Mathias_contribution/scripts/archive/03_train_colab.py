"""
03_train_colab.py (Archiv)
---------------------------
Kurz: Colab-Version des Trainings (zum Kopieren in Zellen). Wird durch das
Notebook 03_train_colab.ipynb bzw. Colab Notebook/BA_Thesis_Model_Training_...
ersetzt. Mountet Drive, installiert Pakete und trainiert Prithvi-EO v2 auf
GhanaMiningPrithvi (SmallMinesDS); Checkpoints landen auf Drive.
"""
# =============================================================================
# GOOGLE COLAB – Prithvi-EO v2 Training auf SmallMinesDS
# =============================================================================
# Anleitung:
# 1. colab.research.google.com → Neues Notebook
# 2. Runtime → Change runtime type → GPU (T4) → Save
# 3. Drei Zellen anlegen und je einen der folgenden Blöcke reinkopieren und ausführen.
# =============================================================================


# ============ ZELLE 1: Drive mounten (zuerst ausführen, im Browser freigeben) ============

from google.colab import drive
drive.mount('/content/drive')


# ============ ZELLE 2: Pakete installieren (einmalig, dauert 1–2 Min) ============

!pip install -q terratorch==0.99.7 segmentation-models-pytorch==0.3.4
!pip install -q lightning==2.4.0 albumentations==1.4.10
!pip install -q rasterio==1.3.11


# ============ ZELLE 3: Training (komplett ab hier kopieren und in einer Zelle ausführen) ============

import os
import torch
from segmentation_models_pytorch.encoders import encoders as smp_encoders
import rasterio
import numpy as np
from terratorch.models import PrithviModelFactory
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.tasks import SemanticSegmentationTask
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Pfade (angepasst für typische Drive-Struktur: Ordner in Ordner) ───────────
DATASET_PATH   = '/content/drive/MyDrive/GhanaMiningPrithvi/GhanaMiningPrithvi'
CHECKPOINT_DIR = '/content/drive/MyDrive/checkpoints/prithvi-v2-300'

# Prüfen, ob training/ und validation/ existieren und Dateien haben
train_dir = os.path.join(DATASET_PATH, 'training')
val_dir   = os.path.join(DATASET_PATH, 'validation')
if not os.path.isdir(train_dir):
    raise FileNotFoundError(f"Ordner nicht gefunden: {train_dir}\nAuf Drive muss unter 'GhanaMiningPrithvi' ein Ordner 'GhanaMiningPrithvi' mit 'training' und 'validation' liegen.")
if not os.path.isdir(val_dir):
    raise FileNotFoundError(f"Ordner nicht gefunden: {val_dir}\nBitte 'validation' auf Drive hochladen.")
n_train = len([f for f in os.listdir(train_dir) if f.endswith('_IMG.tif')])
n_val   = len([f for f in os.listdir(val_dir) if f.endswith('_IMG.tif')])
print(f"Training:   {n_train} Patches in {train_dir}")
print(f"Validation: {n_val} Patches in {val_dir}")
if n_train == 0 or n_val == 0:
    raise ValueError("training oder validation ist leer. Bitte Datenstruktur auf Drive prüfen.")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Band-Konfiguration ───────────────────────────────────────────────────────
ghana_mining_bands = ["BLUE", "GREEN", "RED", "VNIR_5", "SWIR_1", "SWIR_2"]
means = [1473.81388377, 1703.35249650, 1696.67685941, 3832.39764247, 3156.11122121, 2226.06822112]
stds  = [ 223.43533204,  285.53613398,  413.82320306,  389.61483882,  451.49534791,  468.26765909]

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    ToTensorV2()
])

datamodule = GenericNonGeoSegmentationDataModule(
    batch_size=4,
    num_workers=2,
    train_data_root=os.path.join(DATASET_PATH, 'training'),
    val_data_root=os.path.join(DATASET_PATH, 'validation'),
    test_data_root=os.path.join(DATASET_PATH, 'validation'),
    img_grep="*_IMG.tif",
    label_grep="*_MASK.tif",
    means=means,
    stds=stds,
    num_classes=2,
    train_transform=train_transform,
    dataset_bands=ghana_mining_bands,
    output_bands=ghana_mining_bands,
    no_data_replace=0,
    no_label_replace=-1,
)

model_args = {
    "backbone": "prithvi_eo_v2_300",
    "bands": ghana_mining_bands,
    "in_channels": 6,
    "num_classes": 2,
    "pretrained": True,
    "decoder": "UperNetDecoder",
    "rescale": True,
    "backbone_num_frames": 1,
    "head_dropout": 0.1,
    "decoder_scale_modules": True,
}

task = SemanticSegmentationTask(
    model_args=model_args,
    model_factory="PrithviModelFactory",
    loss="ce",
    lr=1e-3,
    ignore_index=-1,
    optimizer="AdamW",
    optimizer_hparams={"weight_decay": 0.05},
    freeze_backbone=True,
    class_names=['Non_mining', 'Mining'],
)

datamodule.setup("fit")

checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINT_DIR,
    monitor=task.monitor,
    save_top_k=1,
    save_last=True,
    filename="prithvi-v2-300-{epoch:02d}-{val_loss:.4f}",
)
early_stopping = EarlyStopping(monitor=task.monitor, min_delta=0.001, patience=10)
logger = TensorBoardLogger(save_dir=CHECKPOINT_DIR, name='logs')

trainer = Trainer(
    devices=1,
    precision="16-mixed",
    callbacks=[RichProgressBar(), checkpoint_callback, early_stopping],
    logger=logger,
    max_epochs=50,
    default_root_dir=CHECKPOINT_DIR,
    log_every_n_steps=1,
    check_val_every_n_epoch=1,
)

print("Training startet (ca. 3–5 h auf T4)...")
print(f"Checkpoints: {CHECKPOINT_DIR}\n")
trainer.fit(model=task, datamodule=datamodule)

print("\nTraining fertig. Test-Auswertung...")
res = trainer.test(model=task, datamodule=datamodule)
print("Test-Ergebnis:", res)
print("Bester Checkpoint:", checkpoint_callback.best_model_path)
print("Datei von Drive herunterladen und lokal als models/prithvi-v2-300-best.ckpt speichern.")
