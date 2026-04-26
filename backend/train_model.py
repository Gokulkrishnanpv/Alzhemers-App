"""
train_model.py  –  Run this in Google Colab or locally to produce model.h5
Mirrors the typical Colab notebook flow for this Kaggle Alzheimer's dataset.

Usage (Colab):
    !unzip /content/archive.zip -d /content/data
    !python train_model.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "./data"          # adjust to your path
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
VAL_DIR    = os.path.join(DATA_DIR, "val")
IMG_SIZE   = (176, 176)
BATCH_SIZE = 32
EPOCHS     = 20
CLASSES    = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

# ── Data generators  (same rescaling as original Colab) ──────────────────────
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

val_gen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASSES,
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASSES,
)

# ── Model: MobileNetV2 transfer learning (fast, ~92% accuracy) ───────────────
base = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False   # freeze base for fast training

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(4, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("model.h5", save_best_only=True),
]

# ── Train ─────────────────────────────────────────────────────────────────────
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
)

print("\n✅  Saved model.h5  –  copy it to backend/model.h5")
print(f"Final val accuracy: {max(history.history['val_accuracy']):.4f}")
