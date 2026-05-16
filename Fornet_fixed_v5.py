import os
import cv2
import itertools
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from keras.utils import to_categorical
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import MeanIoU
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────
DATA_ROOT = os.environ.get("FORNET_DATA_ROOT", r"D:\fornet")
OUT_DIR   = os.path.join(DATA_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

META_CSV = os.path.join(DATA_ROOT, "meta_data.csv")

# ─────────────────────────────────────────────
# 2. AUTO-DISCOVER IMAGE / MASK FOLDERS
# ─────────────────────────────────────────────
def _find_subdir(root, hint):
    for dp, _, fns in os.walk(root):
        base = os.path.basename(dp).lower()
        if hint in base and any(
            f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))
            for f in fns
        ):
            return dp
    return None

IMAGES_DIR = _find_subdir(DATA_ROOT, "image") or os.path.join(DATA_ROOT, "images")
MASKS_DIR  = _find_subdir(DATA_ROOT, "mask")  or os.path.join(DATA_ROOT, "masks")
VALID_EXTS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']

# ─────────────────────────────────────────────
# 3. FILE INDEX
# ─────────────────────────────────────────────
def index_files(folder):
    by_name, by_stem = {}, {}
    for f in Path(folder).rglob("*.*"):
        if f.is_file():
            by_name[f.name.lower()] = str(f)
            by_stem[f.stem.lower()] = str(f)
    return by_name, by_stem

IMG_BY_NAME, IMG_BY_STEM = index_files(IMAGES_DIR)
MSK_BY_NAME, MSK_BY_STEM = index_files(MASKS_DIR)

def resolve_path(by_name, by_stem, folder_path, name):
    s = str(name)
    hit = by_name.get(s.lower())
    if hit:
        return hit
    stem = Path(s).stem.lower()
    hit = by_stem.get(stem)
    if hit:
        return hit
    base = str(Path(folder_path) / Path(s).stem)
    for ext in VALID_EXTS:
        cand = base + ext
        if Path(cand).exists():
            return cand
    return None

# ─────────────────────────────────────────────
# 4. CONSTANTS
# ─────────────────────────────────────────────
PATCH_SIZE   = 256
IMG_CHANNELS = 3
N_CLASSES    = 2
N_SAMPLES    = 3500
BATCH_SIZE   = 64
EPOCHS       = 80
LR           = 0.0005
N_DISPLAY    = 8
CLASS_NAMES  = ["Background", "Foreground"]

# ─────────────────────────────────────────────
# 5. LOAD METADATA & RESOLVE PATHS
# ─────────────────────────────────────────────
df = pd.read_csv(META_CSV).iloc[:N_SAMPLES].reset_index(drop=True)

img_paths = [resolve_path(IMG_BY_NAME, IMG_BY_STEM, IMAGES_DIR, df['image'][i]) for i in range(len(df))]
msk_paths = [resolve_path(MSK_BY_NAME, MSK_BY_STEM, MASKS_DIR,  df['mask'][i])  for i in range(len(df))]

for i, (ip, mp) in enumerate(zip(img_paths, msk_paths)):
    if ip is None:
        raise FileNotFoundError(f"Row {i}: cannot find image '{df['image'][i]}' under {IMAGES_DIR}")
    if mp is None:
        raise FileNotFoundError(f"Row {i}: cannot find mask  '{df['mask'][i]}'  under {MASKS_DIR}")

# ─────────────────────────────────────────────
# 6. READ, RESIZE, NORMALISE
# ─────────────────────────────────────────────
image_dataset = []
mask_dataset  = []

for i in tqdm(range(len(df)), desc="Loading data"):
    img = cv2.imread(img_paths[i], cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cv2 cannot read: {img_paths[i]}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    image_dataset.append(img)

    mask = cv2.imread(msk_paths[i], cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"cv2 cannot read: {msk_paths[i]}")
    mask = cv2.resize(mask, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_NEAREST)
    mask_dataset.append(mask)

image_dataset = np.array(image_dataset, dtype=np.float32)
mask_dataset  = np.array(mask_dataset,  dtype=np.uint8)

# ─────────────────────────────────────────────
# 7. BINARISE & ONE-HOT ENCODE
# ─────────────────────────────────────────────
mask_dataset[mask_dataset > 0] = 1
labels_cat = to_categorical(
    mask_dataset,
    num_classes=N_CLASSES
)

# ─────────────────────────────────────────────
# 8. TRAIN / VAL / TEST SPLIT  (80 / 10 / 10)
# ─────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    image_dataset, labels_cat, test_size=0.20, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42)

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 9. U-NET MODEL
# ─────────────────────────────────────────────
def build_unet(n_classes=2, img_h=256, img_w=256, img_c=3):
    inputs = Input((img_h, img_w, img_c))

    c1 = Conv2D(16,  3, activation='relu', padding='same')(inputs)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(16,  3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(2)(c1)

    c2 = Conv2D(32,  3, activation='relu', padding='same')(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(32,  3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(2)(c2)

    c3 = Conv2D(64,  3, activation='relu', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64,  3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D(2)(c3)

    c4 = Conv2D(128, 3, activation='relu', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, 3, activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(2)(c4)

    c5 = Conv2D(256, 3, activation='relu', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, 3, activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(128, 2, strides=2, padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, 2, strides=2, padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64,  3, activation='relu', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64,  3, activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(32, 2, strides=2, padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32,  3, activation='relu', padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32,  3, activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(16, 2, strides=2, padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16,  3, activation='relu', padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(16,  3, activation='relu', padding='same')(c9)

    outputs = Conv2D(n_classes, 1, activation='softmax')(c9)
    return Model(inputs, outputs)

model = build_unet(n_classes=N_CLASSES)
model.compile(
    optimizer=optimizers.Adam(learning_rate=LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ─────────────────────────────────────────────
# 10. CALLBACKS
# ─────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        os.path.join(OUT_DIR, "best_model.keras"),
        monitor='val_loss', save_best_only=True, mode='min', verbose=1
    ),
]

# ─────────────────────────────────────────────
# 11. TRAIN
# ─────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    shuffle=True
)

# ─────────────────────────────────────────────
# 12. PREDICT ON TEST SET
# ─────────────────────────────────────────────
y_pred        = model.predict(X_test, verbose=1)
y_pred_argmax = np.argmax(y_pred,  axis=-1)
y_test_argmax = np.argmax(y_test,  axis=-1)

# flat arrays needed for sklearn metrics
y_true_flat  = y_test_argmax.flatten()
y_pred_flat  = y_pred_argmax.flatten()
y_score_flat = y_pred[:, :, :, 1].flatten()   # foreground softmax probability

# Mean IoU
iou_metric = MeanIoU(num_classes=N_CLASSES)
iou_metric.update_state(y_test_argmax, y_pred_argmax)
mean_iou = iou_metric.result().numpy()
print(f"\nMean IoU on test set: {mean_iou:.4f}")

# Save final model
model.save(os.path.join(OUT_DIR, "unet_model_final.keras"))
print(f"Model saved → {os.path.join(OUT_DIR, 'unet_model_final.keras')}")


# ═════════════════════════════════════════════════════════════════════════════
# GRAPH 1 — Train vs Val LOSS
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
epochs_range = range(1, len(history.history['loss']) + 1)
ax.plot(epochs_range, history.history['loss'],     color='#378ADD',
        linewidth=2, label='Train loss')
ax.plot(epochs_range, history.history['val_loss'], color='#D85A30',
        linewidth=2, label='Val loss',   linestyle='--')
ax.set_title('Train vs validation loss', fontsize=13)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Loss',  fontsize=11)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'graph1_loss_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved → graph1_loss_curves.png")


# ═════════════════════════════════════════════════════════════════════════════
# GRAPH 2 — Train vs Val ACCURACY
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs_range, history.history['accuracy'],     color='#378ADD',
        linewidth=2, label='Train accuracy')
ax.plot(epochs_range, history.history['val_accuracy'], color='#D85A30',
        linewidth=2, label='Val accuracy', linestyle='--')
ax.set_title('Train vs validation accuracy', fontsize=13)
ax.set_xlabel('Epoch',    fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'graph2_accuracy_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved → graph2_accuracy_curves.png")


# ═════════════════════════════════════════════════════════════════════════════
# GRAPH 3 — Confusion Matrix (counts)
# ═════════════════════════════════════════════════════════════════════════════
cm = confusion_matrix(y_true_flat, y_pred_flat)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_title('Confusion matrix — pixel counts', fontsize=13)
ax.set_xlabel('Predicted label', fontsize=11)
ax.set_ylabel('True label',      fontsize=11)
ax.set_xticks([0, 1]);  ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_yticks([0, 1]);  ax.set_yticklabels(CLASS_NAMES, fontsize=11)
thresh = cm.max() / 2.0
for i, j in itertools.product(range(2), range(2)):
    ax.text(j, i, f'{cm[i,j]:,}',
            ha='center', va='center', fontsize=13,
            color='white' if cm[i,j] > thresh else 'black')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'graph3_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved → graph3_confusion_matrix.png")


# ═════════════════════════════════════════════════════════════════════════════
# GRAPH 4 — Prediction Heatmaps
#   5 columns: Input | Ground truth | Prediction | Prob heatmap | Error map
# ═════════════════════════════════════════════════════════════════════════════
N_ROWS     = 6
col_titles = ['Input image', 'Ground truth', 'Prediction',
              'Foreground probability', 'Error map']

fig, axes = plt.subplots(N_ROWS, 5, figsize=(18, N_ROWS * 3.2))
for col, t in enumerate(col_titles):
    axes[0, col].set_title(t, fontsize=11, fontweight='bold', pad=8)

for row in range(N_ROWS):
    img      = X_test[row]
    gt       = y_test_argmax[row]
    pred     = y_pred_argmax[row]
    prob_map = y_pred[row, :, :, 1]
    err      = (pred != gt).astype(np.uint8)

    err_rgb = np.ones((*err.shape, 3), dtype=np.float32) * 0.85
    err_rgb[err == 1] = [1.0, 0.2, 0.2]

    axes[row, 0].imshow(img);                              axes[row, 0].axis('off')
    axes[row, 1].imshow(gt,      cmap='gray', vmin=0, vmax=1); axes[row, 1].axis('off')
    axes[row, 2].imshow(pred,    cmap='gray', vmin=0, vmax=1); axes[row, 2].axis('off')
    hm = axes[row, 3].imshow(prob_map, cmap='jet', vmin=0, vmax=1)
    axes[row, 3].axis('off')
    plt.colorbar(hm, ax=axes[row, 3], fraction=0.046, pad=0.04)
    axes[row, 4].imshow(err_rgb);                          axes[row, 4].axis('off')
    axes[row, 0].set_ylabel(f'Sample {row + 1}', fontsize=10)

plt.suptitle('Prediction heatmaps and error maps', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'graph4_heatmaps.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved → graph4_heatmaps.png")


# ═════════════════════════════════════════════════════════════════════════════
# GRAPH 5 — ROC Curve (foreground class)
# ═════════════════════════════════════════════════════════════════════════════
fpr, tpr, _ = roc_curve(y_true_flat, y_score_flat)
roc_auc     = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='#378ADD', lw=2,
        label=f'ROC curve  (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='#aaa', lw=1,
        linestyle='--', label='Random classifier')
ax.fill_between(fpr, tpr, alpha=0.08, color='#378ADD')
ax.set_xlim([0.0, 1.0]);  ax.set_ylim([0.0, 1.02])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate',  fontsize=12)
ax.set_title('ROC curve — foreground class', fontsize=13)
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'graph5_roc_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved → graph5_roc_curve.png")


# ═════════════════════════════════════════════════════════════════════════════
# GRAPH 6 — Final prediction output images
#   3 columns: Input | Ground truth | Predicted mask
# ═════════════════════════════════════════════════════════════════════════════
for i in range(N_DISPLAY):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(X_test[i])
    axes[0].set_title('Input image',      fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(y_test_argmax[i], cmap='gray')
    axes[1].set_title('Ground truth mask', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(y_pred_argmax[i], cmap='gray')
    axes[2].set_title('Predicted mask',    fontsize=12)
    axes[2].axis('off')

    plt.suptitle(f'Test sample {i + 1}  —  '
                 f'IoU ≈ {mean_iou:.4f}', fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'graph6_prediction_{i+1:02d}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → graph6_prediction_{i+1:02d}.png")

print(f"\nAll outputs saved to: {OUT_DIR}")
print(f"Final Mean IoU: {mean_iou:.4f}")