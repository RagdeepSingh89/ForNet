{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3b3a1737",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import cv2\n",
    "import itertools\n",
    "import numpy as np\n",
    "from pathlib import Path\n",
    "import pandas as pd\n",
    "from tqdm import tqdm\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import confusion_matrix, roc_curve, auc\n",
    "from keras.utils import to_categorical\n",
    "from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout\n",
    "from keras.models import Model\n",
    "from keras import optimizers\n",
    "from keras.callbacks import EarlyStopping, ModelCheckpoint\n",
    "from keras.metrics import MeanIoU\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.gridspec as gridspec"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "87feb4b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ─────────────────────────────────────────────\n",
    "# 1. PATHS\n",
    "# ─────────────────────────────────────────────\n",
    "DATA_ROOT = os.environ.get(\"FORNET_DATA_ROOT\", r\"D:\\fornet\")\n",
    "OUT_DIR   = os.path.join(DATA_ROOT, \"outputs\")\n",
    "os.makedirs(OUT_DIR, exist_ok=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ee986f44",
   "metadata": {
    "lines_to_next_cell": 1
   },
   "outputs": [],
   "source": [
    "META_CSV = os.path.join(DATA_ROOT, \"meta_data.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "503c924d",
   "metadata": {
    "lines_to_next_cell": 1
   },
   "outputs": [],
   "source": [
    "# ─────────────────────────────────────────────\n",
    "# 2. AUTO-DISCOVER IMAGE / MASK FOLDERS\n",
    "# ─────────────────────────────────────────────\n",
    "def _find_subdir(root, hint):\n",
    "    for dp, _, fns in os.walk(root):\n",
    "        base = os.path.basename(dp).lower()\n",
    "        if hint in base and any(\n",
    "            f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))\n",
    "            for f in fns\n",
    "        ):\n",
    "            return dp\n",
    "    return None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "86cf7fa2",
   "metadata": {
    "lines_to_next_cell": 1
   },
   "outputs": [],
   "source": [
    "IMAGES_DIR = _find_subdir(DATA_ROOT, \"image\") or os.path.join(DATA_ROOT, \"images\")\n",
    "MASKS_DIR  = _find_subdir(DATA_ROOT, \"mask\")  or os.path.join(DATA_ROOT, \"masks\")\n",
    "VALID_EXTS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "952e8f4e",
   "metadata": {
    "lines_to_next_cell": 1
   },
   "outputs": [],
   "source": [
    "# ─────────────────────────────────────────────\n",
    "# 3. FILE INDEX\n",
    "# ─────────────────────────────────────────────\n",
    "def index_files(folder):\n",
    "    by_name, by_stem = {}, {}\n",
    "    for f in Path(folder).rglob(\"*.*\"):\n",
    "        if f.is_file():\n",
    "            by_name[f.name.lower()] = str(f)\n",
    "            by_stem[f.stem.lower()] = str(f)\n",
    "    return by_name, by_stem"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5a7ea591",
   "metadata": {
    "lines_to_next_cell": 1
   },
   "outputs": [],
   "source": [
    "IMG_BY_NAME, IMG_BY_STEM = index_files(IMAGES_DIR)\n",
    "MSK_BY_NAME, MSK_BY_STEM = index_files(MASKS_DIR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8b1e6419",
   "metadata": {
    "lines_to_next_cell": 1
   },
   "outputs": [],
   "source": [
    "def resolve_path(by_name, by_stem, folder_path, name):\n",
    "    s = str(name)\n",
    "    hit = by_name.get(s.lower())\n",
    "    if hit:\n",
    "        return hit\n",
    "    stem = Path(s).stem.lower()\n",
    "    hit = by_stem.get(stem)\n",
    "    if hit:\n",
    "        return hit\n",
    "    base = str(Path(folder_path) / Path(s).stem)\n",
    "    for ext in VALID_EXTS:\n",
    "        cand = base + ext\n",
    "        if Path(cand).exists():\n",
    "            return cand\n",
    "    return None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "64c9c9b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ─────────────────────────────────────────────\n",
    "# 4. CONSTANTS\n",
    "# ─────────────────────────────────────────────\n",
    "PATCH_SIZE   = 256\n",
    "IMG_CHANNELS = 3\n",
    "N_CLASSES    = 2\n",
    "N_SAMPLES    = 3500\n",
    "BATCH_SIZE   = 64\n",
    "EPOCHS       = 80\n",
    "LR           = 0.0005\n",
    "N_DISPLAY    = 8\n",
    "CLASS_NAMES  = [\"Background\", \"Foreground\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a4f38a6e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ─────────────────────────────────────────────\n",
    "# 5. LOAD METADATA & RESOLVE PATHS\n",
    "# ─────────────────────────────────────────────\n",
    "df = pd.read_csv(META_CSV).iloc[:N_SAMPLES].reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0146e826",
   "metadata": {},
   "outputs": [],
   "source": [
    "img_paths = [resolve_path(IMG_BY_NAME, IMG_BY_STEM, IMAGES_DIR, df['image'][i]) for i in range(len(df))]\n",
    "msk_paths = [resolve_path(MSK_BY_NAME, MSK_BY_STEM, MASKS_DIR,  df['mask'][i])  for i in range(len(df))]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "89f7a140",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i, (ip, mp) in enumerate(zip(img_paths, msk_paths)):\n",
    "    if ip is None:\n",
    "        raise FileNotFoundError(f\"Row {i}: cannot find image '{df['image'][i]}' under {IMAGES_DIR}\")\n",
    "    if mp is None:\n",
    "        raise FileNotFoundError(f\"Row {i}: cannot find mask  '{df['mask'][i]}'  under {MASKS_DIR}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e1789c94",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ─────────────────────────────────────────────\n",
    "# 6. READ, RESIZE, NORMALISE\n",
    "# ─────────────────────────────────────────────\n",
    "image_dataset = []\n",
    "mask_dataset  = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a08db10a",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in tqdm(range(len(df)), desc=\"Loading data\"):\n",
    "    img = cv2.imread(img_paths[i], cv2.IMREAD_COLOR)\n",
    "    if img is None:\n",
    "        raise FileNotFoundError(f\"cv2 cannot read: {img_paths[i]}\")\n",
    "    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "    img = cv2.resize(img, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_LINEAR)\n",
    "    img = img.astype(np.float32) / 255.0\n",
    "    image_dataset.append(img)\n",
    "\n",
    "    mask = cv2.imread(msk_paths[i], cv2.IMREAD_GRAYSCALE)\n",
    "    if mask is None:\n",
    "        raise FileNotFoundError(f\"cv2 cannot read: {msk_paths[i]}\")\n",
    "    mask = cv2.resize(mask, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_NEAREST)\n",
    "    mask_dataset.append(mask)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "88deb167",
   "metadata": {},
   "outputs": [],
   "source": [
    "image_dataset = np.array(image_dataset, dtype=np.float32)\n",
    "mask_dataset  = np.array(mask_dataset,  dtype=np.uint8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "24d4d151",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ─────────────────────────────────────────────\n",
    "# 7. BINARISE & ONE-HOT ENCODE\n",
    "# ─────────────────────────────────────────────\n",
    "mask_dataset[mask_dataset > 0] = 1\n",
    "labels_cat = to_categorical(\n",
    "    mask_dataset,\n",
    "    num_classes=N_CLASSES\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c91b3c88",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ─────────────────────────────────────────────\n",
    "# 8. TRAIN / VAL / TEST SPLIT  (80 / 10 / 10)\n",
    "# ─────────────────────────────────────────────\n",
    "X_train, X_temp, y_train, y_temp = train_test_split(\n",
    "    image_dataset, labels_cat, test_size=0.20, random_state=42)\n",
    "X_val, X_test, y_val, y_test = train_test_split(\n",
    "    X_temp, y_temp, test_size=0.50, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5c3a349c",
   "metadata": {
    "lines_to_next_cell": 1
   },
   "outputs": [],
   "source": [
    "print(f\"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "91a61f0c",
   "metadata": {
    "lines_to_next_cell": 1
   },
   "outputs": [],
   "source": [
    "# ─────────────────────────────────────────────\n",
    "# 9. U-NET MODEL\n",
    "# ─────────────────────────────────────────────\n",
    "def build_unet(n_classes=2, img_h=256, img_w=256, img_c=3):\n",
    "    inputs = Input((img_h, img_w, img_c))\n",
    "\n",
    "    c1 = Conv2D(16,  3, activation='relu', padding='same')(inputs)\n",
    "    c1 = Dropout(0.2)(c1)\n",
    "    c1 = Conv2D(16,  3, activation='relu', padding='same')(c1)\n",
    "    p1 = MaxPooling2D(2)(c1)\n",
    "\n",
    "    c2 = Conv2D(32,  3, activation='relu', padding='same')(p1)\n",
    "    c2 = Dropout(0.2)(c2)\n",
    "    c2 = Conv2D(32,  3, activation='relu', padding='same')(c2)\n",
    "    p2 = MaxPooling2D(2)(c2)\n",
    "\n",
    "    c3 = Conv2D(64,  3, activation='relu', padding='same')(p2)\n",
    "    c3 = Dropout(0.2)(c3)\n",
    "    c3 = Conv2D(64,  3, activation='relu', padding='same')(c3)\n",
    "    p3 = MaxPooling2D(2)(c3)\n",
    "\n",
    "    c4 = Conv2D(128, 3, activation='relu', padding='same')(p3)\n",
    "    c4 = Dropout(0.2)(c4)\n",
    "    c4 = Conv2D(128, 3, activation='relu', padding='same')(c4)\n",
    "    p4 = MaxPooling2D(2)(c4)\n",
    "\n",
    "    c5 = Conv2D(256, 3, activation='relu', padding='same')(p4)\n",
    "    c5 = Dropout(0.3)(c5)\n",
    "    c5 = Conv2D(256, 3, activation='relu', padding='same')(c5)\n",
    "\n",
    "    u6 = Conv2DTranspose(128, 2, strides=2, padding='same')(c5)\n",
    "    u6 = concatenate([u6, c4])\n",
    "    c6 = Conv2D(128, 3, activation='relu', padding='same')(u6)\n",
    "    c6 = Dropout(0.2)(c6)\n",
    "    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)\n",
    "\n",
    "    u7 = Conv2DTranspose(64, 2, strides=2, padding='same')(c6)\n",
    "    u7 = concatenate([u7, c3])\n",
    "    c7 = Conv2D(64,  3, activation='relu', padding='same')(u7)\n",
    "    c7 = Dropout(0.2)(c7)\n",
    "    c7 = Conv2D(64,  3, activation='relu', padding='same')(c7)\n",
    "\n",
    "    u8 = Conv2DTranspose(32, 2, strides=2, padding='same')(c7)\n",
    "    u8 = concatenate([u8, c2])\n",
    "    c8 = Conv2D(32,  3, activation='relu', padding='same')(u8)\n",
    "    c8 = Dropout(0.2)(c8)\n",
    "    c8 = Conv2D(32,  3, activation='relu', padding='same')(c8)\n",
    "\n",
    "    u9 = Conv2DTranspose(16, 2, strides=2, padding='same')(c8)\n",
    "    u9 = concatenate([u9, c1])\n",
    "    c9 = Conv2D(16,  3, activation='relu', padding='same')(u9)\n",
    "    c9 = Dropout(0.2)(c9)\n",
    "    c9 = Conv2D(16,  3, activation='relu', padding='same')(c9)\n",
    "\n",
    "    outputs = Conv2D(n_classes, 1, activation='softmax')(c9)\n",
    "    return Model(inputs, outputs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8dda227a",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = build_unet(n_classes=N_CLASSES)\n",
    "model.compile(\n",
    "    optimizer=optimizers.Adam(learning_rate=LR),\n",
    "    loss='categorical_crossentropy',\n",
    "    metrics=['accuracy']\n",
    ")\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c3ad9835",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ─────────────────────────────────────────────\n",
    "# 10. CALLBACKS\n",
    "# ─────────────────────────────────────────────\n",
    "callbacks = [\n",
    "    EarlyStopping(monitor='val_loss', patience=10,\n",
    "                  restore_best_weights=True, verbose=1),\n",
    "    ModelCheckpoint(\n",
    "        os.path.join(OUT_DIR, \"best_model.keras\"),\n",
    "        monitor='val_loss', save_best_only=True, mode='min', verbose=1\n",
    "    ),\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "21a07027",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ─────────────────────────────────────────────\n",
    "# 11. TRAIN\n",
    "# ─────────────────────────────────────────────\n",
    "history = model.fit(\n",
    "    X_train, y_train,\n",
    "    batch_size=BATCH_SIZE,\n",
    "    epochs=EPOCHS,\n",
    "    validation_data=(X_val, y_val),\n",
    "    callbacks=callbacks,\n",
    "    shuffle=True\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dbfaaed5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ─────────────────────────────────────────────\n",
    "# 12. PREDICT ON TEST SET\n",
    "# ─────────────────────────────────────────────\n",
    "y_pred        = model.predict(X_test, verbose=1)\n",
    "y_pred_argmax = np.argmax(y_pred,  axis=-1)\n",
    "y_test_argmax = np.argmax(y_test,  axis=-1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2b1716b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# flat arrays needed for sklearn metrics\n",
    "y_true_flat  = y_test_argmax.flatten()\n",
    "y_pred_flat  = y_pred_argmax.flatten()\n",
    "y_score_flat = y_pred[:, :, :, 1].flatten()   # foreground softmax probability"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dbc103d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Mean IoU\n",
    "iou_metric = MeanIoU(num_classes=N_CLASSES)\n",
    "iou_metric.update_state(y_test_argmax, y_pred_argmax)\n",
    "mean_iou = iou_metric.result().numpy()\n",
    "print(f\"\\nMean IoU on test set: {mean_iou:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fedd08cf",
   "metadata": {
    "lines_to_next_cell": 2
   },
   "outputs": [],
   "source": [
    "# Save final model\n",
    "model.save(os.path.join(OUT_DIR, \"unet_model_final.keras\"))\n",
    "print(f\"Model saved → {os.path.join(OUT_DIR, 'unet_model_final.keras')}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "28452923",
   "metadata": {
    "lines_to_next_cell": 2
   },
   "outputs": [],
   "source": [
    "# ═════════════════════════════════════════════════════════════════════════════\n",
    "# GRAPH 1 — Train vs Val LOSS\n",
    "# ═════════════════════════════════════════════════════════════════════════════\n",
    "fig, ax = plt.subplots(figsize=(9, 5))\n",
    "epochs_range = range(1, len(history.history['loss']) + 1)\n",
    "ax.plot(epochs_range, history.history['loss'],     color='#378ADD',\n",
    "        linewidth=2, label='Train loss')\n",
    "ax.plot(epochs_range, history.history['val_loss'], color='#D85A30',\n",
    "        linewidth=2, label='Val loss',   linestyle='--')\n",
    "ax.set_title('Train vs validation loss', fontsize=13)\n",
    "ax.set_xlabel('Epoch', fontsize=11)\n",
    "ax.set_ylabel('Loss',  fontsize=11)\n",
    "ax.legend(fontsize=11)\n",
    "ax.grid(alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(OUT_DIR, 'graph1_loss_curves.png'), dpi=150, bbox_inches='tight')\n",
    "plt.close()\n",
    "print(\"Saved → graph1_loss_curves.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "996d5c10",
   "metadata": {
    "lines_to_next_cell": 2
   },
   "outputs": [],
   "source": [
    "# ═════════════════════════════════════════════════════════════════════════════\n",
    "# GRAPH 2 — Train vs Val ACCURACY\n",
    "# ═════════════════════════════════════════════════════════════════════════════\n",
    "fig, ax = plt.subplots(figsize=(9, 5))\n",
    "ax.plot(epochs_range, history.history['accuracy'],     color='#378ADD',\n",
    "        linewidth=2, label='Train accuracy')\n",
    "ax.plot(epochs_range, history.history['val_accuracy'], color='#D85A30',\n",
    "        linewidth=2, label='Val accuracy', linestyle='--')\n",
    "ax.set_title('Train vs validation accuracy', fontsize=13)\n",
    "ax.set_xlabel('Epoch',    fontsize=11)\n",
    "ax.set_ylabel('Accuracy', fontsize=11)\n",
    "ax.legend(fontsize=11)\n",
    "ax.grid(alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(OUT_DIR, 'graph2_accuracy_curves.png'), dpi=150, bbox_inches='tight')\n",
    "plt.close()\n",
    "print(\"Saved → graph2_accuracy_curves.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "92833187",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ═════════════════════════════════════════════════════════════════════════════\n",
    "# GRAPH 3 — Confusion Matrix (counts)\n",
    "# ═════════════════════════════════════════════════════════════════════════════\n",
    "cm = confusion_matrix(y_true_flat, y_pred_flat)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8f739084",
   "metadata": {
    "lines_to_next_cell": 2
   },
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(figsize=(7, 6))\n",
    "im = ax.imshow(cm, interpolation='nearest', cmap='Blues')\n",
    "plt.colorbar(im, ax=ax)\n",
    "ax.set_title('Confusion matrix — pixel counts', fontsize=13)\n",
    "ax.set_xlabel('Predicted label', fontsize=11)\n",
    "ax.set_ylabel('True label',      fontsize=11)\n",
    "ax.set_xticks([0, 1]);  ax.set_xticklabels(CLASS_NAMES, fontsize=11)\n",
    "ax.set_yticks([0, 1]);  ax.set_yticklabels(CLASS_NAMES, fontsize=11)\n",
    "thresh = cm.max() / 2.0\n",
    "for i, j in itertools.product(range(2), range(2)):\n",
    "    ax.text(j, i, f'{cm[i,j]:,}',\n",
    "            ha='center', va='center', fontsize=13,\n",
    "            color='white' if cm[i,j] > thresh else 'black')\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(OUT_DIR, 'graph3_confusion_matrix.png'), dpi=150, bbox_inches='tight')\n",
    "plt.close()\n",
    "print(\"Saved → graph3_confusion_matrix.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "820b17c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ═════════════════════════════════════════════════════════════════════════════\n",
    "# GRAPH 4 — Prediction Heatmaps\n",
    "#   5 columns: Input | Ground truth | Prediction | Prob heatmap | Error map\n",
    "# ═════════════════════════════════════════════════════════════════════════════\n",
    "N_ROWS     = 6\n",
    "col_titles = ['Input image', 'Ground truth', 'Prediction',\n",
    "              'Foreground probability', 'Error map']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "07064570",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(N_ROWS, 5, figsize=(18, N_ROWS * 3.2))\n",
    "for col, t in enumerate(col_titles):\n",
    "    axes[0, col].set_title(t, fontsize=11, fontweight='bold', pad=8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fa24d293",
   "metadata": {},
   "outputs": [],
   "source": [
    "for row in range(N_ROWS):\n",
    "    img      = X_test[row]\n",
    "    gt       = y_test_argmax[row]\n",
    "    pred     = y_pred_argmax[row]\n",
    "    prob_map = y_pred[row, :, :, 1]\n",
    "    err      = (pred != gt).astype(np.uint8)\n",
    "\n",
    "    err_rgb = np.ones((*err.shape, 3), dtype=np.float32) * 0.85\n",
    "    err_rgb[err == 1] = [1.0, 0.2, 0.2]\n",
    "\n",
    "    axes[row, 0].imshow(img);                              axes[row, 0].axis('off')\n",
    "    axes[row, 1].imshow(gt,      cmap='gray', vmin=0, vmax=1); axes[row, 1].axis('off')\n",
    "    axes[row, 2].imshow(pred,    cmap='gray', vmin=0, vmax=1); axes[row, 2].axis('off')\n",
    "    hm = axes[row, 3].imshow(prob_map, cmap='jet', vmin=0, vmax=1)\n",
    "    axes[row, 3].axis('off')\n",
    "    plt.colorbar(hm, ax=axes[row, 3], fraction=0.046, pad=0.04)\n",
    "    axes[row, 4].imshow(err_rgb);                          axes[row, 4].axis('off')\n",
    "    axes[row, 0].set_ylabel(f'Sample {row + 1}', fontsize=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0e63b3d0",
   "metadata": {
    "lines_to_next_cell": 2
   },
   "outputs": [],
   "source": [
    "plt.suptitle('Prediction heatmaps and error maps', fontsize=14, y=1.01)\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(OUT_DIR, 'graph4_heatmaps.png'), dpi=150, bbox_inches='tight')\n",
    "plt.close()\n",
    "print(\"Saved → graph4_heatmaps.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b15e61af",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ═════════════════════════════════════════════════════════════════════════════\n",
    "# GRAPH 5 — ROC Curve (foreground class)\n",
    "# ═════════════════════════════════════════════════════════════════════════════\n",
    "fpr, tpr, _ = roc_curve(y_true_flat, y_score_flat)\n",
    "roc_auc     = auc(fpr, tpr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e97d0f4e",
   "metadata": {
    "lines_to_next_cell": 2
   },
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(figsize=(7, 6))\n",
    "ax.plot(fpr, tpr, color='#378ADD', lw=2,\n",
    "        label=f'ROC curve  (AUC = {roc_auc:.4f})')\n",
    "ax.plot([0, 1], [0, 1], color='#aaa', lw=1,\n",
    "        linestyle='--', label='Random classifier')\n",
    "ax.fill_between(fpr, tpr, alpha=0.08, color='#378ADD')\n",
    "ax.set_xlim([0.0, 1.0]);  ax.set_ylim([0.0, 1.02])\n",
    "ax.set_xlabel('False Positive Rate', fontsize=12)\n",
    "ax.set_ylabel('True Positive Rate',  fontsize=12)\n",
    "ax.set_title('ROC curve — foreground class', fontsize=13)\n",
    "ax.legend(loc='lower right', fontsize=11)\n",
    "ax.grid(alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(OUT_DIR, 'graph5_roc_curve.png'), dpi=150, bbox_inches='tight')\n",
    "plt.close()\n",
    "print(\"Saved → graph5_roc_curve.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8b8522af",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ═════════════════════════════════════════════════════════════════════════════\n",
    "# GRAPH 6 — Final prediction output images\n",
    "#   3 columns: Input | Ground truth | Predicted mask\n",
    "# ═════════════════════════════════════════════════════════════════════════════\n",
    "for i in range(N_DISPLAY):\n",
    "    fig, axes = plt.subplots(1, 3, figsize=(12, 4))\n",
    "\n",
    "    axes[0].imshow(X_test[i])\n",
    "    axes[0].set_title('Input image',      fontsize=12)\n",
    "    axes[0].axis('off')\n",
    "\n",
    "    axes[1].imshow(y_test_argmax[i], cmap='gray')\n",
    "    axes[1].set_title('Ground truth mask', fontsize=12)\n",
    "    axes[1].axis('off')\n",
    "\n",
    "    axes[2].imshow(y_pred_argmax[i], cmap='gray')\n",
    "    axes[2].set_title('Predicted mask',    fontsize=12)\n",
    "    axes[2].axis('off')\n",
    "\n",
    "    plt.suptitle(f'Test sample {i + 1}  —  '\n",
    "                 f'IoU ≈ {mean_iou:.4f}', fontsize=11)\n",
    "    plt.tight_layout()\n",
    "    out_path = os.path.join(OUT_DIR, f'graph6_prediction_{i+1:02d}.png')\n",
    "    plt.savefig(out_path, dpi=150, bbox_inches='tight')\n",
    "    plt.close()\n",
    "    print(f\"Saved → graph6_prediction_{i+1:02d}.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d29beb8d",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f\"\\nAll outputs saved to: {OUT_DIR}\")\n",
    "print(f\"Final Mean IoU: {mean_iou:.4f}\")"
   ]
  }
 ],
 "metadata": {
  "jupytext": {
   "cell_metadata_filter": "-all",
   "main_language": "python",
   "notebook_metadata_filter": "-all"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
