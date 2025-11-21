# embedding_index.py
"""
Embedding & FAISS Index Builder for CLAP Audio Retrieval
Saves:
 - output/faiss_index.bin
 - output/metadata.pkl
 - output/confusion_matrix.png
 - output/classification_report.json
Exports functions used by the Streamlit UI.
"""

import os
import io
import json
import pickle
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import faiss
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import ClapProcessor, ClapModel
from sklearn.metrics import confusion_matrix, classification_report

# ---------- CONFIG ----------
CSV_PATH = "metadata_simple.csv"
AUDIO_FOLDER = "Data"
OUTPUT_FOLDER = "output"
MODEL_ID = "laion/clap-htsat-unfused"
TARGET_SR = 48000
CLIP_SECONDS = 10
FAISS_PATH = os.path.join(OUTPUT_FOLDER, "faiss_index.bin")
META_PATH = os.path.join(OUTPUT_FOLDER, "metadata.pkl")
CM_PATH = os.path.join(OUTPUT_FOLDER, "confusion_matrix.png")
REPORT_PATH = os.path.join(OUTPUT_FOLDER, "classification_report.json")
# ----------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# model cache globals
_PROCESSOR = None
_MODEL = None
_DEVICE = None


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clap(model_id: str = MODEL_ID):
    """
    Load and cache CLAP processor & model.
    Returns (processor, model, device).
    """
    global _PROCESSOR, _MODEL, _DEVICE
    if _PROCESSOR is not None and _MODEL is not None:
        return _PROCESSOR, _MODEL, _DEVICE

    _DEVICE = get_device()
    print(f"[CLAP] loading {model_id} on {_DEVICE} ...")
    _PROCESSOR = ClapProcessor.from_pretrained(model_id)
    _MODEL = ClapModel.from_pretrained(model_id)
    _MODEL.to(_DEVICE)
    _MODEL.eval()
    return _PROCESSOR, _MODEL, _DEVICE


def _read_audio_any(source, sr: int = TARGET_SR, duration: int = CLIP_SECONDS) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Load audio from a path or a file-like (bytes) object.
    Returns (y, sr) or (None, None) on failure.
    """
    # If source is a path string
    try:
        if isinstance(source, str):
            y, sr_ret = librosa.load(source, sr=sr, mono=True, duration=duration)
        else:
            # file-like object (e.g. Streamlit upload)
            # read bytes and use soundfile to decode
            bio = io.BytesIO(source.read() if hasattr(source, "read") else source)
            y_sf, sr_ret = sf.read(bio, dtype="float32")
            if y_sf is None:
                return None, None
            if y_sf.ndim > 1:
                y = np.mean(y_sf, axis=1)
            else:
                y = y_sf
    except Exception:
        # fallback: try soundfile on path/file
        try:
            if isinstance(source, str):
                y_sf, sr_ret = sf.read(source, dtype="float32")
            else:
                bio = io.BytesIO(source.read() if hasattr(source, "read") else source)
                y_sf, sr_ret = sf.read(bio, dtype="float32")
            if y_sf is None:
                return None, None
            if y_sf.ndim > 1:
                y = np.mean(y_sf, axis=1)
            else:
                y = y_sf
        except Exception:
            return None, None

    if y is None or y.size == 0:
        return None, None

    # pad/trim
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    return y.astype(np.float32), sr


def safe_load_audio(path_or_file) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Public wrapper for loading and making sure we return mono float32 and TARGET_SR samples.
    Accepts file path (str) or file-like (e.g. streamlit uploaded file).
    """
    return _read_audio_any(path_or_file, sr=TARGET_SR, duration=CLIP_SECONDS)


def encode_audio_batch(filepaths: List[str], processor: ClapProcessor, model: ClapModel, device: str, batch_size: int = 8):
    """
    Given list of file paths, compute CLAP audio embeddings (normalized).
    Returns (embeddings_array, valid_indices).
    """
    embeddings = []
    valid_idx = []
    n = len(filepaths)
    i = 0
    while i < n:
        batch_paths = filepaths[i: i + batch_size]
        audios = []
        idx_map = []
        for j, p in enumerate(batch_paths):
            y, sr = safe_load_audio(p)
            if y is None:
                continue
            audios.append(y)
            idx_map.append(i + j)
        if len(audios) == 0:
            i += batch_size
            continue
        inputs = processor(audios=audios, sampling_rate=TARGET_SR, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_audio_features(**inputs)  # (B, D)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1).cpu().numpy()
        for k, emb in enumerate(feats):
            embeddings.append(emb)
            valid_idx.append(idx_map[k])
        i += batch_size

    if len(embeddings) == 0:
        return None, []
    return np.vstack(embeddings).astype(np.float32), valid_idx


def build_faiss_index(embeddings: np.ndarray, use_inner_product: bool = True):
    """
    Build and save FAISS index. We use IndexFlatIP and assume all embeddings are normalized.
    """
    d = embeddings.shape[1]
    if use_inner_product:
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, FAISS_PATH)
    print("[FAISS] saved to", FAISS_PATH)
    return index


def build_index(csv_path: str = CSV_PATH, audio_folder: str = AUDIO_FOLDER, processor=None, model=None, device=None):
    """
    Build audio embeddings from csv & audio folder, save metadata and FAISS index.
    Returns (index, metadata_df, audio_embeddings).
    """
    processor, model, device = (processor, model, device) if (processor and model and device) else load_clap()

    df = pd.read_csv(csv_path)
    if "filename" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'filename' and 'label' columns")

    filepaths = [os.path.join(audio_folder, fn) for fn in df["filename"].tolist()]

    print("[Index] encoding audio embeddings...")
    audio_embs, valid_idx = encode_audio_batch(filepaths, processor, model, device)
    if audio_embs is None or len(valid_idx) == 0:
        raise RuntimeError("No audio embeddings produced")

    df_valid = df.iloc[valid_idx].reset_index(drop=True)

    # save metadata and audio embeddings to disk
    df_valid.to_pickle(META_PATH)
    np.save(os.path.join(OUTPUT_FOLDER, "audio_embeddings.npy"), audio_embs)

    # build faiss index
    index = build_faiss_index(audio_embs)
    return index, df_valid, audio_embs


def load_index() -> Tuple[Optional[faiss.Index], Optional[pd.DataFrame]]:
    """
    Load FAISS index & metadata from disk (if present).
    """
    if not os.path.exists(FAISS_PATH) or not os.path.exists(META_PATH):
        return None, None
    index = faiss.read_index(FAISS_PATH)
    df = pd.read_pickle(META_PATH)
    return index, df


def search_index(index: faiss.Index, query_emb: np.ndarray, top_k: int = 5):
    """
    Query FAISS index with a single vector (1D). Returns (ids, scores).
    """
    q = np.array(query_emb).astype(np.float32).reshape(1, -1)
    D, I = index.search(q, top_k)
    return I[0], D[0]


# ---------------- EVALUATION: confusion matrix & report ----------------
def evaluate_and_save(text_embs: np.ndarray, audio_embs: np.ndarray, labels: List[str]):
    """
    text_embs: (num_classes, dim) or (N_text, dim)
    audio_embs: (N_audio, dim)
    labels: array-like of true labels for each audio row (length == N_audio)
    This function matches each text embedding to the audio entry (nearest).
    Saves confusion matrix PNG and classification report JSON in output folder.
    Returns (cm, y_true, y_pred, report_dict)
    """
    # ensure numpy arrays
    text_embs = np.asarray(text_embs)
    audio_embs = np.asarray(audio_embs)
    labels = np.asarray(labels)

    # compute similarity (inner product) — both emb should be normalized already
    sims = text_embs @ audio_embs.T  # shape (N_text, N_audio) or (N_query, N_audio)

    # For each text row pick best audio; map to predicted label
    pred_indices = np.argmax(sims, axis=1)
    pred_labels = labels[pred_indices]

    # If text_embs correspond to class prototypes (one per class), y_true should be classes mapped
    # But simpler: if text_embs correspond to class labels (one per class), we can make pairs:
    y_true = labels  # actual labels for each audio item (length N_audio)
    # We need predicted label for each audio item. We'll invert mapping: for each audio, find nearest text (argmax over text)
    nearest_text_for_audio = np.argmax((audio_embs @ text_embs.T), axis=1)
    # nearest_text_for_audio gives index of closest text prototype for each audio sample
    # Build predicted labels array for audio items:
    # we need the mapping from text_emb index -> text label. We expect the caller to pass text_labels array.
    # To keep general, caller should use evaluate_and_save_by_class_prototypes(...) below.
    raise RuntimeError("Use evaluate_and_save_by_class_prototypes(text_labels, text_embs, audio_embs, labels) instead.")


def evaluate_and_save_by_class_prototypes(text_labels: List[str], text_embs: np.ndarray, audio_embs: np.ndarray, true_labels: List[str]):
    """
    Evaluate when text_embs are prototypes for classes (text_labels list).
    text_labels: e.g. ['drum','keys']
    text_embs: (num_classes, dim)
    audio_embs: (N_audio, dim)
    true_labels: length N_audio (strings)
    Saves confusion matrix (png) and classification report (json).
    """
    # normalize to numpy
    text_embs = np.asarray(text_embs)
    audio_embs = np.asarray(audio_embs)
    true_labels = np.asarray([str(x) for x in true_labels])

    # compute sims: for each audio compute similarity to each text prototype
    sims = audio_embs @ text_embs.T  # shape (N_audio, num_classes)
    pred_idx = np.argmax(sims, axis=1)
    pred_labels = np.array([str(text_labels[i]) for i in pred_idx])

    # build confusion matrix (need label ordering)
    classes = sorted(list(set(true_labels) | set(text_labels)))
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)

    # save confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=200)
    plt.close()
    print("[EVAL] Confusion matrix saved to:", CM_PATH)

    # classification report (as dict and save JSON)
    report = classification_report(true_labels, pred_labels, labels=classes, output_dict=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print("[EVAL] Classification report saved to:", REPORT_PATH)

    return cm, true_labels, pred_labels, report, CM_PATH, REPORT_PATH


# ----------------------- CLI entry -----------------------
if __name__ == "__main__":
    # Build index and evaluate using class prototypes (built from unique labels)
    print("=== CLAP + FAISS pipeline starting ===")
    proc, mod, dev = load_clap()

    index, meta_df, audio_embs = build_index(CSV_PATH, AUDIO_FOLDER, proc, mod, dev)
    print("[MAIN] Index built; metadata rows:", len(meta_df))

    # Build text prototypes: one text prompt per unique label
    # convert labels to strings
    unique_labels = sorted(list(meta_df["label"].astype(str).unique()))
    print("[MAIN] Unique labels:", unique_labels)

    # create simple prompt / description per label (customize if needed)
    text_prompts = []
    for lbl in unique_labels:
        s = lbl
        # simple presets
        if lbl.lower() in ("0", "drum", "drums", "percussion"):
            s = "drum percussion drum loop"
        elif lbl.lower() in ("1", "key", "keys", "keyboard", "piano", "synth"):
            s = "keyboard keys piano synth melody"
        # else keep label string
        text_prompts.append(s)

    # compute text embeddings for prototypes
    # process in batches via processor/model
    print("[MAIN] computing text prototype embeddings...")
    inputs = proc(text=text_prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        t_feats = mod.get_text_features(**inputs)
        t_feats = torch.nn.functional.normalize(t_feats, p=2, dim=1).cpu().numpy()

    # evaluate
    cm, y_true, y_pred, report, cm_path, report_path = evaluate_and_save_by_class_prototypes(
        text_labels=unique_labels,
        text_embs=t_feats,
        audio_embs=audio_embs,
        true_labels=meta_df["label"].astype(str).tolist()
    )

    print("=== Done ===")
