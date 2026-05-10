# ================================================================
# visualize_embeddings.py
# Visualisasi Latent Space embedding wajah via t-SNE + UMAP
#
# ⚠️  PENTING: Visualisasi pakai embedding PER-FOTO (bukan per-orang)
#     supaya cluster terbentuk dengan baik.
#     Database recognition tetap pakai mean aggregation dari
#     extract_embeddings.py — tidak berubah.
#
# Cara pakai:
#   python visualize_embeddings.py
#
# Pastikan croppingfacesraw/ sudah ada (crop_faces.py sudah jalan)!
# ================================================================

import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import openvino as ov
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

try:
    import umap
except ImportError:
    print("📦 Installing umap-learn...")
    import subprocess
    subprocess.run(["pip", "install", "umap-learn"], check=True)
    import umap

from config import (
    CROP_OUTPUT,
    ARCFACE_XML, ARCFACE_BIN,
    CROP_TARGET_SIZE,
)

# ── Warna palette — diperluas untuk banyak orang ────────────────
COLORS = [
    "#E74C3C", "#2ECC71", "#3498DB", "#F39C12", "#9B59B6",
    "#1ABC9C", "#F1C40F", "#E67E22", "#00CED1", "#FF69B4",
    "#8B4513", "#708090", "#DC143C", "#00FA9A", "#1E90FF",
    "#FFD700", "#DA70D6", "#32CD32", "#FF6347", "#40E0D0",
]


# ================================================================
# UTILS
# ================================================================

def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    if face_bgr.shape[:2] != CROP_TARGET_SIZE:
        face_bgr = cv2.resize(face_bgr, CROP_TARGET_SIZE, interpolation=cv2.INTER_AREA)
    face_rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_norm = face_rgb.astype(np.float32) / 255.0
    face_chw  = np.transpose(face_norm, (2, 0, 1))
    return np.expand_dims(face_chw, axis=0)


def get_embedding(face_bgr, compiled_model, output_node) -> np.ndarray:
    tensor    = preprocess_face(face_bgr)
    result    = compiled_model([tensor])[output_node]
    embedding = result.flatten()
    norm      = np.linalg.norm(embedding)
    return embedding / (norm + 1e-6)


def parse_identity(filename: str) -> str:
    """NamaOrang_001.jpg → 'NamaOrang'"""
    stem  = Path(filename).stem
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else stem


def shorten_label(name: str, max_len: int = 12) -> str:
    """Potong nama panjang agar label tidak tumpang tindih."""
    return name if len(name) <= max_len else name[:max_len] + "…"


# ================================================================
# LOAD ARCFACE
# ================================================================

def load_arcface():
    if not ARCFACE_XML.exists():
        raise FileNotFoundError(f"❌ Tidak ditemukan: {ARCFACE_XML}")
    if not ARCFACE_BIN.exists():
        raise FileNotFoundError(f"❌ Tidak ditemukan: {ARCFACE_BIN}")

    core           = ov.Core()
    device         = "GPU" if "GPU" in core.available_devices else "CPU"
    ov_model       = core.read_model(model=str(ARCFACE_XML), weights=str(ARCFACE_BIN))
    compiled_model = core.compile_model(model=ov_model, device_name=device)
    output_node    = compiled_model.output(0)

    print(f"✅ ArcFace loaded — device: {device}")
    return compiled_model, output_node


# ================================================================
# EKSTRAK EMBEDDING PER-FOTO UNTUK VISUALISASI
# ================================================================

def collect_per_photo_embeddings():
    """
    Baca semua foto di croppingfacesraw/ dan ekstrak embedding
    satu per satu — bukan di-aggregate.
    Ini yang dipakai untuk visualisasi supaya cluster terbentuk.
    """
    if not CROP_OUTPUT.exists():
        raise FileNotFoundError(
            f"❌ Folder tidak ditemukan: {CROP_OUTPUT}\n"
            f"   Jalankan crop_faces.py terlebih dahulu!"
        )

    all_images = sorted(CROP_OUTPUT.glob("*.jpg"))
    if not all_images:
        raise RuntimeError(f"❌ Tidak ada .jpg di {CROP_OUTPUT}")

    print(f"\n📂 Source : {CROP_OUTPUT.resolve()}")
    print(f"🖼️  Total  : {len(all_images)} foto\n")

    compiled_arcface, output_node = load_arcface()

    embeddings = []
    names      = []

    for img_path in tqdm(all_images, desc="Extracting per-photo embeddings"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        try:
            emb = get_embedding(img, compiled_arcface, output_node)
            if emb.shape == (512,):
                embeddings.append(emb)
                names.append(parse_identity(img_path.name))
        except Exception:
            continue

    return np.array(embeddings), np.array(names)


# ================================================================
# VISUALISASI
# ================================================================

def visualize(embeddings: np.ndarray, names: np.ndarray,
              save_path: str = "latent_space.png"):

    N, D           = embeddings.shape
    unique_persons = sorted(set(names))
    n_persons      = len(unique_persons)

    print("\n" + "=" * 70)
    print(f"{'LATENT SPACE VISUALIZATION':^70}")
    print("=" * 70)
    print(f"   Total foto    : {N}")
    print(f"   Dimensi       : {D}-D → 2-D")
    print(f"   Total orang   : {n_persons}")
    print("=" * 70 + "\n")

    le         = LabelEncoder()
    labels_int = le.fit_transform(names)

    patches = [
        mpatches.Patch(
            color=COLORS[i % len(COLORS)],
            label=shorten_label(name),
        )
        for i, name in enumerate(le.classes_)
    ]

    # ── t-SNE ───────────────────────────────────────────────────
    print("⏳ Menjalankan t-SNE...")
    # Perplexity harus < jumlah sampel, idealnya 5–50
    perp     = min(30, max(5, N // max(n_persons, 1)))
    tsne     = TSNE(
        n_components=2,
        perplexity=perp,
        max_iter=1000,
        random_state=42,
        init="pca",
    )
    emb_tsne = tsne.fit_transform(embeddings)
    print(f"   ✅ t-SNE selesai  (perplexity={perp})")

    # ── UMAP ────────────────────────────────────────────────────
    print("⏳ Menjalankan UMAP...")
    # n_neighbors: lebih besar → lebih global, minimal 2
    n_neigh  = min(15, max(2, N // max(n_persons, 1)))
    reducer  = umap.UMAP(
        n_components=2,
        n_neighbors=n_neigh,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    emb_umap = reducer.fit_transform(embeddings)
    print(f"   ✅ UMAP selesai   (n_neighbors={n_neigh})\n")

    # ── Plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor("#0F1117")

    titles  = ["t-SNE  (Local Connectivity)", "UMAP  (Global Structure & Cosine)"]
    data_2d = [emb_tsne, emb_umap]

    for ax, title, emb2d in zip(axes, titles, data_2d):
        ax.set_facecolor("#1A1D27")

        for i, person in enumerate(le.classes_):
            mask  = labels_int == i
            color = COLORS[i % len(COLORS)]

            # Scatter semua titik orang ini
            ax.scatter(
                emb2d[mask, 0], emb2d[mask, 1],
                c=color, s=80, alpha=0.75,
                edgecolors="white", linewidths=0.4,
                label=shorten_label(person),
            )

            # Label singkat di centroid cluster
            center = emb2d[mask].mean(axis=0)
            ax.annotate(
                shorten_label(person, max_len=10),
                center,
                fontsize=7, fontweight="bold",
                color="white", ha="center", va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor=color,
                    alpha=0.7,
                    edgecolor="none",
                ),
            )

        ax.set_title(title, color="white", fontsize=13, pad=12)
        ax.tick_params(colors="gray")
        ax.grid(True, color="#2A2D3A", linestyle="--", alpha=0.4)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

        # Legend di luar plot agar tidak overlap
        ax.legend(
            handles=patches,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            fontsize=7,
            facecolor="#1A1D27",
            labelcolor="white",
            edgecolor="#555",
            title="Identitas",
            title_fontsize=8,
        )

    fig.suptitle(
        f"Latent Space ArcFace  |  {N} foto  |  {n_persons} orang  |  {D}-D → 2-D",
        color="white", fontsize=14, y=1.01,
    )
    plt.tight_layout()

    out_path = Path(save_path)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"💾 Plot disimpan : {out_path.resolve()}")
    plt.show()
    plt.close()

    # ── Silhouette Score ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'SILHOUETTE SCORE':^70}")
    print("=" * 70)
    for method_name, emb2d in [("t-SNE", emb_tsne), ("UMAP", emb_umap)]:
        if len(set(labels_int)) < 2:
            print(f"   {method_name}: ⚠️  Butuh minimal 2 orang.")
            continue
        score  = silhouette_score(emb2d, labels_int)
        status = (
            "🌟 EXCELLENT" if score > 0.6 else
            "✅ GOOD"      if score > 0.3 else
            "⚠️  OVERLAP"
        )
        print(f"   {method_name:<8}: {score:.4f}  {status}")
    print("\n   💡 Mendekati 1.0 = cluster antar orang terpisah dengan jelas")
    print("=" * 70)


# ================================================================
if __name__ == "__main__":
    embeddings, names = collect_per_photo_embeddings()
    visualize(embeddings, names, save_path="latent_space.png")
