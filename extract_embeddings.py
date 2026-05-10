# ================================================================
# extract_embeddings.py
# Ekstrak embedding wajah dari croppingfacesraw/
# → 1 embedding per foto (individual) → disimpan sebagai .pkl
#
# Cara pakai:
#   python extract_embeddings.py
#
# Pastikan crop_faces.py sudah dijalankan lebih dulu!
# ================================================================

import cv2
import pickle
import time
import numpy as np
import openvino as ov
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from config import (
    CROP_OUTPUT,
    ARCFACE_XML, ARCFACE_BIN,
    CROP_TARGET_SIZE,
    EMBEDDING_OUTPUT, EMBEDDING_FILE,
)


# ================================================================
# PREPROCESSING UTILS
# ================================================================

def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Meningkatkan kontras wajah di ruang LAB agar fitur lebih jelas
    bagi ArcFace, terutama pada kondisi pencahayaan tidak merata.

    - Bekerja pada channel L (lightness) saja → warna tidak berubah
    - clipLimit=2.0  : batas amplifikasi kontras, cukup kuat tanpa noise berlebih
    - tileGridSize   : ukuran grid patch lokal (standar 8×8)
    """
    lab        = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b    = cv2.split(lab)
    clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_merged = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)


def align_face(img_bgr: np.ndarray,
               left_eye: np.ndarray,
               right_eye: np.ndarray,
               target_size: tuple = (112, 112)) -> np.ndarray | None:
    """
    Face Alignment berdasarkan koordinat kedua mata (landmarks).
    Meluruskan kemiringan wajah secara geometris sebelum dikirim ke ArcFace,
    sehingga embedding lebih stabil dan akurat.

    Langkah:
      1. Hitung sudut kemiringan dari garis antar mata
      2. Rotasi gambar agar kedua mata sejajar horizontal
      3. Crop area wajah setelah rotasi, lalu resize ke target_size

    Return None jika crop kosong (wajah terlalu dekat ke tepi gambar).
    """
    dY    = float(right_eye[1] - left_eye[1])
    dX    = float(right_eye[0] - left_eye[0])
    angle = np.degrees(np.arctan2(dY, dX))

    eye_center = (
        int((left_eye[0] + right_eye[0]) / 2),
        int((left_eye[1] + right_eye[1]) / 2),
    )

    M       = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
    h, w    = img_bgr.shape[:2]
    rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC)

    dist  = np.sqrt(dX ** 2 + dY ** 2)
    scale = 2.2                                         # faktor zoom area wajah
    x1    = int(eye_center[0] - dist * scale)
    y1    = int(eye_center[1] - dist * scale)
    x2    = int(eye_center[0] + dist * scale)
    y2    = int(eye_center[1] + dist * scale)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face_crop = rotated[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None

    return cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)


def enhance_face(img_bgr: np.ndarray,
                 target_size: tuple = (112, 112)) -> np.ndarray:
    """
    Pipeline preprocessing lengkap sebelum embedding:
      1. Resize ke target_size (jika belum 112×112)
      2. CLAHE — normalisasi pencahayaan adaptif

    Catatan: align_face() dipanggil terpisah (butuh keypoints landmarks).
    Fungsi ini dipakai sebagai fallback ketika landmarks tidak tersedia,
    atau sebagai tahap akhir setelah alignment.
    """
    if img_bgr.shape[:2] != target_size:
        img_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)

    img_bgr = apply_clahe(img_bgr)
    return img_bgr


# ================================================================
# ARCFACE UTILS
# ================================================================

def preprocess_face_arcface(face_bgr: np.ndarray) -> np.ndarray:
    """
    BGR face (112×112) → float32 tensor [1, 3, 112, 112] ternormalisasi 0–1.
    Sesuai format input ArcFace / Buffalo_SC OpenVINO.
    """
    if face_bgr.shape[:2] != CROP_TARGET_SIZE:
        face_bgr = cv2.resize(face_bgr, CROP_TARGET_SIZE, interpolation=cv2.INTER_AREA)

    face_rgb  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_norm = face_rgb.astype(np.float32) / 255.0
    face_chw  = np.transpose(face_norm, (2, 0, 1))
    return np.expand_dims(face_chw, axis=0)             # (1, 3, 112, 112)


def get_embedding(face_bgr: np.ndarray, compiled_model, output_node) -> np.ndarray:
    """
    Ekstrak 512-D embedding dari wajah BGR.
    Preprocessing yang diterapkan sebelum inferensi:
      - CLAHE (enhance_face)
      - Resize + normalisasi channel (preprocess_face_arcface)

    Return: numpy array shape (512,) yang sudah L2-normalize.
    """
    # ── Terapkan CLAHE sebelum dikirim ke ArcFace ───────────────
    face_enhanced = enhance_face(face_bgr, target_size=CROP_TARGET_SIZE)

    tensor    = preprocess_face_arcface(face_enhanced)
    result    = compiled_model([tensor])[output_node]
    embedding = result.flatten()
    norm      = np.linalg.norm(embedding)
    return embedding / (norm + 1e-6)                    # L2 normalize


def parse_identity_from_filename(filename: str) -> str:
    """
    Ambil nama identitas dari nama file format: NIM_NAMA_NOMOR.jpg
    → 'NAMA'
    Misal: 1101223224_Rheira Nisrina Abiyah_030.jpg → 'Rheira Nisrina Abiyah'
    """
    stem  = Path(filename).stem                         # '1101223224_Rheira Nisrina Abiyah_030'
    parts = stem.split("_")                            # ['1101223224', 'Rheira', 'Nisrina', 'Abiyah', '030']
    
    # parts[0] = NIM (angka)
    # parts[-1] = NOMOR (angka)
    # parts[1:-1] = NAMA (bisa lebih dari 1 kata)
    if len(parts) >= 3:
        nama = "_".join(parts[1:-1])
        return nama
    else:
        return stem


# ================================================================
# LOAD MODEL
# ================================================================

def load_arcface_openvino(xml_path: Path, bin_path: Path):
    """Load ArcFace dari .xml + .bin via OpenVINO Core."""
    print("\n" + "=" * 70)
    print(f"{'LOADING ARCFACE OPENVINO':^70}")
    print("=" * 70)

    if not xml_path.exists():
        raise FileNotFoundError(f"❌ Tidak ditemukan: {xml_path}")
    if not bin_path.exists():
        raise FileNotFoundError(f"❌ Tidak ditemukan: {bin_path}")

    print(f"   .xml  : {xml_path}")
    print(f"   .bin  : {bin_path}")

    core   = ov.Core()
    device = "GPU" if "GPU" in core.available_devices else "CPU"
    print(f"   Device: {device}")

    ov_model       = core.read_model(model=str(xml_path), weights=str(bin_path))
    compiled_model = core.compile_model(model=ov_model, device_name=device)

    input_layer = compiled_model.input(0)
    output_node = compiled_model.output(0)

    print(f"   Input shape  : {input_layer.partial_shape}")
    print(f"   Output shape : {output_node.partial_shape}")

    # Quick sanity check
    dummy    = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    test_emb = get_embedding(dummy, compiled_model, output_node)
    print(f"   Sanity check : shape={test_emb.shape}, norm={np.linalg.norm(test_emb):.4f} (harus ~1.0)")
    print("✅ ArcFace OpenVINO loaded!\n")

    return compiled_model, output_node


# ================================================================
# MAIN EMBEDDING PIPELINE
# ================================================================

def run_extraction():
    # ── Validasi folder crop ────────────────────────────────────
    if not CROP_OUTPUT.exists():
        raise FileNotFoundError(
            f"❌ Folder crop tidak ditemukan: {CROP_OUTPUT}\n"
            f"   Jalankan crop_faces.py terlebih dahulu!"
        )

    all_images = sorted(CROP_OUTPUT.glob("*.jpg"))
    if not all_images:
        raise RuntimeError(f"❌ Tidak ada file .jpg di {CROP_OUTPUT}.")

    # ── Load model ──────────────────────────────────────────────
    compiled_arcface, output_node = load_arcface_openvino(ARCFACE_XML, ARCFACE_BIN)

    # ── Kelompokkan file per identitas ──────────────────────────
    identity_map: dict[str, list[Path]] = {}
    for img_path in all_images:
        identity = parse_identity_from_filename(img_path.name)
        identity_map.setdefault(identity, []).append(img_path)

    print("=" * 70)
    print(f"{'EMBEDDING EXTRACTION PIPELINE':^70}")
    print("=" * 70)
    print(f"📂 Source folder  : {CROP_OUTPUT.resolve()}")
    print(f"🖼️  Total images   : {len(all_images)}")
    print(f"👥 Total persons  : {len(identity_map)}")
    print(f"⚙️  Strategy       : CLAHE → Individual Embeddings → L2 Normalize")
    print("=" * 70 + "\n")

    # ── Database ────────────────────────────────────────────────
    face_database = {
        "embeddings": [],      # list of np.ndarray (512,) — 1 per foto
        "names"     : [],      # list of str — nama per foto (bisa duplicate)
        "image_paths": [],     # list of str — path per foto
    }

    start_time = time.time()

    for identity, img_paths in tqdm(
        sorted(identity_map.items()),
        desc="Extracting",
        unit="person",
    ):
        valid_count = 0

        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                tqdm.write(f"   ⚠️  Gagal baca: {img_path.name}")
                continue

            try:
                # get_embedding() sudah include CLAHE di dalamnya
                emb = get_embedding(img, compiled_arcface, output_node)
                if emb.shape == (512,):
                    # Store individual embedding + path + name (per photo, not aggregated)
                    face_database["embeddings"].append(emb)
                    face_database["names"].append(identity)
                    face_database["image_paths"].append(str(img_path))
                    valid_count += 1
            except Exception as e:
                tqdm.write(f"   ⚠️  Error pada {img_path.name}: {e}")
                continue

        if valid_count == 0:
            tqdm.write(f"   ❌ {identity}: tidak ada embedding valid, dilewati.")
            continue

        tqdm.write(
            f"   ✅ {identity:<35} : {valid_count} photos extracted"
        )

    if not face_database["embeddings"]:
        raise RuntimeError("❌ Tidak ada embedding yang berhasil diekstrak!")

    # Convert ke numpy array untuk similarity search yang cepat
    face_database["embeddings"] = np.array(face_database["embeddings"])

    duration = time.time() - start_time

    # ── Simpan database ─────────────────────────────────────────
    EMBEDDING_OUTPUT.mkdir(parents=True, exist_ok=True)
    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(face_database, f)

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'EXTRACTION COMPLETE':^70}")
    print("=" * 70)
    print(f"� Total photos     : {len(face_database['names'])}")
    print(f"📐 Embedding matrix : {face_database['embeddings'].shape}")
    print(f"⏱️  Processing time  : {duration:.2f} seconds")
    print(f"💾 Saved to         : {EMBEDDING_FILE.resolve()}")
    print("-" * 70)
    print(f"{'IDENTITY':<40} | {'PATH':^25} | {'NORM':>6}")
    print("-" * 70)
    for name, path, emb in zip(
        face_database["names"],
        face_database["image_paths"],
        face_database["embeddings"],
    ):
        path_display = Path(path).name[:25] if path else "N/A"
        print(f"{name:<40} | {path_display:^25} | {np.linalg.norm(emb):>6.4f}")
    print("=" * 70)

    return face_database


# ================================================================
# QUICK RECOGNITION TEST (opsional, jalankan setelah ekstraksi)
# ================================================================

def test_recognition(face_database: dict, threshold=0.45):
    """
    Test sederhana: ambil 1 foto acak per orang dari croppingfacesraw,
    coba kenali, laporkan akurasi.
    CLAHE otomatis diterapkan via get_embedding().
    """
    import random

    print("\n" + "=" * 70)
    print(f"{'RECOGNITION SELF-TEST':^70}")
    print("=" * 70)

    compiled_arcface, output_node = load_arcface_openvino(ARCFACE_XML, ARCFACE_BIN)

    db_embeddings = face_database["embeddings"]
    db_names      = face_database["names"]

    identity_map: dict[str, list[Path]] = {}
    for img_path in sorted(CROP_OUTPUT.glob("*.jpg")):
        identity = parse_identity_from_filename(img_path.name)
        identity_map.setdefault(identity, []).append(img_path)

    total   = 0
    correct = 0

    for identity, img_paths in sorted(identity_map.items()):
        sample_path = random.choice(img_paths)
        img         = cv2.imread(str(sample_path))
        if img is None:
            continue

        # CLAHE sudah diterapkan di dalam get_embedding()
        emb          = get_embedding(img, compiled_arcface, output_node)
        similarities = np.dot(db_embeddings, emb)
        best_idx     = int(np.argmax(similarities))
        best_sim     = float(similarities[best_idx])
        predicted    = db_names[best_idx] if best_sim >= threshold else "Unknown"

        is_correct = predicted.lower() == identity.lower()
        status     = "✅" if is_correct else "❌"

        total   += 1
        correct += int(is_correct)

        print(f"{status} {identity:<30} → {predicted:<30} (sim={best_sim:.3f})")

    accuracy = (correct / total * 100) if total > 0 else 0
    print("-" * 70)
    print(f"Accuracy : {accuracy:.1f}%  ({correct}/{total} benar)")
    print("=" * 70)


# ================================================================
if __name__ == "__main__":
    db = run_extraction()

    # Uncomment baris di bawah untuk langsung test setelah ekstraksi:
    # test_recognition(db, threshold=0.45)