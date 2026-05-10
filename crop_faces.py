## ================================================================
# config.py — Semua path & parameter di satu tempat
# Sesuaikan bagian PATH sesuai lokasi file kamu
# ================================================================

from pathlib import Path

# ──────────────────────────────────────────────────────────────
# PATH — Sesuaikan ini
# ──────────────────────────────────────────────────────────────

# Folder dataset input: berisi subfolder per orang
# Contoh struktur:
#   dataset/knownface/
#       Budi/
#           foto1.jpg
#           foto2.jpg
#       Siti/
#           foto1.jpg
DATASET_SOURCE = Path("C:\\Users\\syawal\\Pictures\\dataset\\knownface")

# Model YOLO OpenVINO (.xml dan .bin harus satu folder)
YOLO_XML = Path("C:\\Users\\syawal\\Downloads\\Database TA\\yolov11n-face_openvino_model\\yolov11n-face.xml")
YOLO_BIN = Path("C:\\Users\\syawal\\Downloads\\Database TA\\yolov11n-face_openvino_model\\yolov11n-face.bin")

# Model ArcFace / Buffalo_SC OpenVINO (.xml dan .bin harus satu folder)
ARCFACE_XML = Path("C:\\Users\\syawal\\Downloads\\Database TA\\buffalo_sc_rec.xml")
ARCFACE_BIN = Path("C:\\Users\\syawal\\Downloads\\Database TA\\buffalo_sc_rec.bin")

# Output folder hasil crop (flat, semua .jpg dalam 1 folder)
CROP_OUTPUT = Path("croppingfacesraw")

# Output folder hasil embedding
EMBEDDING_OUTPUT = Path("embeddings")
EMBEDDING_FILE   = EMBEDDING_OUTPUT / "face_database.pkl"

# ──────────────────────────────────────────────────────────────
# PARAMETER DETEKSI & CROPPING
# ──────────────────────────────────────────────────────────────
YOLO_INPUT_SIZE      = (640, 640)   # input standar YOLOv11
YOLO_CONF_THRESHOLD  = 0.60
YOLO_NMS_THRESHOLD   = 0.45

CROP_TARGET_SIZE     = (112, 112)   # standar ArcFace
CROP_PADDING_RATIO   = 0.15

# ──────────────────────────────────────────────────────────────
# PARAMETER EMBEDDING & RECOGNITION
# ──────────────────────────────────────────────────────────────
RECOGNITION_THRESHOLD = 0.45


# ================================================================
# crop_faces.py
# Deteksi & crop wajah dari dataset → disimpan di croppingfacesraw/
# Menggunakan YOLO OpenVINO (.xml + .bin) — tanpa Colab, tanpa GPU cloud
#
# Cara pakai:
#   python crop_faces.py
# ================================================================

import cv2
import shutil
import time
import numpy as np
import openvino as ov
from pathlib import Path
from tqdm import tqdm

from config import (
    DATASET_SOURCE,
    YOLO_XML, YOLO_BIN,
    YOLO_INPUT_SIZE, YOLO_CONF_THRESHOLD, YOLO_NMS_THRESHOLD,
    CROP_OUTPUT, CROP_TARGET_SIZE, CROP_PADDING_RATIO,
)


# ================================================================
# UTILS
# ================================================================

def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Normalisasi pencahayaan wajah agar fitur lebih jelas untuk ArcFace."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def preprocess_yolo(img: np.ndarray, input_size=(640, 640)) -> np.ndarray:
    """
    BGR image → float32 tensor [1, 3, H, W] ternormalisasi 0–1.
    Sesuai format input YOLOv11 OpenVINO.
    """
    img_rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size)
    img_norm    = img_resized.astype(np.float32) / 255.0
    img_chw     = np.transpose(img_norm, (2, 0, 1))
    return np.expand_dims(img_chw, axis=0)          # (1, 3, 640, 640)


def postprocess_yolo(
    output: np.ndarray,
    orig_shape: tuple,
    input_size=(640, 640),
    conf_thresh=0.60,
    nms_thresh=0.45,
) -> list:
    """
    Parse output YOLOv11 → list bounding box [x1, y1, x2, y2].

    Output YOLOv11 shape: (1, 5, 8400)
    Format tiap anchor  : [cx, cy, w, h, conf]
    """
    pred = output[0].T          # (8400, 5)

    h_orig, w_orig = orig_shape[:2]
    sx = w_orig / input_size[0]
    sy = h_orig / input_size[1]

    boxes_xywh = []
    scores     = []

    for det in pred:
        cx, cy, w, h, conf = det
        if conf < conf_thresh:
            continue
        x1 = int((cx - w / 2) * sx)
        y1 = int((cy - h / 2) * sy)
        bw = int(w * sx)
        bh = int(h * sy)
        boxes_xywh.append([x1, y1, bw, bh])
        scores.append(float(conf))

    if not boxes_xywh:
        return []

    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, conf_thresh, nms_thresh)
    result  = []
    for i in indices:
        x, y, w, h = boxes_xywh[i]
        result.append([x, y, x + w, y + h])        # kembalikan xyxy

    return result


def square_crop_face(img: np.ndarray, box: list, padding_ratio=0.15):
    """
    Crop wajah secara square + padding dari bounding box [x1,y1,x2,y2].
    Return: cropped face (numpy BGR) atau None jika gagal.
    """
    x1, y1, x2, y2    = box
    h_orig, w_orig     = img.shape[:2]
    w_box, h_box       = x2 - x1, y2 - y1

    side = max(w_box, h_box)
    pad  = int(padding_ratio * side)
    cx   = x1 + w_box // 2
    cy   = y1 + h_box // 2
    half = (side // 2) + pad

    x1_p = max(0, cx - half)
    y1_p = max(0, cy - half)
    x2_p = min(w_orig, cx + half)
    y2_p = min(h_orig, cy + half)

    face = img[y1_p:y2_p, x1_p:x2_p]
    return face if face.size > 0 else None


# ================================================================
# LOAD MODEL
# ================================================================

def load_yolo_openvino(xml_path: Path, bin_path: Path):
    """Load YOLOv11 dari .xml + .bin via OpenVINO Core."""
    print("\n" + "=" * 70)
    print(f"{'LOADING YOLO OPENVINO':^70}")
    print("=" * 70)

    if not xml_path.exists():
        raise FileNotFoundError(f"❌ Tidak ditemukan: {xml_path}")
    if not bin_path.exists():
        raise FileNotFoundError(f"❌ Tidak ditemukan: {bin_path}")

    print(f"   .xml  : {xml_path}")
    print(f"   .bin  : {bin_path}")

    core      = ov.Core()
    device    = "GPU" if "GPU" in core.available_devices else "CPU"
    print(f"   Device: {device}")

    ov_model       = core.read_model(model=str(xml_path), weights=str(bin_path))
    compiled_model = core.compile_model(model=ov_model, device_name=device)

    input_layer  = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    print(f"   Input shape  : {input_layer.partial_shape}")
    print(f"   Output shape : {output_layer.partial_shape}")
    print("✅ YOLO OpenVINO loaded!\n")

    return compiled_model, output_layer


# ================================================================
# MAIN CROP PIPELINE
# ================================================================

def run_cropping():
    # ── Validasi dataset ────────────────────────────────────────
    if not DATASET_SOURCE.exists():
        raise FileNotFoundError(
            f"❌ Folder dataset tidak ditemukan: {DATASET_SOURCE}\n"
            f"   Pastikan path di config.py sudah benar."
        )

    # ── Load model ──────────────────────────────────────────────
    compiled_yolo, output_layer = load_yolo_openvino(YOLO_XML, YOLO_BIN)

    # ── Reset output folder ─────────────────────────────────────
    if CROP_OUTPUT.exists():
        shutil.rmtree(CROP_OUTPUT)
    CROP_OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"📂 Output folder : {CROP_OUTPUT.resolve()}\n")

    # ── Ambil daftar folder orang ───────────────────────────────
    student_folders = sorted([f for f in DATASET_SOURCE.iterdir() if f.is_dir()])
    if not student_folders:
        raise RuntimeError(f"❌ Tidak ada subfolder di {DATASET_SOURCE}. "
                           "Pastikan struktur dataset benar.")

    print("=" * 70)
    print(f"{'FACE CROPPING PIPELINE':^70}")
    print("=" * 70)
    print(f"📂 Dataset source : {DATASET_SOURCE.resolve()}")
    print(f"👥 Persons found  : {len(student_folders)}")
    print(f"🎯 Conf threshold : {YOLO_CONF_THRESHOLD}")
    print(f"📐 Target size    : {CROP_TARGET_SIZE}")
    print("=" * 70 + "\n")

    start_time   = time.time()
    summary_data = []
    total_saved  = 0

    for student_folder in tqdm(student_folders, desc="Overall Progress", unit="person"):
        identity = student_folder.name
        images   = sorted([
            f for f in student_folder.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        if not images:
            tqdm.write(f"   ⚠️  {identity}: tidak ada gambar, dilewati.")
            summary_data.append((identity, 0))
            continue

        count_saved  = 0
        count_failed = 0

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                count_failed += 1
                continue

            # Inferensi YOLO
            tensor = preprocess_yolo(img, YOLO_INPUT_SIZE)
            output = compiled_yolo([tensor])[output_layer]
            boxes  = postprocess_yolo(
                output, img.shape,
                YOLO_INPUT_SIZE,
                YOLO_CONF_THRESHOLD,
                YOLO_NMS_THRESHOLD,
            )

            if not boxes:
                count_failed += 1
                continue

            # Ambil 1 wajah terbesar (1 foto = 1 orang di dataset)
            boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            face = square_crop_face(img, boxes[0], CROP_PADDING_RATIO)

            if face is None:
                count_failed += 1
                continue

            # Resize + CLAHE
            face_resized  = cv2.resize(face, CROP_TARGET_SIZE, interpolation=cv2.INTER_AREA)
            face_enhanced = apply_clahe(face_resized)

            count_saved += 1
            filename = f"{identity}_{count_saved:03d}.jpg"
            cv2.imwrite(str(CROP_OUTPUT / filename), face_enhanced)

        summary_data.append((identity, count_saved))
        total_saved += count_saved

        tqdm.write(
            f"   ✅ {identity:<30} : {count_saved} saved"
            + (f"  ⚠️ {count_failed} failed" if count_failed else "")
        )

    duration = time.time() - start_time

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'FINAL SUMMARY':^70}")
    print("-" * 70)
    print(f"{'IDENTITY':<40} | {'FACES SAVED':>11}")
    print("-" * 70)
    for name, count in sorted(summary_data):
        print(f"{name:<40} | {count:>11}")
    print("-" * 70)
    print(f"{'TOTAL':<40} | {total_saved:>11}")
    print("=" * 70)
    print(f"\n⏱️  Processing time : {duration:.2f} seconds")
    print(f"📂 Output folder   : {CROP_OUTPUT.resolve()}")

    all_files = sorted(CROP_OUTPUT.glob("*.jpg"))
    print(f"🖼️  Total files     : {len(all_files)}")
    if all_files:
        print(f"   Sample          : {[f.name for f in all_files[:5]]}")
    print("=" * 70)


# ================================================================
if __name__ == "__main__":
    run_cropping()
