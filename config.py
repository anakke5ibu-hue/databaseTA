# ================================================================
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
