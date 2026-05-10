from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Optional
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import numpy as np
import cv2
from openvino import Core
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════
# LOAD MODEL AI (sekali saat server start)
# ═══════════════════════════════════════════════════════════════
print("[INIT] Loading YOLO face detector...")
face_detector = YOLO("yolov11n-face_openvino_model")
print("[SUCCESS] YOLO loaded!")

print("[INIT] Loading ArcFace OpenVINO model...")
ie = Core()
compiled_model = ie.compile_model(model="buffalo_sc_rec.xml", device_name="CPU")
output_layer = compiled_model.output(0)
print("[SUCCESS] ArcFace loaded!")

app = FastAPI()

# --- AGAR TEMANMU BISA AKSES (CORS) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Konfigurasi Database
DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "2026",
    "port": "5432"
}

# Path ke file face_database.pkl milik deteksi_final.py
# Sesuaikan dengan lokasi folder recog kamu
PKL_PATH = "models/face_database.pkl"

def update_pkl_database(nama: str, nim: str, embedding: np.ndarray):
    """
    Update file face_database.pkl agar user baru langsung dikenali
    oleh deteksi_final.py TANPA perlu restart server.
    """
    import pickle
    from pathlib import Path

    db_path = Path(PKL_PATH)
    key = f"{nim}_{nama}"

    if db_path.exists():
        with open(db_path, "rb") as f:
            db = pickle.load(f)
        embeddings = db.get("embeddings", np.array([]))
        names      = db.get("names", [])
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim < 2:
            embeddings = np.array(embeddings).reshape(-1, embedding.shape[0]) if len(embeddings) else np.empty((0, embedding.shape[0]))
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings = np.empty((0, embedding.shape[0]))
        names      = []

    # Kalau nim sudah ada, update embedding-nya (jangan duplikat)
    if key in names:
        idx = names.index(key)
        embeddings[idx] = embedding
        print(f"[PKL] Update embedding untuk {key}")
    else:
        embeddings = np.vstack([embeddings, embedding.reshape(1, -1)])
        names.append(key)
        print(f"[PKL] Tambah user baru ke pkl: {key}")

    with open(db_path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "names": names}, f)
    print(f"[PKL] face_database.pkl diperbarui — total {len(names)} orang")

# ═══════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════

class SearchRequest(BaseModel):
    embedding: List[float]

class LogRequest(BaseModel):
    nama: str
    nim: str
    waktu: str
    tanggal: str
    gate: Optional[str] = "Gate 4"
    confidence: Optional[float] = None
    status: str

# ═══════════════════════════════════════════════════════════════
# AUTO DELETE LOG LEBIH DARI 1 BULAN
# ═══════════════════════════════════════════════════════════════

def auto_delete_old_logs():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        satu_bulan_lalu = datetime.now() - timedelta(days=30)
        cur.execute(
            "DELETE FROM log_akses WHERE tanggal < %s;",
            (satu_bulan_lalu.date(),)
        )
        deleted_count = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        print(f"[AUTO CLEANUP] {deleted_count} log dihapus (lebih dari 30 hari) — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"[AUTO CLEANUP] Gagal: {e}")

# Jalankan scheduler — cek setiap hari jam 00:00
scheduler = BackgroundScheduler()
scheduler.add_job(auto_delete_old_logs, 'cron', hour=0, minute=0)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/")
def home():
    return {"status": "Online", "message": "Jembatan FastAPI-PostgreSQL Siap!"}

# ─── Identifikasi Wajah ───────────────────────────────────────
# ─── Identifikasi Wajah ───────────────────────────────────────
@app.post("/identify-face")
def identify_face(request: SearchRequest):
    print(f"ANGKA EMBEDDING DARI GATE: {request.embedding}")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        vector_str = str(request.embedding)

        query = """
            SELECT u.nama, u.nim, u.is_blocked, w.embedding <=> %s AS distance
            FROM users_parkir u
            JOIN wajah_embeddings w ON u.id = w.user_id
            ORDER BY distance ASC
            LIMIT 1;
        """

        cur.execute(query, (vector_str,))
        result = cur.fetchone()

        if result:
            print(f"DEBUG SERVER -> Terdeteksi: {result['nama']} | Distance: {result['distance']} | Blocked: {result['is_blocked']}")
        else:
            print("DEBUG SERVER -> Tidak ada wajah yang mirip di database")

        cur.close()
        conn.close()

        if result and result['distance'] < 0.5:
            if result['is_blocked']:
                return {"status": "unknown", "message": "Akses ditolak (akun diblokir)"}
            else:
                return {
                    "status": "success",
                    "data": {
                        "nama": result['nama'],
                        "nim": result['nim'],
                        "similarity": round(1 - result['distance'], 2)
                    }
                }
        else:
            return {"status": "unknown", "message": "Wajah tidak dikenali"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Register User ────────────────────────────────────────────
@app.post("/register")
def register(nama: str, nim: str, embedding: List[float]):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("INSERT INTO users_parkir (nama, nim) VALUES (%s, %s) RETURNING id", (nama, nim))
        user_id = cur.fetchone()[0]
        cur.execute("INSERT INTO wajah_embeddings (user_id, embedding) VALUES (%s, %s)", (user_id, str(embedding)))
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "Berhasil daftar!"}
    except Exception as e:
        return {"status": "Gagal", "error": str(e)}

# ─── Register User dari Foto (dipanggil dari RegistrasiPage) ──
@app.post("/register_from_photos")
async def register_from_photos(
    nama: str = Form(...),
    nim: str = Form(...),
    photos: List[UploadFile] = File(...)
):
    print(f"[REGISTER] Menerima request registrasi: {nama} ({nim}), {len(photos)} foto")
    all_embeddings = []

    for i, photo in enumerate(photos):
        contents = await photo.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            print(f"[WARNING] Foto ke-{i+1} gagal didecode, dilewati")
            continue

        # Deteksi wajah pakai YOLO
        results = face_detector(frame, verbose=False, conf=0.5)

        if len(results[0].boxes) == 0:
            print(f"[WARNING] Tidak ada wajah terdeteksi di foto ke-{i+1}, dilewati")
            continue

        # Crop wajah
        box = results[0].boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        # Preprocessing ArcFace: resize 112x112, NCHW, normalisasi InsightFace
        face_resized = cv2.resize(face_crop, (112, 112))
        input_blob = face_resized.transpose(2, 0, 1).reshape(1, 3, 112, 112).astype(np.float32)
        input_blob = (input_blob - 127.5) / 128.0

        # Inferensi OpenVINO
        result = compiled_model([input_blob])[output_layer]
        all_embeddings.append(result.flatten())
        print(f"[OPENVINO] Embedding berhasil dari foto ke-{i+1}")

    if not all_embeddings:
        raise HTTPException(status_code=400, detail="Wajah tidak terdeteksi di semua foto. Pastikan wajah terlihat jelas.")

    # Rata-rata embedding + normalisasi L2
    final_embedding = np.mean(all_embeddings, axis=0)
    final_embedding /= np.linalg.norm(final_embedding)
    embedding_list = final_embedding.tolist()
    print(f"[SUCCESS] Final embedding dari {len(all_embeddings)} foto siap disimpan")

    # Simpan ke PostgreSQL
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("INSERT INTO users_parkir (nama, nim) VALUES (%s, %s) RETURNING id", (nama, nim))
        user_id = cur.fetchone()[0]
        cur.execute("INSERT INTO wajah_embeddings (user_id, embedding) VALUES (%s, %s)", (user_id, str(embedding_list)))
        conn.commit()
        cur.close()
        conn.close()
        print(f"[DB] User {nama} ({nim}) berhasil disimpan ke database")

        # ── Update face_database.pkl agar langsung dikenali tanpa restart ──
        try:
            update_pkl_database(nama, nim, final_embedding)
        except Exception as pkl_err:
            print(f"[WARNING] Gagal update pkl: {pkl_err} — tapi data sudah tersimpan di DB")

        return {"message": f"User {nama} ({nim}) berhasil didaftarkan dari {len(all_embeddings)} foto!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal simpan ke database: {str(e)}")

# ─── Ambil Semua User (untuk tab Blokir User) ─────────────────
@app.get("/users")
def get_users():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
                    
            SELECT 
                u.id, u.nama, u.nim, u.is_blocked,
                l.waktu AS jam_terakhir,
                l.tanggal AS tanggal_terakhir
            FROM users_parkir u
            LEFT JOIN LATERAL (
                SELECT waktu, tanggal FROM log_akses
                WHERE nim = u.nim
                ORDER BY no DESC
                LIMIT 1
            ) l ON true
            ORDER BY u.id ASC;
        """)
        results = cur.fetchall()
        cur.close()
        conn.close()
        return list(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # ─── Ambil daftar NIM yang diblokir (dipanggil deteksi_final.py) ──
@app.get("/blokir_user")
def get_blokir_user():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        # Ambil hanya user yang is_blocked = TRUE
        cur.execute("SELECT nim FROM users_parkir WHERE is_blocked = TRUE;")
        results = cur.fetchall()
        cur.close()
        conn.close()
        return list(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Blokir User (set is_blocked = TRUE & catat ke blokir_user) ───
@app.post("/users/{nim}/block")
def block_user(nim: str):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        # Update status blokir
        cur.execute("UPDATE users_parkir SET is_blocked = TRUE WHERE nim = %s RETURNING id", (nim,))
        user = cur.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User tidak ditemukan")
        
        # Catat ke tabel blokir_user
        cur.execute("""
            INSERT INTO blokir_user (nama, nim, waktu, tanggal, status, alasan)
            SELECT nama, nim, NOW()::text, CURRENT_DATE, 'BLOKIR', 'Diblokir oleh admin'
            FROM users_parkir WHERE nim = %s
        """, (nim,))
        
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "success", "message": f"User {nim} diblokir"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Unblock User (set is_blocked = FALSE & catat ke blokir_user) ───
@app.post("/users/{nim}/unblock")
def unblock_user(nim: str):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("UPDATE users_parkir SET is_blocked = FALSE WHERE nim = %s RETURNING id", (nim,))
        user = cur.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User tidak ditemukan")
        
        # Catat ke tabel blokir_user
        cur.execute("""
            INSERT INTO blokir_user (nama, nim, waktu, tanggal, status, alasan)
            SELECT nama, nim, NOW()::text, CURRENT_DATE, 'UNBLOKIR', 'Dibuka blokir oleh admin'
            FROM users_parkir WHERE nim = %s
        """, (nim,))
        
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "success", "message": f"User {nim} diunblok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# # ─── Hapus User / Blokir (hapus dari users_parkir) ────────────
# @app.delete("/users/{nim}")
# def delete_user(nim: str):
#     try:
#         conn = psycopg2.connect(**DB_CONFIG)
#         cur = conn.cursor()
#         cur.execute("""
#             DELETE FROM wajah_embeddings
#             WHERE user_id = (SELECT id FROM users_parkir WHERE nim = %s);
#         """, (nim,))
#         cur.execute("DELETE FROM users_parkir WHERE nim = %s RETURNING id;", (nim,))
#         deleted = cur.fetchone()
#         conn.commit()
#         cur.close()
#         conn.close()
#         if deleted:
#             return {"status": "success", "message": f"User {nim} berhasil dihapus"}
#         else:
#             raise HTTPException(status_code=404, detail="User tidak ditemukan")
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# ─── Simpan Log Akses (dipanggil realtime dari Dashboard) ─────
@app.post("/log_akses")
def save_log(entry: LogRequest):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO log_akses (nama, nim, waktu, tanggal, gate, confidence, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """, (
            entry.nama,
            entry.nim,
            entry.waktu,
            entry.tanggal,
            entry.gate,
            entry.confidence,
            entry.status,
        ))
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Ambil Log Akses (untuk tab Log Akses di Dashboard) ───────
@app.get("/log_akses")
def get_logs():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT no, nama, nim, waktu, tanggal, gate, confidence, status
            FROM log_akses
            ORDER BY no DESC
            LIMIT 500;
        """)
        results = cur.fetchall()
        cur.close()
        conn.close()
        return list(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# SQL UNTUK BUAT TABEL (jalankan sekali di psql kalau belum ada)
# ═══════════════════════════════════════════════════════════════
#
# CREATE TABLE IF NOT EXISTS log_akses (
#     no         SERIAL PRIMARY KEY,
#     nama       TEXT,
#     nim        TEXT,
#     waktu      TEXT,
#     tanggal    DATE,
#     gate       TEXT DEFAULT 'Gate 4',
#     confidence FLOAT,
#     status     TEXT
# );
#
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)