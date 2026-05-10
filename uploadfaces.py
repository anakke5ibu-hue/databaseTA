import pickle
import psycopg2
import os
from pgvector.psycopg2 import register_vector

# --- KONFIGURASI ---
FILE_PKL = "C:\\Users\\syawal\\Downloads\\Database TA\\embeddings\\face_database.pkl"
DB_CONFIG = {
    "host": "localhost",
    "database": "postgres", # Pastikan nama DB sesuai pgAdmin
    "user": "postgres",
    "password": "2026", # Password kamu
    "port": "5432"
}

def migrate_data():
    try:
        # 1. Load data dari file pkl
        if not os.path.exists(FILE_PKL):
            print(f"Error: File {FILE_PKL} tidak ditemukan!")
            return

        with open(FILE_PKL, 'rb') as f:
            data = pickle.load(f)

        list_embedding = data['embeddings']
        list_paths = data['image_paths']
        total_data = len(list_paths)

        # 2. Koneksi ke Postgres
        conn = psycopg2.connect(**DB_CONFIG)
        register_vector(conn)
        cur = conn.cursor()

        print(f"Mulai migrasi {total_data} data wajah...")

        for i in range(total_data):
            # Ambil nama file tanpa ekstensi (misal: 1101223224_Rheira Nisrina Abiyah_030)
            full_path = list_paths[i]
            file_name = os.path.splitext(os.path.basename(full_path))[0]
            
            # Logika Pemisahan: NIM_NAMA_NOMOR
            parts = file_name.split("_")
            
            if len(parts) >= 2:
                nim_asli = parts[0]   # Mengambil 1101223224
                nama_asli = parts[1]  # Mengambil Rheira Nisrina Abiyah
            else:
                nim_asli = f"UNKNOWN-{i}"
                nama_asli = file_name

            # 1. Insert ke tabel users_parkir (Gunakan ON CONFLICT agar NIM tidak double)
            cur.execute(
                "INSERT INTO users_parkir (nama, nim) VALUES (%s, %s) "
                "ON CONFLICT (nim) DO UPDATE SET nama = EXCLUDED.nama RETURNING id",
                (nama_asli, nim_asli)
            )
            user_id = cur.fetchone()[0]

            # 2. Insert ke tabel wajah_embeddings
            cur.execute(
                "INSERT INTO wajah_embeddings (user_id, embedding, foto_path) VALUES (%s, %s, %s)",
                (user_id, list_embedding[i], full_path)
            )

            print(f"   [OK] {nim_asli} - {nama_asli}")

        conn.commit()
        print("\nMigrasi Selesai! Nama sudah bersih dan database rapi.")

    except Exception as e:
        print(f"Error: {e}")
        if 'conn' in locals(): conn.rollback()
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    migrate_data()