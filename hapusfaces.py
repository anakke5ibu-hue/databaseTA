import psycopg2
from tkinter import messagebox

# Konfigurasi Database (Samakan dengan main_2.py kamu)
DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "2026",
    "port": "5432"
}

def delete_user_by_nim(nim):
    conn = None
    try:
        # 1. Koneksi ke Postgres
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # 2. Cari ID user berdasarkan NIM terlebih dahulu
        cur.execute("SELECT id, nama FROM users_parkir WHERE nim = %s", (nim,))
        user = cur.fetchone()

        if not user:
            print(f"Data dengan NIM {nim} tidak ditemukan.")
            return False

        user_id, nama = user
        print(f"Menghapus data: {nama} ({nim})...")

        # 3. Hapus data di tabel wajah_embeddings dulu (karena foreign key)
        cur.execute("DELETE FROM wajah_embeddings WHERE user_id = %s", (user_id,))
        
        # 4. Hapus data di tabel users_parkir
        cur.execute("DELETE FROM users_parkir WHERE id = %s", (user_id,))

        # 5. Commit perubahan
        conn.commit()
        print(f"Berhasil! Data {nama} telah dihapus dari database.")
        cur.close()
        return True

    except Exception as e:
        print(f"Error saat menghapus: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    target_nim = input("Masukkan NIM yang ingin dihapus: ")
    
    # Konfirmasi sebelum hapus
    confirm = input(f"Yakin ingin menghapus NIM {target_nim}? (y/n): ")
    if confirm.lower() == 'y':
        delete_user_by_nim(target_nim)
    else:
        print("Penghapusan dibatalkan.")