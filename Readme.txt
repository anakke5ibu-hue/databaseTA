Skript untuk Cek Database
SQL
SELECT nim, nama, COUNT(id) as jumlah_foto 
FROM (
    SELECT u.nim, u.nama, w.id 
    FROM users_parkir u 
    JOIN wajah_embeddings w ON u.id = w.user_id
) AS data 
GROUP BY nim, nama;

uvicorn main:app --host 0.0.0.0 --port 8000 ===> untuk menjalankan