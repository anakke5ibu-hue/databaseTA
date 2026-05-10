import pickle

with open(r"C:\\Users\\syawal\\Downloads\Database TA\\EMBEDDINGS\\face_database.pkl", "rb") as f:
    data = pickle.load(f)

print("Keys:", data.keys())
print("Jumlah embeddings:", len(data["embeddings"]))
print("Contoh names:", data["names"][:3])