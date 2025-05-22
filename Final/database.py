import sqlite3
import numpy as np

def initialize_db(db_path="embeddings.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_name TEXT,
                    page_num INTEGER,
                    chunk TEXT,
                    embedding BLOB)''')
    
    conn.commit()
    conn.close()

def store_embedding(pdf_name, page_num, chunk, embedding, db_path="embeddings.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('''INSERT INTO embeddings (pdf_name, page_num, chunk, embedding) 
                 VALUES (?, ?, ?, ?)''', 
              (pdf_name, page_num, chunk, sqlite3.Binary(embedding.tobytes())))
    
    conn.commit()
    conn.close()

def fetch_embeddings(query_embedding, top_k=3, db_path="embeddings.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('SELECT id, pdf_name, page_num, chunk, embedding FROM embeddings')
    rows = c.fetchall()

    embeddings = []
    for row in rows:
        embedding = np.frombuffer(row[4], dtype=np.float32)
        embeddings.append((row[0], row[1], row[2], row[3], embedding))

    conn.close()

    distances = []
    for row in embeddings:
        dist = np.linalg.norm(row[4] - query_embedding)
        distances.append((row, dist))

    distances.sort(key=lambda x: x[1])
    return distances[:top_k]
