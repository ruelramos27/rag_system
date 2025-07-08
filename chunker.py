
import os
import json

CHUNK_FILE = "chunks.json"
PROCESSED_FILES = "processed_files.json"

def chunk_text(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def save_chunks(chunks):
    with open(CHUNK_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

def load_chunks():
    if os.path.exists(CHUNK_FILE):
        with open(CHUNK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def load_processed_files():
    if os.path.exists(PROCESSED_FILES):
        with open(PROCESSED_FILES, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_processed_files(files):
    with open(PROCESSED_FILES, "w", encoding="utf-8") as f:
        json.dump(files, f)
