
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

def build_index(chunks, save_path="index.faiss", emb_path="embeddings.npy"):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks)
    np.save(emb_path, embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, save_path)
    return embedder, index

def load_index(index_path="index.faiss", emb_path="embeddings.npy"):
    if os.path.exists(index_path) and os.path.exists(emb_path):
        index = faiss.read_index(index_path)
        embeddings = np.load(emb_path)
        return index, embeddings
    return None, None

def append_to_index(index, embedder, new_chunks, emb_path="embeddings.npy", index_path="index.faiss"):
    new_embeddings = embedder.encode(new_chunks)
    index.add(new_embeddings)
    if os.path.exists(emb_path):
        old_embeddings = np.load(emb_path)
        all_embeddings = np.vstack((old_embeddings, new_embeddings))
    else:
        all_embeddings = new_embeddings
    np.save(emb_path, all_embeddings)
    faiss.write_index(index, index_path)

def retrieve(query, embedder, index, chunks, k=3):
    query_vec = embedder.encode([query])
    _, I = index.search(query_vec, k)
    return [chunks[i] for i in I[0]]
