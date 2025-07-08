from utils import get_new_text_files, load_text_from_files
from chunker import chunk_text, save_chunks, load_chunks, load_processed_files, save_processed_files
from database import build_index, load_index, append_to_index, retrieve
from ollama_client import query_ollama
from sentence_transformers import SentenceTransformer
from s2t import speech_text
import gc

import os, queue, sys, json, numpy as np
import sounddevice as sd
import vosk

# === Setup Vosk model path ===
MODEL_PATH = "vosk-model-small-en-us-0.15"
vosk_model = None  
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Vosk model not found at {MODEL_PATH}")
    vosk_model = vosk.Model(MODEL_PATH)
except Exception as e:
    print(e)
    sys.exit(1)

# === Function to capture voice ===
def listen_once(volume_threshold=1.0):
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print("⚠️", status, file=sys.stderr)
        volume_norm = np.linalg.norm(indata) * 10
        if volume_norm < volume_threshold:
            return
        q.put(bytes(indata))

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
        print("🎤 Listening... Speak now.")

        try:
            while True:
                data = q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        print("🗣 You said:", text)
                        return text
        except KeyboardInterrupt:
            print("\n⛔ Interrupted.")
            return None


def main():
    folder_path = "text"
    processed_files = load_processed_files()
    all_chunks = load_chunks()
    new_files = get_new_text_files(folder_path, processed_files)

    if new_files:
        print(f"🆕 New files detected: {new_files}")
        new_text = load_text_from_files(folder_path, new_files)
        new_chunks = chunk_text(new_text)
        all_chunks += new_chunks
        save_chunks(all_chunks)
        save_processed_files(processed_files + new_files)
        print(f"✅ Added {len(new_chunks)} new chunks.")
    else:
        print("📂 No new files to process.")

    index, embeddings = load_index()
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    if index is None:
        print("📦 Building new FAISS index...")
        embedder, index = build_index(all_chunks)
        print("✅ New index built.")
    elif new_files:
        print("📦 Appending new chunks to FAISS index...")
        append_to_index(index, embedder, new_chunks)
        print("✅ Index updated.")

    # === CHOOSE INPUT METHOD ===
    while True:
        print("\n🔍 Choose input method:")
        print("1. 🎙️ Voice Input")
        print("2. 📝 Text Input")
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            mode = "voice"
            print("🎙️ You chose Voice Input.")
            question = listen_once()
            retrieved_chunks = retrieve(question, embedder, index, all_chunks)
            context = "\n".join(retrieved_chunks)
            answer = query_ollama(context, question)
            print("\n💬 Ollama Answer:\n", answer)
            speech_text(answer, filename="response.mp3")
        
        elif choice == "2":
            mode = "text"
            print("📝 You chose Text Input.")
            question = input("❓ Enter your question: ")
            retrieved_chunks = retrieve(question, embedder, index, all_chunks)
            context = "\n".join(retrieved_chunks)
            answer = query_ollama(context, question)
            print("\n💬 Ollama Answer:\n", answer)
            speech_text(answer, filename="response.mp3")

            
        else:
            print("GOODBYE")
            break
      
    



if __name__ == "__main__":
    main()
    # Clean up Vosk model to free memory    
    if vosk_model:
     del vosk_model
     import gc
     gc.collect()

