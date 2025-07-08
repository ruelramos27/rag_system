from gtts import gTTS
import os
import time


def speech_text(answer, filename):
    tts = gTTS(text=answer, lang='en', slow=False)
    tts.save(filename)
    os.system(f'start "" "{os.path.abspath(filename)}"')

# Wait long enough for the audio to finish (adjust if needed)
    time.sleep(20)  # seconds

# Now delete
    if os.path.exists(filename):
        os.remove(filename)
        print("✅ MP3 deleted.")
    else:
        print("⚠️ File not found.")


