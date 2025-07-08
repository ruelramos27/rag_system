
import os

def get_new_text_files(directory, already_processed):
    new_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt") and filename not in already_processed:
            new_files.append(filename)
    return new_files

def load_text_from_files(directory, file_list):
    full_text = ""
    for filename in file_list:
        path = os.path.join(directory, filename)
        with open(path, "r", encoding="utf-8") as f:
            full_text += f.read() + "\n"
    return full_text
