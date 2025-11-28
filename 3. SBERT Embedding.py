from sentence_transformers import SentenceTransformer
import os
import pickle

# SBERT Module
model = SentenceTransformer('all-MiniLM-L6-v2')
cleaned_folder = r"C:__________________"

# Output folder path
output_folder = os.path.join(cleaned_folder, "output")
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "embedding_output.pkl")
embedding_output = {}

# SBERT Embedding 
for filename in os.listdir(cleaned_folder):
    file_path = os.path.join(cleaned_folder, filename)
    if filename.endswith(".txt") and os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if len(line.strip()) > 0]

        embeddings = model.encode(sentences, show_progress_bar=True)
        embedding_output[filename] = {
            "sentences": sentences,
            "embeddings": embeddings
        }

# Save embeddings using pickle
with open(output_path, "wb") as f:
    pickle.dump(embedding_output, f)

print(f"Completed: {output_path}")

