import pickle
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt

# Embedding output file path
pkl_path = r"C:\Users\ehdgh\Desktop\Programs\python\CA_TM\061525_SBERT\pdf_cleaning_73\Markdown\Cleaned\output\embedding_output.pkl"

with open(pkl_path, "rb") as f:
    embedding_output = pickle.load(f)

# Sentences and embeddings extraction
all_sentences = []
all_embeddings = []

for file_data in embedding_output.values():
    all_sentences.extend(file_data["sentences"])
    all_embeddings.extend(file_data["embeddings"])

# numpy array
all_embeddings = np.array(all_embeddings)

# data info
print(f"Total count: {len(all_sentences)}")
print(f"Emdedding Dimension: {all_embeddings.shape}")  

# UMAP model apply
reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(all_embeddings)

# 2D visualization of UMAP result
plt.figure(figsize=(10, 8))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=2, alpha=0.5)
plt.title("UMAP 2D Projection of SBERT Sentence Embeddings")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.grid(True)
plt.show()

# result save
npy_path = r"C:\Users\ehdgh\Desktop\Programs\python\CA_TM\061525_SBERT\pdf_cleaning_73\Markdown\Cleaned\output\embedding_2d.npy"
np.save(npy_path, embedding_2d)
print(f"Completed: {npy_path}")

