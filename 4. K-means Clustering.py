import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from keybert import KeyBERT
import re

# Korean font setting
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

# Embedding and sentence data load
embedding_path = r"C:________________________________________"

# Load the full embedding and sentences
pkl_path = r"C:________________________________________.pkl"
with open(pkl_path, "rb") as f:
    embedding_output = pickle.load(f)

all_sentences = []
all_embeddings = []

for entry in embedding_output.values():
    all_sentences.extend(entry["sentences"])
    all_embeddings.extend(entry["embeddings"])
all_embeddings = np.array(all_embeddings)

### K-means Clustering: Elbow Method and Silhouette Score
# 1. Elbow Method 
elbow_inertia = []
K_range = range(2, 16)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embedding_2d)
    elbow_inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, elbow_inertia, marker='o')
plt.title("Elbow Method: Optimal Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.xticks(K_range)
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Silhouette Score 
sil_scores = []
K_refined = range(2, 16)

for k in K_refined:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embedding_2d)
    score = silhouette_score(embedding_2d, labels)
    sil_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(K_refined, sil_scores, marker='o', color='green')
# plt.title("Silhouette Score: Refined Range (k = 2 to 16)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.xticks(K_refined)
plt.grid(True)
plt.tight_layout()
plt.show()


### AFTER Model Decision 
# Cluster with KMeans
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embedding_2d)

# Cluster ID with example sentences
cluster_examples = defaultdict(list)
for sent, label in zip(all_sentences, cluster_labels):
    cluster_examples[label].append(sent)

# Sentences count per cluster
print("\n Sentence Count:")
for cluster_id in sorted(cluster_examples.keys()):
    count = len(cluster_examples[cluster_id])
    print(f" - Cluster {cluster_id}: {count} ")


### Representative sentences per cluster (top 30 by cosine similarity) =========
import numpy as np
from sklearn.preprocessing import normalize

# 1) L2-normalize original SBERT embeddings once
emb_norm = normalize(all_embeddings, norm="l2")

def print_top_representatives_per_cluster(
    emb_norm, labels, sentences, top_n=30
):
    unique_clusters = sorted(np.unique(labels))
    for cid in unique_clusters:
        # indices for this cluster (based on UMAP-space labels)
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            print(f"\n[Cluster {cid}] (empty)")
            continue

        # 2) centroid in SBERT space (mean of normalized vectors), then re-normalize
        centroid = emb_norm[idx].mean(axis=0, keepdims=True)
        centroid /= (np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-12)

        # 3) cosine similarity = dot product because vectors are L2-normalized
        sims = emb_norm[idx] @ centroid.T  # shape (Nc, 1)
        sims = sims.ravel()

        take = min(top_n, idx.size)
        top_local = np.argsort(-sims)[:take]

        print(f"\n[Cluster {cid}] n={idx.size} | Top {take} representative sentences (cosine):")
        for rank, loc in enumerate(top_local, 1):
            global_i = idx[loc]
            print(f"{rank:2d}. {all_sentences[global_i]}")

print_top_representatives_per_cluster(
    emb_norm=emb_norm,
    labels=cluster_labels,
    sentences=all_sentences,
    top_n=30
)

