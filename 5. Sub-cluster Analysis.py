
import os
import glob
import re
import pickle
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
import matplotlib.pyplot as plt
import difflib

# data paths
PICKLE_PATH = r"C:\Users\ehdgh\Desktop\Programs\python\CA_TM\073025_SBERT\pdf_cleaning_73\Markdown\Cleaned\output\cluster3_data.pkl"
TXT_DIR     = r"C:\Users\ehdgh\Desktop\Programs\python\CA_TM\073025_SBERT\pdf_cleaning_73\Markdown\Cleaned"

# Load Cluster 3 data
with open(PICKLE_PATH, "rb") as f:
    data = pickle.load(f)
cluster3_sentences  = data["sentences"]
cluster3_embeddings = np.array(data["embeddings"])

# Sub-cluster analysis within Cluster 3
inertias = []
sil_scores = []
K_range = range(2, 11)
for k in K_range:
    kmeans_tmp = KMeans(n_clusters=k, random_state=42)
    labels_tmp = kmeans_tmp.fit_predict(cluster3_embeddings)
    inertias.append(kmeans_tmp.inertia_)
    sil_scores.append(silhouette_score(cluster3_embeddings, labels_tmp))
plt.plot(K_range, sil_scores, marker='o')
plt.title("Silhouette Score for Sub-Cluster Count")
plt.xlabel("Number of Sub-Clusters")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()
best_k = K_range[np.argmax(sil_scores)]

n_clusters = 5  
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
sub_labels = kmeans.fit_predict(cluster3_embeddings)

#  Sub-cluster sentence grouping
subcluster_sentences = defaultdict(list)
for sent, label in zip(cluster3_sentences, sub_labels):
    subcluster_sentences[label].append(sent)


# Sentence to Citation Mapping
def normalize_text(s: str) 
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)  # 문장부호 제거
    return s.strip()

# Text files load
file_texts      = {}  # {path: raw_text}
file_texts_norm = {}  # {path: normalized_text}

for path in glob.glob(os.path.join(TXT_DIR, "*.txt")):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        file_texts[path] = raw
        file_texts_norm[path] = normalize_text(raw)
    except Exception:
        pass

# File name with (Lastname et al., YEAR)
def filename_to_citation(path: str) -> str:
    """
    기대형식 예:
      "Kelly et al. - 2018 - Title.txt"
      "Jeon and Park - 2018 - Title.txt"
      "Kim, Utz, and Choi - 2019 - Title.txt"
      "Smith - 2020 - Title.txt"
    위 어떤 경우든 "(Kelly et al., 2018)" 형태를 생성.
    """
    fname = os.path.basename(path)

    # "<authors> - <year> - ..." 
    m = re.match(r"^(.*?)\s*-\s*(\d{4})\s*-\s*.*", fname)
    if not m:
        return "(source unknown)"

    authors_str, year = m.group(1), m.group(2)
    a = authors_str.strip()

    # 1) "et al." 
    if re.search(r"\bet\.?\s*al\.?\b", a, flags=re.I):
        first = a.split()[0]
        author_part = f"{first} et al."

    else:
        # 2) First Author + et al.
        if "," in a:
            first = a.split(",")[0].strip()
            author_part = f"{first} et al."

        # 3) and multi author → First + et al.
        elif re.search(r"\band\b", a, flags=re.I):
            first = re.split(r"\band\b", a, flags=re.I)[0].strip()
            author_part = f"{first} et al."

        # 4) Etc
        else:
            author_part = a

    # remove extra spaces and trailing dot
    author_part = re.sub(r"\s+", " ", author_part).strip().rstrip(".")
    return f"({author_part}, {year})"

# search best source file for a sentence
def token_jaccard(a_tokens, b_tokens):
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / union if union else 0.0

def best_source_for_sentence(sentence: str,
                             jaccard_threshold: float = 0.15,
                             difflib_threshold: float = 0.82) -> str 
    # 1) Exact substring match
    for path, raw in file_texts.items():
        if sentence in raw:
            return path

    # 2) Token Jaccard on normalized text
    s_norm = normalize_text(sentence)
    s_tokens = set(s_norm.split())
    best_path, best_score = None, 0.0

    for path, text_norm in file_texts_norm.items():
        t_tokens = set(text_norm.split())
        overlap = len(s_tokens & t_tokens)
        if overlap < 3:
            continue  
        score = token_jaccard(s_tokens, t_tokens)
        if score > best_score:
            best_score = score
            best_path = path

    if best_path and best_score >= jaccard_threshold:
        return best_path

    # 3) difflib(sequence matcher) on normalized text
    best_path_dl, best_score_dl = None, 0.0
    for path, text_norm in file_texts_norm.items():
        score = difflib.SequenceMatcher(None, s_norm, text_norm).ratio()
        if score > best_score_dl:
            best_score_dl = score
            best_path_dl = path

    if best_path_dl and best_score_dl >= difflib_threshold:
        return best_path_dl

    return None

# Cache for sentence to citation mapping
sentence_citation_cache = {}

def sentence_to_citation(sentence: str) -> str:
    if sentence in sentence_citation_cache:
        return sentence_citation_cache[sentence]
    src = best_source_for_sentence(sentence)
    cit = filename_to_citation(src) if src else "(source unknown)"
    sentence_citation_cache[sentence] = cit
    return cit


### Sub-cluster Representative Sentences with Citations
n_clusters_detected = len(set(sub_labels))

for cluster_id in range(n_clusters_detected):
    cluster_indices = [i for i, lbl in enumerate(sub_labels) if lbl == cluster_id]
    if not cluster_indices:
        print(f"\n❗ Cluster {cluster_id} is empty.")
        continue

    cluster_embeddings = np.array([cluster3_embeddings[i] for i in cluster_indices])
    cluster_sentences  = [cluster3_sentences[i] for i in cluster_indices]

    # Mean embedding and similarities
    mean_embedding = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
    similarities = cosine_similarity(cluster_embeddings, mean_embedding).flatten()

    # Top-N representative sentences
    TOP_N = 35
    top_indices_local = np.argsort(similarities)[-TOP_N:][::-1]

    print(f"\n[Sub-cluster {cluster_id}] Top n Representative Sentences:")
    for rank, local_idx in enumerate(top_indices_local, 1):
        sent = cluster_sentences[local_idx]
        cit  = sentence_to_citation(sent)
        print(f"{rank}. {sent} {cit}")

