from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def calculate_similarities(model_choice, query_text, documents):
    """Calcul des similartiés cosinus entre la question et les documents"""
    model = SentenceTransformer(model_choice)
    query_vec = model.encode([query_text])
    doc_vecs = model.encode(documents)
    similarities = cosine_similarity(doc_vecs, query_vec).flatten()
    return similarities, doc_vecs, query_vec


def rank_documents(documents, similarities, doc_vecs, threshold):
    """Tri des documents par similarité décroissante"""
    ranked = sorted(
        zip(documents, similarities, doc_vecs), key=lambda x: x[1], reverse=True
    )
    filtered = [(doc, sim, vec) for doc, sim, vec in ranked if sim >= threshold]
    return filtered


def do_graph(filtered, documents, query_vec):
    # PCA pour affichage 2D https://fr.wikipedia.org/wiki/Analyse_en_composantes_principales
    filtered_docs, filtered_sims, filtered_vecs = zip(*filtered)
    reduced = PCA(n_components=2).fit_transform(np.vstack([query_vec, filtered_vecs]))
    query_2d, docs_2d = reduced[0], reduced[1:]

    # Tracé
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(docs_2d[:, 0], docs_2d[:, 1], c="blue", label="Documents")
    ax.scatter(query_2d[0], query_2d[1], c="red", marker="X", s=100, label="Requête")

    for i, (x, y) in enumerate(docs_2d):
        ax.text(x + 0.01, y + 0.01, f"{filtered_sims[i]:.2f}", fontsize=9)
        ax.text(x + 0.01, y - 0.04, f"{i + 1}. {documents[i]}", fontsize=7)

    ax.set_title("Projection PCA 2D avec seuil de similarité")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(True)
    ax.legend()
    return fig, filtered_docs, filtered_sims
