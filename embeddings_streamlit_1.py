import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

MAX_LABEL = 60

st.set_page_config(page_title="Similarité Cosinus", layout="wide")

st.title("🔍 Visualisation de la Similarité Cosinus avec des Embeddings")

# Entrée utilisateur
query_text = st.text_input(
    "🔴 Requête (texte à comparer)", "l’intelligence artificielle dans l’éducation"
)

default_docs = [
    "les robots dans les écoles",
    "les voitures autonomes sur les routes",
    "l’apprentissage automatique et les étudiants",
    "le réchauffement climatique et les océans",
    "l’enseignement assisté par IA en classe",
]

st.markdown("✏️ Modifiez les documents ci-dessous :")
documents = []
cols = st.columns(2)
for i in range(len(default_docs)):
    doc = cols[i % 2].text_input(f"Document {i + 1}", value=default_docs[i])
    documents.append(doc)

if st.button("🧠 Calculer et Afficher"):
    with st.spinner("Encodage des textes et calcul des similarités..."):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_vec = model.encode([query_text])
        doc_vecs = model.encode(documents)
        similarities = cosine_similarity(doc_vecs, query_vec).flatten()

        # Réduction PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(np.vstack([query_vec, doc_vecs]))
        query_2d, docs_2d = reduced[0], reduced[1:]

        # Tracé matplotlib
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(docs_2d[:, 0], docs_2d[:, 1], c="blue", label="Documents")
        ax.scatter(
            query_2d[0], query_2d[1], c="red", label="Requête", marker="X", s=100
        )

        for i, (x, y) in enumerate(docs_2d):
            ax.text(x + 0.01, y + 0.01, f"{similarities[i]:.2f}", fontsize=9)
            ax.text(
                x + 0.01,
                y - 0.04,
                f"{i + 1}. {documents[i][:MAX_LABEL]}...",
                fontsize=8,
            )

        ax.set_title("Similarité cosinus (réduction PCA 2D)")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("### 🔢 Résultats")
        for i, sim in enumerate(similarities):
            st.write(f"**Doc {i + 1}** – Similarité cosinus : `{sim:.4f}`")
