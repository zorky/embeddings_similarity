import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Similarité Cosinus", layout="wide")
st.title("🔍 Similarité Cosinus entre Requête et Documents")

# 🧠 Sélection du modèle
model_choice = st.selectbox(
    "🧠 Choisissez un modèle d'embedding",
    [
        "all-MiniLM-L6-v2",              # rapide et compact
        "all-mpnet-base-v2",             # plus précis, plus lourd
        "paraphrase-MiniLM-L6-v2",       # orienté paraphrases
        "multi-qa-MiniLM-L6-cos-v1"      # pour recherche sémantique
    ],
    index=0
)

# 📌 Entrée de la requête
query_text = st.text_input("📝 Requête :", "l’intelligence artificielle dans l’éducation")

# ✏️ Entrée des documents
default_docs = [
    "les robots dans les écoles",
    "les voitures autonomes sur les routes",
    "l’apprentissage automatique et les étudiants",
    "le réchauffement climatique et les océans",
    "l’enseignement assisté par IA en classe"
]

st.markdown("✏️ Modifiez les documents à comparer :")
documents = []
cols = st.columns(2)
for i in range(len(default_docs)):
    doc = cols[i % 2].text_input(f"Document {i+1}", value=default_docs[i])
    documents.append(doc)

# 🎚️ Seuil de filtrage
threshold = st.slider("🔎 Seuil de similarité minimum à afficher", 0.0, 1.0, 0.2, step=0.05)

# 🔘 Lancer l’analyse
if st.button("🔍 Calculer et Afficher"):

    with st.spinner(f"Encodage avec le modèle : `{model_choice}` ..."):
        model = SentenceTransformer(model_choice)
        query_vec = model.encode([query_text])
        doc_vecs = model.encode(documents)
        similarities = cosine_similarity(doc_vecs, query_vec).flatten()

        # Tri des documents par similarité décroissante
        ranked = sorted(zip(documents, similarities, doc_vecs), key=lambda x: x[1], reverse=True)
        filtered = [(doc, sim, vec) for doc, sim, vec in ranked if sim >= threshold]

        if not filtered:
            st.warning("Aucun document n'a dépassé le seuil de similarité.")
        else:
            # PCA pour affichage
            filtered_docs, filtered_sims, filtered_vecs = zip(*filtered)
            reduced = PCA(n_components=2).fit_transform(np.vstack([query_vec, filtered_vecs]))
            query_2d, docs_2d = reduced[0], reduced[1:]

            # Tracé
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.scatter(docs_2d[:, 0], docs_2d[:, 1], c='blue', label='Documents')
            ax.scatter(query_2d[0], query_2d[1], c='red', marker='X', s=100, label='Requête')

            for i, (x, y) in enumerate(docs_2d):
                ax.text(x + 0.01, y + 0.01, f"{filtered_sims[i]:.2f}", fontsize=9)
                ax.text(x + 0.01, y - 0.04, f"{i + 1}. {documents[i]}", fontsize=7)

            ax.set_title("Projection PCA 2D avec seuil de similarité")
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            # Tableau trié
            st.markdown("### 📊 Documents au-dessus du seuil")
            for i, (doc, sim) in enumerate(zip(filtered_docs, filtered_sims), 1):
                st.markdown(f"**Doc {i}** – Similarité : `{sim:.4f}`  \n📄 _{doc}_")

