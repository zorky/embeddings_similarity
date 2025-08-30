import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import fitz  # PyMuPDF
import pandas as pd
import io

import os

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Similarité Cosinus - Fichiers", layout="wide")
st.title("📁 Analyse de similarité entre une requête et des fichiers")

# Choix du modèle
model_choice = st.selectbox(
    "🧠 Choisissez un modèle d'embedding",
    [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2",
        "multi-qa-MiniLM-L6-cos-v1",
    ],
)

# Entrée de la requête
query_text = st.text_input(
    "🔍 Requête (texte ou question)", "l’intelligence artificielle dans l’éducation"
)

# Upload de fichiers
uploaded_files = st.file_uploader(
    "📂 Chargez vos fichiers (.txt ou .pdf)",
    type=["txt", "pdf"],
    accept_multiple_files=True,
)

# Enregistrement local des fichiers uploadés
# for file in uploaded_files:
#     save_path = os.path.join(UPLOAD_DIR, file.name)
#     with open(save_path, "wb") as f:
#         f.write(file.read())

# Seuil
threshold = st.slider("🎚️ Seuil de similarité minimale", 0.0, 1.0, 0.3, step=0.05)


# Fonction d’extraction texte
def extract_text(file):
    if file.type == "text/plain":
        return file.read().decode("utf-8", errors="ignore")
    elif file.type == "application/pdf":
        text = ""
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    return ""


# Bouton de traitement
if st.button("🧠 Analyser les similarités") and uploaded_files:
    with st.spinner("Encodage et calcul des similarités..."):
        model = SentenceTransformer(model_choice)
        query_vec = model.encode([query_text])

        file_texts = []
        file_names = []
        for file in uploaded_files:
            content = extract_text(file)
            if content.strip():
                file_texts.append(content)
                file_names.append(file.name)

        if not file_texts:
            st.warning("Aucun texte lisible extrait des fichiers.")
        else:
            file_vecs = model.encode(file_texts)
            sims = cosine_similarity(file_vecs, query_vec).flatten()

            ranked = sorted(
                zip(file_names, file_texts, sims, file_vecs),
                key=lambda x: x[2],
                reverse=True,
            )
            filtered = [
                (name, text, sim, vec)
                for name, text, sim, vec in ranked
                if sim >= threshold
            ]

            if not filtered:
                st.warning("Aucun fichier ne dépasse le seuil.")
            else:
                # Projection PCA
                names, texts, sims_filtered, vecs_filtered = zip(*filtered)
                reduced = PCA(n_components=2).fit_transform(
                    np.vstack([query_vec, vecs_filtered])
                )
                query_2d, docs_2d = reduced[0], reduced[1:]

                fig, ax = plt.subplots(figsize=(9, 6))
                ax.scatter(docs_2d[:, 0], docs_2d[:, 1], c="blue", label="Fichiers")
                ax.scatter(
                    query_2d[0],
                    query_2d[1],
                    c="red",
                    marker="X",
                    s=100,
                    label="Requête",
                )

                for i, (x, y) in enumerate(docs_2d):
                    ax.text(x + 0.01, y + 0.01, f"{sims_filtered[i]:.2f}", fontsize=9)
                    ax.text(x + 0.01, y - 0.04, f"{i + 1}. {names[i][:20]}", fontsize=8)

                ax.set_title("Projection PCA 2D - Requête vs Fichiers")
                ax.set_xlabel("PCA 1")
                ax.set_ylabel("PCA 2")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)

                st.markdown("### 📊 Résultats triés (fichiers au-dessus du seuil)")

                results = []
                for i, (name, sim, text) in enumerate(
                    zip(names, sims_filtered, texts), 1
                ):
                    st.markdown(f"**{i}. `{name}`** – Similarité : `{sim:.4f}`")
                    st.markdown(f"> _Extrait_ : {text.strip()[:300]}…")
                    results.append(
                        {
                            "Rang": i,
                            "Nom du fichier": name,
                            "Similarité": round(sim, 4),
                            "Extrait": text.strip()[:500],
                        }
                    )

                # Export CSV
                df = pd.DataFrame(results)
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue().encode("utf-8")

                st.download_button(
                    label="📥 Télécharger les résultats en CSV",
                    data=csv_data,
                    file_name="resultats_similarite.csv",
                    mime="text/csv",
                )
