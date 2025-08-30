import streamlit as st

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from referentiels.models import embeddings_models
from referentiels.docs import default_docs

from vectors_similarities import calculate_similarities, rank_documents, do_graph

st.set_page_config(page_title="Similarité Cosinus", layout="wide")
st.title("🔍 Similarité Cosinus entre Requête et Documents")

# 🧠 Sélection du modèle
model_choice = st.selectbox(
    "🧠 Choisissez un modèle d'embedding", embeddings_models, index=0
)

# 📌 Entrée de la requête
query_text = st.text_input(
    "📝 Requête :", "l’intelligence artificielle dans l’éducation"
)

st.markdown("✏️ Modifiez les documents à comparer :")
documents = []
cols = st.columns(2)
for i in range(len(default_docs)):
    doc = cols[i % 2].text_input(f"Document {i + 1}", value=default_docs[i])
    documents.append(doc)

# 🎚️ Seuil de filtrage
threshold = st.slider(
    "🔎 Seuil de similarité minimum à afficher", 0.0, 1.0, 0.4, step=0.05
)

# 🔘 Lancer l’analyse
if st.button("🔍 Calculer et Afficher"):
    with st.spinner(f"Encodage avec le modèle : `{model_choice}` ..."):
        similarities, doc_vecs, query_vec = calculate_similarities(
            model_choice, query_text=query_text, documents=documents
        )

        # Tri des documents par similarité décroissante
        filtered = rank_documents(
            documents=documents,
            doc_vecs=doc_vecs,
            similarities=similarities,
            threshold=threshold,
        )
        if not filtered:
            st.warning("Aucun document n'a dépassé le seuil de similarité.")
        else:
            fig, filtered_docs, filtered_sims = do_graph(
                filtered=filtered, documents=documents, query_vec=query_vec
            )
            st.pyplot(fig)

            # Tableau trié
            st.markdown("### 📊 Documents au-dessus du seuil")
            for i, (doc, sim) in enumerate(zip(filtered_docs, filtered_sims), 1):
                st.markdown(f"**Doc {i}** – Similarité : `{sim:.4f}`  \n📄 _{doc}_")
