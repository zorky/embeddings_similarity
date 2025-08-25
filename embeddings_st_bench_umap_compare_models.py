import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import umap
import plotly.express as px
import pandas as pd

# 📌 Liste de modèles disponibles
MODEL_LIST = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "paraphrase-MiniLM-L3-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "distiluse-base-multilingual-cased-v2"
]

DEFAULT_TEXTS = [
    "Le soleil brille aujourd'hui sur Paris.",
    "Les voitures électriques deviennent populaires.",
    "La cuisine italienne est délicieuse.",
    "Les pandas vivent principalement en Chine.",
    "L'intelligence artificielle transforme le monde."
]
DEFAULT_QUESTION = "Quel est l'impact de la technologie sur la société ?"

st.title("📊 Comparaison des modèles d'embeddings")

selected_models = st.multiselect("Sélectionne les modèles à comparer :", MODEL_LIST, default=MODEL_LIST[:2])
texts = [st.text_input(f"Texte {i+1}", DEFAULT_TEXTS[i]) for i in range(5)]
question = st.text_input("Question", DEFAULT_QUESTION)
threshold = st.slider("Seuil de similarité cosinus", 0.0, 1.0, 0.2)

@st.cache_resource
def load_model(name):
    return SentenceTransformer(name)

if st.button("Comparer les modèles"):
    all_sentences = texts + [question]
    all_data = []

    for model_name in selected_models:
        model = load_model(model_name)
        embeddings = model.encode(all_sentences)
        similarities = cosine_similarity([embeddings[-1]], embeddings[:-1])[0]
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)

        for i in range(len(all_sentences)):
            all_data.append({
                "x": reduced[i][0],
                "y": reduced[i][1],
                "texte": all_sentences[i],
                "type": "Question" if i == len(texts) else f"Texte {i+1}",
                "similarité": 1.0 if i == len(texts) else similarities[i],
                "modèle": model_name
            })

    df = pd.DataFrame(all_data)
    fig = px.scatter(
        df,
        x="x", y="y",
        color="modèle",
        symbol="type",
        size=df["similarité"].apply(lambda s: 12 if s >= threshold else 6),
        hover_data={"texte": True, "similarité": True, "modèle": True, "x": False, "y": False},
        title="Comparaison des embeddings selon le modèle"
    )
    st.plotly_chart(fig, use_container_width=True)

