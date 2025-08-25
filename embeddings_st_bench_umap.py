import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import umap
import plotly.express as px

from referentiels.models import embeddings_models

# 📌 Liste de modèles disponibles
MODEL_LIST = embeddings_models
# MODEL_LIST = [
#     "all-MiniLM-L6-v2",
#     "paraphrase-MiniLM-L3-v2",
#     "multi-qa-MiniLM-L6-cos-v1",
#     "distiluse-base-multilingual-cased-v2"
# ]

# 📋 Textes et question par défaut
DEFAULT_TEXTS = [
    "Le soleil brille aujourd'hui sur Paris.",
    "Les voitures électriques deviennent populaires.",
    "La cuisine italienne est délicieuse.",
    "Les pandas vivent principalement en Chine.",
    "L'intelligence artificielle transforme le monde."
]
DEFAULT_QUESTION = "Quel est l'impact de la technologie sur la société ?"

# 🧠 Interface Streamlit
st.title("🔍 Visualisation interactive des embeddings")

model_name = st.selectbox("Choisis un modèle :", MODEL_LIST)
texts = [st.text_input(f"Texte {i+1}", DEFAULT_TEXTS[i]) for i in range(5)]
question = st.text_input("Question", DEFAULT_QUESTION)
threshold = st.slider("Seuil de similarité cosinus", 0.0, 1.0, 0.5)

# 🔍 Chargement du modèle
@st.cache_resource
def load_model(name):
    return SentenceTransformer(name)

model = load_model(model_name)

# 🔄 Embeddings
all_sentences = texts + [question]
embeddings = model.encode(all_sentences)

# 📊 Similarité cosinus
similarities = cosine_similarity([embeddings[-1]], embeddings[:-1])[0]

# 📉 Réduction dimensionnelle avec UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
reduced = reducer.fit_transform(embeddings)

# 📍 Préparation des données pour Plotly
data = {
    "x": reduced[:, 0],
    "y": reduced[:, 1],
    "label": [f"Texte {i+1}" for i in range(5)] + ["Question"],
    "texte": texts + [question],
    "similarité": list(similarities) + [1.0],
    "type": ["Texte"] * 5 + ["Question"]
}

# 🎨 Couleurs selon similarité
color_scale = [sim if i < 5 else 1.0 for i, sim in enumerate(data["similarité"])]

# 🎯 Affichage interactif
if st.button("Afficher le graphe interactif"):
    fig = px.scatter(
        data,
        x="X", y="Y",
        color=color_scale,
        color_continuous_scale="Viridis",
        hover_name="label",
        hover_data={"texte": True, "similarité": True, "x": False, "y": False},
        symbol="type",
        size=[12 if sim >= threshold else 6 for sim in data["similarité"]],
        title="Projection UMAP des embeddings"
    )
    st.plotly_chart(fig, use_container_width=True)
