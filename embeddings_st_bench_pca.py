import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 📌 Liste de modèles disponibles
MODEL_LIST = [
    "all-MiniLM-L6-v2",
    "paraphrase-MiniLM-L3-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "distiluse-base-multilingual-cased-v2",
]

# 📋 Textes et question par défaut
DEFAULT_TEXTS = [
    "Le soleil brille aujourd'hui sur Paris.",
    "Les voitures électriques deviennent populaires.",
    "La cuisine italienne est délicieuse.",
    "Les pandas vivent principalement en Chine.",
    "L'intelligence artificielle transforme le monde.",
]
DEFAULT_QUESTION = "Quel est l'impact de la technologie sur la société ?"

# 🧠 Interface Streamlit
st.title("🧬 Visualisation d'Embeddings avec Sentence Transformers")

model_name = st.selectbox("Choisis un modèle d'embedding :", MODEL_LIST)
texts = [st.text_input(f"Texte {i + 1}", DEFAULT_TEXTS[i]) for i in range(5)]
question = st.text_input("Question", DEFAULT_QUESTION)
threshold = st.slider("Seuil de similarité cosinus", 0.0, 1.0, 0.4)


# 🔍 Chargement du modèle
@st.cache_resource
def load_model(name):
    return SentenceTransformer(name)


model = load_model(model_name)

# 🔄 Calcul des embeddings
all_sentences = texts + [question]
embeddings = model.encode(all_sentences)

# 📊 Similarité cosinus
similarities = cosine_similarity([embeddings[-1]], embeddings[:-1])[0]

# 📉 Réduction dimensionnelle
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# 🎯 Affichage des résultats
if st.button("Générer le graphe 2D"):
    fig, ax = plt.subplots()
    for i, (text, sim) in enumerate(zip(texts, similarities)):
        if sim >= threshold:
            ax.scatter(*reduced_embeddings[i], label=f"Texte {i + 1} ({sim:.2f})")
            ax.annotate(f"T{i + 1}", reduced_embeddings[i])

    # Position de la question
    ax.scatter(*reduced_embeddings[-1], color="red", label="Question", marker="x")
    ax.annotate("Q", reduced_embeddings[-1])
    ax.set_title("Projection PCA des embeddings")
    ax.legend()
    st.pyplot(fig)
