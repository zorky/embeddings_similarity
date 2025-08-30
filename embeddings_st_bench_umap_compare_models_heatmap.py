import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import umap
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import io

# from embeddings_st_bench_pca import MODEL_LIST
from referentiels.models import embeddings_models

# 📌 Liste de modèles disponibles
MODEL_LIST = embeddings_models

DEFAULT_TEXTS = [
    "Le soleil brille aujourd'hui sur Paris.",
    "Les voitures électriques deviennent populaires.",
    "La cuisine italienne est délicieuse.",
    "Les pandas vivent principalement en Chine.",
    "L'intelligence artificielle transforme le monde.",
]
DEFAULT_QUESTION = "Quel est l'impact de la technologie sur la société ?"

st.title("📊 Comparaison des modèles d'embeddings")

selected_models = st.multiselect(
    "Sélectionne les modèles à comparer :", MODEL_LIST, default=MODEL_LIST[:2]
)
texts = [st.text_input(f"Texte {i + 1}", DEFAULT_TEXTS[i]) for i in range(5)]
question = st.text_input("Question", DEFAULT_QUESTION)
threshold = st.slider("Seuil de similarité cosinus", 0.0, 1.0, 0.2)


@st.cache_resource
def load_model(name):
    return SentenceTransformer(name)


import streamlit as st


def _caculate_similarity(embeddings):
    """Calcule la similarité cosinus entre les embeddings d'une question et ceux des textes."""
    embeddings_texts = embeddings[:-1]  # les textes
    embeddings_question = embeddings[-1]  # la question
    similarities = cosine_similarity([embeddings_question], embeddings_texts)[0]

    return similarities


if st.button("Comparer les modèles"):
    # --- Calcul des embeddings textes / question et similarité cosinus, et projection UMAP
    all_sentences = texts + [question]
    all_data = []
    embeddings_models = []

    for model_name in selected_models:
        model = load_model(model_name)
        embeddings = model.encode(all_sentences)

        similarities = _caculate_similarity(embeddings)

        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)

        for i in range(len(all_sentences)):
            x = reduced[i][0]
            y = reduced[i][1]
            is_question = i == len(texts)
            all_data.append(
                {
                    "x": x,
                    "y": y,
                    "texte": f"{all_sentences[i]} ({x:.2f}, {y:.2f})",
                    "type": f"Question" if is_question else f"Texte {i + 1}",
                    "similarité": 1.0 if is_question else similarities[i],
                    "position": f"({x:.2f}, {y:.2f})",
                    "modèle": model_name,
                    "symbole": "star" if is_question else "circle",
                    # "border_color": "red" if is_question else "rgba(0,0,0,0)"
                }
            )

    # --- Graphe points des résultats
    df = pd.DataFrame(all_data)
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="modèle",
        # symbol="type",
        symbol="symbole",
        size=df["similarité"].apply(lambda s: 12 if s >= threshold else 6),
        hover_data={
            "texte": True,
            "similarité": True,
            "modèle": True,
            "position": True,
            "x": False,
            "y": False,
        },
        title="Comparaison des embeddings selon le modèle",
    )
    for _, row in df[df["type"] == "Question"].iterrows():
        fig.add_annotation(
            x=row["x"],
            y=row["y"],
            text=f"📍 Q ({row['modèle']})",
            showarrow=True,
            arrowhead=2,
            ax=20,
            ay=-30,
            font=dict(color="black", size=12),
            bgcolor="lightyellow",
            bordercolor="black",
        )
    st.plotly_chart(fig, use_container_width=True)

    # --- Construction de la matrice de similarité ---
    heatmap_data = {}
    embedding_question = None  # pour affichage plus bas

    for model_name in selected_models:
        model = load_model(model_name)
        embeddings = model.encode(all_sentences)
        similarities = _caculate_similarity(embeddings)
        heatmap_data[model_name] = similarities

    # Création du DataFrame pour la heatmap avec les textes réels comme index
    df_heatmap = pd.DataFrame(heatmap_data, index=texts)

    # Filtrage selon le seuil
    df_heatmap_filtered = df_heatmap[df_heatmap.max(axis=1) >= threshold]

    # --- Affichage de la position et de l'embedding de la question ---
    st.write(
        f"📍 Position UMAP de la question : x = {reduced[-1][0]:.2f}, y = {reduced[-1][1]:.2f}"
    )

    with st.expander("📐 Embedding brut de la question (N dimensions)"):
        embedding_df = pd.DataFrame(
            [embedding_question],
            columns=[f"Dim {i + 1}" for i in range(len(embedding_question))],
        )
        st.dataframe(embedding_df)

    # --- Affichage de la heatmap ---
    st.subheader("🧮 Matrice de similarité cosinus (heatmap)")
    if not df_heatmap_filtered.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_heatmap_filtered, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
        st.pyplot(fig)

        # Format : YYYY-MM-DD-HHhMM
        timestamp = datetime.now().strftime("%Y-%m-%d-%Hh%M")
        filename_csv = f"similarite_cosinus-{timestamp}.csv"
        filename_excel = f"similarite_cosinus-{timestamp}.xlsx"

        csv_data = df_heatmap_filtered.to_csv(index=True).encode("utf-8")

        st.download_button(
            label="📥 Télécharger la matrice en CSV",
            data=csv_data,
            file_name=filename_csv,
            mime="text/csv",
        )

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df_heatmap_filtered.to_excel(writer, sheet_name="Similarité", index=True)

        st.download_button(
            label="📥 Télécharger la matrice en Excel",
            data=excel_buffer.getvalue(),
            file_name=filename_excel,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.warning("Aucun texte ne dépasse le seuil de similarité sélectionné.")
