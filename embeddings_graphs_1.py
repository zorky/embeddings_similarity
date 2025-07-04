from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Texte de la requête et des documents
query_text = "l’intelligence artificielle dans l’éducation"
documents = [
    "les robots dans les écoles",
    "les voitures autonomes sur les routes",
    "l’apprentissage automatique et les étudiants",
    "le réchauffement climatique et les océans",
    "l’enseignement assisté par IA en classe"
]

# Chargement du modèle d'embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encodage des textes
query_vec = model.encode([query_text])
doc_vecs = model.encode(documents)

# Calcul des similarités cosinus
similarities = cosine_similarity(doc_vecs, query_vec).flatten()

# Réduction PCA à 2D pour visualisation
pca = PCA(n_components=2)
reduced = pca.fit_transform(np.vstack([query_vec, doc_vecs]))
query_2d, docs_2d = reduced[0], reduced[1:]

# Visualisation
plt.figure(figsize=(8, 6))
plt.scatter(docs_2d[:, 0], docs_2d[:, 1], c='blue', label='Documents')
plt.scatter(query_2d[0], query_2d[1], c='red', label='Requête', marker='X', s=100)

# Affichage des étiquettes
for i, (x, y) in enumerate(docs_2d):
    plt.text(x + 0.01, y + 0.01, f"{similarities[i]:.2f}", fontsize=9)
    plt.text(x + 0.01, y - 0.04, f"Doc {i+1}", fontsize=9)

plt.title("Similarité cosinus entre la requête et les documents (réduction 2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
