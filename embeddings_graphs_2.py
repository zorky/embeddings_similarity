from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

MAX_LABEL=60

# Texte de la requête et documents
query_text = "l’intelligence artificielle dans l’éducation"
documents = [
    "les robots dans les écoles",
    "les voitures autonomes sur les routes",
    "l’apprentissage automatique et les étudiants",
    "le réchauffement climatique et les océans",
    "l’enseignement assisté par IA en classe"
]

# Chargement du modèle d'embedding
model = SentenceTransformer("all-MiniLM-L6-v2")  # petit, rapide et efficace

# Encodage des textes
query_vec = model.encode([query_text])
doc_vecs = model.encode(documents)

# Calcul des similarités cosinus
similarities = cosine_similarity(doc_vecs, query_vec).flatten()

# Réduction en 2D pour affichage
pca = PCA(n_components=2)
reduced = pca.fit_transform(np.vstack([query_vec, doc_vecs]))
query_2d, docs_2d = reduced[0], reduced[1:]

# Affichage
plt.figure(figsize=(9, 6))
plt.scatter(docs_2d[:, 0], docs_2d[:, 1], c='blue', label='Documents')
plt.scatter(query_2d[0], query_2d[1], c='red', marker='X', s=100, label='Requête')

# Affichage des noms et similarités
for i, (x, y) in enumerate(docs_2d):
    plt.text(x + 0.01, y + 0.01, f"{similarities[i]:.2f}", fontsize=9)
    plt.text(x + 0.01, y - 0.04, f"{i+1}. {documents[i][:MAX_LABEL]}...", fontsize=8)

plt.title("Similarité cosinus entre la requête et les documents (réduction PCA 2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
