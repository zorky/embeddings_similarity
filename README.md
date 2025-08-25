# Environnement virtuel avec uv

```bash
$ uv pip install -r requirements.txt
```

activation du venv

```bash
$ source .venv/Scripts/activate
```

# Scripts 

Lancer le script via uv

```bash
$ uv run embeddings_graphs_1.py
```

ou directement avec python dans l'environnement virtuel

```bash
$ python embeddings_graphs_1.py
```

# Streamlit

Lancer l'application Streamlit

```bash
$ streamlit run embeddings_streamlit_1.py
```

# Streamlit par docker

```bash
$ docker-compose up
```

dans le docker-compose.yml, modifier le fichier Streamlit souhaité, section environment et variable STREAMLIT_APP

```yaml
    environment:
      - STREAMLIT_APP=embeddings_streamlit_5.py
```

valeur possible : embeddings_streamlit_1 à 5.py

## UV

```bash
$ uv venv
$ uv init
# $ uv pip sync requirements.txt
# $ uv lock # n'ajoute pas les packages re requirements.txt
$ uv add -r requirements.txt # https://pydevtools.com/handbook/how-to/migrate-requirements.txt/
$ uv add sentence-transformers scikit-learn matplotlib streamlit PyMuPDF pandas umap-learn
```
