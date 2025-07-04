# Image de base avec Python
FROM python:3.10-slim

# Dépendances système pour PyMuPDF (fitz) et matplotlib
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Définir le dossier de travail
WORKDIR /app

RUN mkdir -p /app/uploaded_files

# Copier les fichiers nécessaires
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copier l'application
COPY . ./

# Exposer le port de Streamlit
EXPOSE 8501

# Lancer l'application
#CMD ["streamlit", "run", "embeddings_streamlit_4.py", "--server.port=8501", "--server.address=0.0.0.0"]