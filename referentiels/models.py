embeddings_models = [
    "all-MiniLM-L6-v2",  # 384 D rapide et compact https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    "all-mpnet-base-v2",  # 768 D plus précis, plus lourd
    "paraphrase-MiniLM-L6-v2",  # 384 D orienté paraphrases
    "multi-qa-MiniLM-L6-cos-v1",  # 384 D recherche sémantique
    "dangvantuan/sentence-camembert-base",  # 768 D https://huggingface.co/dangvantuan/sentence-camembert-base
    # "dangvantuan/sentence-camembert-large", # 1024 D https://huggingface.co/dangvantuan/sentence-camembert-large
    "distiluse-base-multilingual-cased-v2",  # 512 D https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2
    # "camembert-base"               # pas fait pour de la recherche sémantique cos 768 D performe en français https://huggingface.co/almanach/camembert-base
]
