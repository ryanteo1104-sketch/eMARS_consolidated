"""
Embedding helpers using SentenceTransformer (optional).
"""
import os
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def load_model(model_name: str):
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is not installed. See requirements.txt")
    model = SentenceTransformer(model_name)
    return model


def encode_all(model, tax_paths, em_texts, normalize=True, show_progress_bar=True):
    tax_emb = model.encode(tax_paths, normalize_embeddings=normalize, show_progress_bar=show_progress_bar)
    em_emb = model.encode(em_texts, normalize_embeddings=normalize, show_progress_bar=show_progress_bar)
    return np.asarray(tax_emb), np.asarray(em_emb)
