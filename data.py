"""
Data loading utilities: load eMARS and taxonomy and produce canonical taxonomy paths.
"""
import os
import pandas as pd
from utils import norm_text, pick_col, pick_tax_col


def load_emars(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    em = pd.read_excel(path)
    # auto-detect text columns
    id_col = pick_col(em.columns, ["accident id", "incident id", "id"]) or em.columns[0]
    title_col = pick_col(em.columns, ["title", "accident title", "incident title", "event title"])
    desc_col = pick_col(em.columns, ["description", "accident description", "incident description", "summary", "narrative"]) or title_col

    em["_title_"] = em[title_col].map(norm_text) if title_col in em.columns else ""
    em["_desc_"] = em[desc_col].map(norm_text) if desc_col in em.columns else ""
    em["_emars_text_"] = (em["_title_"] + " " + em["_desc_"]).str.strip()
    return em, id_col, title_col, desc_col


def load_taxonomy(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    tx = pd.read_excel(path)
    tax_path_col = pick_tax_col(tx)
    tx[tax_path_col] = tx[tax_path_col].fillna("").astype(str).map(lambda s: s.strip())
    tax_paths = [p for p in tx[tax_path_col].tolist() if p]
    tax_paths = pd.Series(tax_paths).drop_duplicates().tolist()
    return tx, tax_path_col, tax_paths
