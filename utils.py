"""
Utility helpers consolidated from the notebook.
"""
import re
import html
import pandas as pd


def pick_col(cols, candidates):
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        for c_l, c in cols_l.items():
            if cand in c_l:
                return c
    return None


def norm_text(x):
    if pd.isna(x):
        return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pick_tax_col(df):
    candidates = ["final_category_path", "category_path", "path", "tax_doc", "taxonomy", "label", "name"]
    for cand in candidates:
        for c in df.columns:
            if cand in c.lower():
                return c
    # fallback: first object dtype column
    for c in df.columns:
        if df[c].dtype == "object":
            return c
    return df.columns[0]


def split_any(path: str):
    s = str(path)
    if ">" in s:
        return [p.strip() for p in s.split(">") if p.strip()]
    if "/" in s:
        return [p.strip() for p in s.split("/") if p.strip()]
    return [s.strip()] if s.strip() else []


def join_path(parts):
    return " > ".join([p for p in parts if p])


def norm_label(s: str) -> str:
    s = str(s)
    s = s.replace("\u00a0", " ")  # nbsp
    s = re.sub(r"\s+", " ", s).strip()
    s = s.lower()
    s = re.sub(r"[\(\)\[\]\{\}]", "", s)
    s = re.sub(r"[^\w\s>/\-]", "", s)  # keep word chars and separators
    s = re.sub(r"\s+", " ", s).strip()
    return s


def simple_stem_word(w: str) -> str:
    w = str(w).lower()
    if len(w) <= 4:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)]
    if w.endswith("es") and len(w) - 2 >= 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) - 1 >= 4:
        return w[:-1]
    return w


def canonical_phrase(phrase: str) -> str:
    p = str(phrase).strip()
    words = re.split(r"\s+", p)
    if len(words) == 1:
        return simple_stem_word(words[0])
    return re.sub(r"\s+", " ", p).strip().lower()


def infer_equipment_family(text: str):
    t = str(text).lower()
    EQUIP_CUES = {
        "Pump":        [r"\bpump\b", r"\bcentrifugal pump\b", r"\breciprocating pump\b", r"\brotary pump\b"],
        "Compressor":  [r"\bcompressor\b", r"\bcentrifugal compressor\b", r"\baxial compressor\b", r"\breciprocating compressor\b"],
        "Motor":       [r"\bmotor\b", r"\bac motor\b", r"\bdc motor\b", r"\belectric motor\b"],
        "Gearbox":     [r"\bgear ?box\b", r"\bgearbox\b", r"\btransmission\b", r"\bdrive train\b"],
        "Turbine":     [r"\bturbine\b", r"\bgas turbine\b", r"\bsteam turbine\b"],
        "Fan/Blower":  [r"\bfan\b", r"\bblower\b"],
        "Agitator":    [r"\bagitator\b", r"\bimpeller\b", r"\bmixer\b"],
    }
    for fam, pats in EQUIP_CUES.items():
        for pat in pats:
            if re.search(pat, t):
                return fam
    return None


def _clean_desc_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = html.unescape(str(s))
    s = s.lower().replace("\u00a0", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _fnum(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default
