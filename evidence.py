"""
Evidence-lock filtering and evaluation adapted from the notebook.
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from utils import _clean_desc_text
from config import (
    EVIDENCE_MATCH_MODE, EVIDENCE_COVERAGE_MIN, EVIDENCE_MIN_MATCHED_TERMS,
    EVIDENCE_FALLBACK_COSINE, EVIDENCE_MAX_TERMS_PER_PATH,
    USE_MISSING_WORD_GUARD, MISSING_TERMS_CSV, MISSING_GUARD_MIN_INSTANCES,
    MISSING_GUARD_MAX_TERMS_PER_PATH, ENABLE_PARENT_BACKOFF,
    PARENT_BACKOFF_COVERAGE_MIN, PARENT_BACKOFF_MIN_MATCHED_TERMS, PARENT_BACKOFF_MIN_COSINE,
    ENABLE_RUNAWAY_PARENT_PATCH, RUNAWAY_PARENT_MIN_COSINE
)

# Minimal alias maps (kept small for readability)
TOKEN_ALIAS_MAP = {
    "leakage": ["leak", "leaking", "release"],
    "explosion": ["explode", "blast"],
}

PHRASE_ALIAS_MAP = {
    "oil spill": ["oil leak", "release of oil"]
}

SYMBOL_ALIAS_MAP = {"cl2": ["chlorine"]}


def _normalize_term(w: str, mode: str = EVIDENCE_MATCH_MODE) -> str:
    w = str(w or "").lower().strip()
    w = w.replace("_", " ").replace("-", " ")
    w = re.sub(r"\s+", " ", w).strip()
    if mode == "normalized":
        w = w.replace("vapour", "vapor")
        toks = [t for t in re.findall(r"[a-z0-9]+", w)]
        return " ".join(toks)
    return w


def _word_forms(w: str):
    w = _normalize_term(w)
    if not w: return []
    out = {w}
    if len(w) > 3:
        out.add(w.rstrip('s'))
    return list(out)


def _term_candidates(term: str, path: str = ""):
    nt = _normalize_term(term)
    cands = [nt] if nt else []
    if nt in PHRASE_ALIAS_MAP:
        cands.extend(PHRASE_ALIAS_MAP[nt])
    if nt in SYMBOL_ALIAS_MAP:
        cands.extend(SYMBOL_ALIAS_MAP[nt])
    toks = re.findall(r"[a-z0-9]+", nt)
    for t in toks:
        for a in TOKEN_ALIAS_MAP.get(t, []):
            cands.append(a)
    return list(dict.fromkeys([_normalize_term(c) for c in cands if c]))


def _term_in_desc(term: str, desc_norm: str, path: str = "") -> bool:
    if not term: return False
    if term in desc_norm: return True
    for cand in _term_candidates(term, path=path):
        if cand in desc_norm: return True
        if re.search(r"\b" + re.escape(cand) + r"\b", desc_norm):
            return True
    return False


def build_taxonomy_term_set(taxon_xlsx_path: str, use_keywords_original=True):
    terms = set()
    try:
        xls = pd.ExcelFile(taxon_xlsx_path)
        if use_keywords_original and "Keywords_Original" in xls.sheet_names:
            kw = pd.read_excel(xls, sheet_name="Keywords_Original")
            kw_col = next((c for c in kw.columns if "keyword" in c.lower()), None)
            if kw_col:
                for k in kw[kw_col].dropna().astype(str):
                    nk = _normalize_term(k)
                    if nk:
                        terms.add(nk)
        # Combined_Taxonomy sheet fallback
        if "Combined_Taxonomy" in xls.sheet_names:
            ct = pd.read_excel(xls, sheet_name="Combined_Taxonomy")
            for col in [c for c in ct.columns if c.lower() in ("path","name")]:
                for v in ct[col].dropna().astype(str):
                    nv = _normalize_term(v)
                    if nv:
                        terms.add(nv)
    except Exception:
        pass
    return terms


def _split_path_any(path: str):
    p = str(path)
    if ">" in p:
        parts = [x.strip() for x in p.split(">") if x.strip()]
    elif "/" in p:
        parts = [x.strip() for x in p.split("/") if x.strip()]
    else:
        parts = [p.strip()] if p.strip() else []
    return parts


def _parent_path(path: str):
    parts = _split_path_any(path)
    if len(parts) <= 1: return None
    return " > ".join(parts[:-1])


def prepare_expected_terms_cache(assign_df: pd.DataFrame, taxonomy_xlsx: str):
    taxonomy_terms = build_taxonomy_term_set(taxonomy_xlsx)
    PATH_TERM_CACHE = {}
    all_paths = assign_df["Consolidated_Path"].dropna().unique().tolist()
    for p in list(all_paths):
        if p != "UNCAT":
            parts = _split_path_any(p)
            leaf = parts[-1] if parts else ""
            candidates = []
            leaf_n = _normalize_term(leaf)
            for t in re.findall(r"[a-z0-9]+", leaf_n):
                if len(t) >= 4:
                    candidates.append(t)
            # basic filtering
            out = []
            seen = set()
            for t in candidates:
                if t in seen: continue
                seen.add(t)
                if not taxonomy_terms or t in taxonomy_terms:
                    out.append(t)
            PATH_TERM_CACHE[p] = out[:EVIDENCE_MAX_TERMS_PER_PATH]
    return PATH_TERM_CACHE


def apply_evidence_gate(assign_df: pd.DataFrame, em_df: pd.DataFrame, path_term_cache: dict):
    desc_series = em_df.set_index(em_df.columns[0])[em_df.columns[1]] if False else None
    # build normalized desc map
    desc_map_raw = dict(zip(em_df.iloc[:,0].tolist(), em_df.iloc[:, em_df.columns.get_loc('_emars_text_') if '_emars_text_' in em_df.columns else 1].astype(str).tolist()))
    desc_map_norm = {k: _normalize_term(_clean_desc_text(v)) for k, v in desc_map_raw.items()}

    ev_rows = []
    for _, r in assign_df.iterrows():
        inc_id = r["Accident ID"]
        path = str(r.get("Consolidated_Path", r.get("Final_Category_Path", "UNCAT")))
        cosv = float(r.get("Cosine", 0.0))
        if path == "UNCAT":
            ev_rows.append({**r, "Evidence_Pass": True, "Evidence_Reason": "pre_uncat"})
            continue
        expected_terms = path_term_cache.get(path, [])
        desc_n = desc_map_norm.get(inc_id, "")
        matched = [t for t in expected_terms if _term_in_desc(t, desc_n, path=path)]
        exp_n = len(expected_terms)
        cov = (len(matched) / exp_n) if exp_n > 0 else np.nan
        if exp_n == 0:
            ev_pass = float(cosv) >= float(EVIDENCE_FALLBACK_COSINE)
            reason = "fallback_cosine" if ev_pass else "reject_fallback_cosine"
        else:
            ev_pass = (len(matched) >= int(EVIDENCE_MIN_MATCHED_TERMS)) and (float(cov) >= float(EVIDENCE_COVERAGE_MIN))
            reason = "pass_expected_terms" if ev_pass else "reject_expected_terms"
        # parent backoff
        chosen_path = path
        chosen = {"expected_terms": expected_terms, "matched": matched, "cov": cov, "ev_pass": ev_pass}
        if (not ev_pass) and ENABLE_PARENT_BACKOFF:
            pp = _parent_path(path)
            if pp:
                p_expected = path_term_cache.get(pp, [])
                p_matched = [t for t in p_expected if _term_in_desc(t, desc_n, path=pp)]
                p_exp_n = len(p_expected)
                p_cov = (len(p_matched) / p_exp_n) if p_exp_n > 0 else np.nan
                p_pass = ((p_exp_n > 0 and len(p_matched) >= int(PARENT_BACKOFF_MIN_MATCHED_TERMS) and float(p_cov) >= float(PARENT_BACKOFF_COVERAGE_MIN))
                          or (p_exp_n == 0 and float(cosv) >= float(PARENT_BACKOFF_MIN_COSINE)))
                if p_pass:
                    chosen_path = pp
                    reason = "accept_parent_backoff"
                    chosen = {"expected_terms": p_expected, "matched": p_matched, "cov": p_cov, "ev_pass": p_pass}
        ev_rows.append({
            **r,
            "Final_Category_Path": chosen_path,
            "Consolidated_Path": chosen_path,
            "Evidence_Expected_Terms": " | ".join(chosen.get("expected_terms", [])),
            "Evidence_Matched_Terms": " | ".join(chosen.get("matched", [])),
            "Evidence_Matched_Count": len(chosen.get("matched", [])),
            "Evidence_Expected_Count": chosen.get("exp_n", len(chosen.get("expected_terms", []))),
            "Evidence_Coverage": chosen.get("cov", 0),
            "Evidence_Pass": bool(chosen.get("ev_pass", False)),
            "Evidence_Reason": reason,
            "Description_Full": desc_map_raw.get(inc_id, ""),
        })
    ev = pd.DataFrame(ev_rows)
    fail_mask = (ev["Consolidated_Path"] != "UNCAT") & (ev["Evidence_Pass"] == False)
    
    # Downgrade them to UNCAT instead of dropping them
    ev.loc[fail_mask, "Consolidated_Path"] = "UNCAT"
    ev.loc[fail_mask, "Final_Category_Path"] = "UNCAT"
    ev.loc[fail_mask, "Evidence_Reason"] = ev.loc[fail_mask, "Evidence_Reason"] + " -> downgraded_to_uncat"
    # Keep ALL rows
    supported = ev.copy()
    # filter supported
    supported = ev[(ev["Consolidated_Path"] == "UNCAT") | (ev["Evidence_Pass"] == True)].copy()
    supported = supported.sort_values(["Accident ID", "Cosine"], ascending=[True, False]).copy()
    supported["Rank"] = supported.groupby("Accident ID").cumcount() + 1
    return supported
