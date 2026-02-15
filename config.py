"""
Configuration constants extracted from the notebook.
"""
import os

# --- Paths (edit if needed) ---
EMARS_XLSX = os.environ.get("EMARS_XLSX", "eMARS_Export_01-12-2025.xlsx")
TAXON_XLSX = os.environ.get("TAXON_XLSX", "Taxonomy_DEXPI_Hierarchical.xlsx")

# --- Embedding model (SentenceTransformers) ---
MODEL_NAME = os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# --- Multi-label settings ---
TOPK = 3
THRESH = 0.35
MARGIN_KEEP = 0.02
UNCAT_IF_MAX_BELOW = 0.30

# --- Plot settings ---
TOPN_BAR = None
FIG_DPI = 200

# --- Hierarchical classification settings ---
PARENT_MIN_SIM = 0.34
PARENT_MARGIN = 0.025
TOPK_CHILD = 2
CHILD_MIN_SIM = 0.40
CHILD_MARGIN = 0.015

# --- Child overload control for tree ---
MIN_CHILD_SUPPORT = 4
MAX_CHILD_PER_PARENT = 25

# --- Evidence-lock settings ---
EVIDENCE_MATCH_MODE = os.environ.get("EVIDENCE_MATCH_MODE", "normalized")
EVIDENCE_COVERAGE_MIN = 0.25
EVIDENCE_MIN_TERM_LEN = 4
EVIDENCE_USE_KEYWORDS_ORIGINAL = True
EVIDENCE_MAX_TERMS_PER_PATH = 2
EVIDENCE_MIN_MATCHED_TERMS = 1
EVIDENCE_FALLBACK_COSINE = 0.50

# Missing-words guard
USE_MISSING_WORD_GUARD = True
MISSING_TERMS_CSV = "eMARS_missing_terms_full_description_SELECTED_TRUE.csv"
MISSING_GUARD_MIN_INSTANCES = 6
MISSING_GUARD_MAX_TERMS_PER_PATH = 2
MISSING_GUARD_USE_JSON_IF_AVAILABLE = True
MISSING_GUARD_JSON = "critical_terms_guard_by_path_from_missing_words.json"

# Parent backoff
ENABLE_PARENT_BACKOFF = True
PARENT_BACKOFF_COVERAGE_MIN = 0.12
PARENT_BACKOFF_MIN_MATCHED_TERMS = 1
PARENT_BACKOFF_MIN_COSINE = 0.43

# Depth render
ENABLE_DEPTH_AWARE_RENDER = True
MAX_DEPTH_RENDER = 3
MIN_LEAF_SUPPORT_RENDER = 3
DEPTH_CAP_LABEL = "Other (depth-capped)"
LOW_SUPPORT_LABEL = "Other (low support)"

# Missed-opportunity patch toggles
ENABLE_MISSED_OPPORTUNITY_PATCH = True
ENABLE_RUNAWAY_PARENT_PATCH = True
RUNAWAY_PARENT_MIN_COSINE = 0.45

# Auto synonyms
ENABLE_AUTO_SYNONYM_EXPANSION = True
AUTO_SYNONYM_MAX_VARIANTS_PER_TERM = 12
