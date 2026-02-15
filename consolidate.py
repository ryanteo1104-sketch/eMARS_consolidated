"""
Consolidation and DEXPI-aided disambiguation adapted from the notebook.
"""
import pandas as pd
import re
from utils import norm_label, split_any, join_path, canonical_phrase, infer_equipment_family

# Canonical map (conservative subset from notebook)
CANON_LEAF_MAP = {
    "flange leak": ("Loss of containment", "Seals / joints / flanges", "Flange leak"),
    "gasket failure": ("Loss of containment", "Seals / joints / flanges", "Gasket failure"),
    "leak": ("Loss of containment", "Generic leak", "Leak"),
    "motor failure": ("Mechanical / Rotating equipment", "Motor / driver", "Motor failure"),
    "bearing failure": ("Mechanical / Rotating equipment", "Bearings", "Bearing failure"),
    "shaft failure": ("Mechanical / Rotating equipment", "Shafts", "Shaft failure"),
}

CONTEXT_MERGE = { ("pressure raising or reducing equipment", "compressor"): "pump or compressor components" }


def consolidate_and_disambiguate(assign_df: pd.DataFrame, em_df: pd.DataFrame, id_col: str):
    inc_text_map = dict(zip(em_df[id_col].tolist(), em_df["_emars_text_"].tolist()))
    out = []
    for _, r in assign_df.iterrows():
        path = r["Final_Category_Path"]
        inc_id = r["Accident ID"]
        if path == "UNCAT":
            out.append({**r, "Consolidated_Path": "UNCAT"})
            continue
        parts = split_any(path)
        leaf = parts[-1] if parts else path
        leaf_n = norm_label(leaf)
        if leaf_n in CANON_LEAF_MAP:
            p0, p1, pleaf = CANON_LEAF_MAP[leaf_n]
            cons_parts = [p0, p1, pleaf]
        else:
            cons_parts = parts
        if len(cons_parts) >= 2:
            parent_seg = norm_label(cons_parts[-2])
            child_seg = norm_label(cons_parts[-1])
            key = (parent_seg, child_seg)
            if key in CONTEXT_MERGE:
                cons_parts[-1] = CONTEXT_MERGE[key]
        # special mapping
        if norm_label(leaf) == "combustion/explosion causes overpressure":
            cons_parts = ["Process deviation", "Pressure deviation", "Internal overpressure (gas)", "Combustion/explosion causes overpressure"]

        # DEXPI-aided disambiguation
        cons_leaf_n = norm_label(cons_parts[-1]) if cons_parts else ""
        if cons_leaf_n in ("shaft failure", "bearing failure"):
            fam = infer_equipment_family(inc_text_map.get(inc_id, ""))
            fam = fam or "Unknown equipment"
            cons_parts = ["Mechanical / Rotating equipment", fam, cons_parts[-1].title()]

        cons_parts = [canonical_phrase(p) for p in cons_parts]
        out.append({**r, "Consolidated_Path": join_path(cons_parts)})
    return pd.DataFrame(out)
