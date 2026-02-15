"""
Hierarchical assignment (parent then child) adapted from the notebook.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils import split_any
import config


def assign_hierarchical(em, id_col, tax_paths, em_emb, tax_emb):
    S = cosine_similarity(em_emb, tax_emb)

    # Build parent -> taxonomy row indices
    tax_parents = []
    for p in tax_paths:
        parts = split_any(p)
        parent = parts[0] if parts else "UNCAT"
        tax_parents.append(parent)

    parent_to_idx = {}
    for j, p in enumerate(tax_parents):
        parent_to_idx.setdefault(p, []).append(j)

    rows = []
    for i, inc_id in enumerate(em[id_col].tolist()):
        sims = S[i]

        parent_scores = {p: float(np.max(sims[idxs])) for p, idxs in parent_to_idx.items()}
        parent_ranked = sorted(parent_scores.items(), key=lambda x: x[1], reverse=True)

        best_parent, best_parent_sim = parent_ranked[0]
        second_parent_sim = parent_ranked[1][1] if len(parent_ranked) > 1 else -1.0
        parent_margin = best_parent_sim - second_parent_sim
        parent_low_conf = (parent_margin < config.PARENT_MARGIN)

        if best_parent_sim < config.PARENT_MIN_SIM:
            rows.append({
                "Accident ID": inc_id,
                "Final_Category_Path": "UNCAT",
                "Cosine": best_parent_sim,
                "Rank": 1,
                "Selected": True,
                "Below_UNCAT_Threshold": True,
                "Parent": "UNCAT",
                "Parent_Sim": best_parent_sim,
                "Parent_Margin": parent_margin,
                "Parent_LowConf": True
            })
            continue

        # child selection within best parent
        idxs = parent_to_idx[best_parent]
        idxs_sorted = sorted(idxs, key=lambda j: sims[j], reverse=True)

        child_best = float(sims[idxs_sorted[0]])
        selected = []
        for j in idxs_sorted:
            if len(selected) >= config.TOPK_CHILD:
                break
            if sims[j] >= config.CHILD_MIN_SIM or (child_best - sims[j] <= config.CHILD_MARGIN):
                selected.append(j)

        if not selected:
            rows.append({
                "Accident ID": inc_id,
                "Final_Category_Path": f"{best_parent} > OTHER",
                "Cosine": best_parent_sim,
                "Rank": 1,
                "Selected": True,
                "Below_UNCAT_Threshold": False,
                "Parent": best_parent,
                "Parent_Sim": best_parent_sim,
                "Parent_Margin": parent_margin,
                "Parent_LowConf": parent_low_conf
            })
        else:
            for rank, j in enumerate(selected, start=1):
                rows.append({
                    "Accident ID": inc_id,
                    "Final_Category_Path": tax_paths[j],
                    "Cosine": float(sims[j]),
                    "Rank": rank,
                    "Selected": True,
                    "Below_UNCAT_Threshold": False,
                    "Parent": best_parent,
                    "Parent_Sim": best_parent_sim,
                    "Parent_Margin": parent_margin,
                    "Parent_LowConf": parent_low_conf
                })

    assign_raw = pd.DataFrame(rows)
    return assign_raw
