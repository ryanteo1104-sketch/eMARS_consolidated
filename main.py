"""
Main runner to execute full pipeline: load data -> embeddings -> assign -> consolidate -> evidence -> render.
"""
import os
import pandas as pd

import config
from data import load_emars, load_taxonomy
from embeddings import load_model, encode_all
from assign import assign_hierarchical
from consolidate import consolidate_and_disambiguate
from evidence import prepare_expected_terms_cache, apply_evidence_gate
from render import collapse_sparse_children, depth_aware_render, export_graph


def run_all(save_dir: str = None):
    save_dir = save_dir or os.getcwd()
    print(f"Working directory: {save_dir}")

    # 1. Load
    em, id_col, title_col, desc_col = load_emars(config.EMARS_XLSX)
    tx, tax_path_col, tax_paths = load_taxonomy(config.TAXON_XLSX)
    print(f"Loaded emars: {len(em)} rows, taxonomy paths: {len(tax_paths)}")

    # 2. Model + embeddings
    try:
        model = load_model(config.MODEL_NAME)
    except Exception as e:
        raise RuntimeError("Failed to load embedding model. Install sentence-transformers and try again.")
    tax_emb, em_emb = encode_all(model, tax_paths, em["_emars_text_"].tolist())

    # 3. Assignment
    assign_raw = assign_hierarchical(em=em, id_col=id_col, tax_paths=tax_paths, em_emb=em_emb, tax_emb=tax_emb)
    assign_raw.to_csv(os.path.join(save_dir, "eMARS_taxonomy_assignment_raw.csv"), index=False)
    print("Saved raw assignment CSV")

    # 4. Consolidation
    assign = consolidate_and_disambiguate(assign_raw, em, id_col)
    assign.to_csv(os.path.join(save_dir, "eMARS_assignment_consolidated.csv"), index=False)
    print("Saved consolidated assignment CSV")

    # 5. Evidence
    path_term_cache = prepare_expected_terms_cache(assign, config.TAXON_XLSX)
    assign_supported = apply_evidence_gate(assign, em, path_term_cache)
    assign_supported.to_csv(os.path.join(save_dir, "output_evidence_locked.csv"), index=False)
    print("Saved evidence-locked CSV")

    # 6. Collapse sparse children and render
    assign_collapsed = collapse_sparse_children(assign_supported)
    assign_rendered = depth_aware_render(assign_collapsed)
    assign_rendered.to_csv(os.path.join(save_dir, "eMARS_assignment_with_render.csv"), index=False)
    print("Saved rendered assignment CSV")

    # 7. Graph export (dot + optional pdf)
    dot, pdf = export_graph(assign_rendered)
    print(f"Graph files: {dot}, {pdf}")

    return {
        "raw": assign_raw,
        "consolidated": assign,
        "evidence": assign_supported,
        "rendered": assign_rendered,
        "dot": dot,
        "pdf": pdf,
    }


if __name__ == "__main__":
    run_all()
