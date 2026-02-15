"""
Rendering helpers: collapse sparse children, depth-aware render, graph export.
"""
import os
import pandas as pd
from utils import split_any, _fnum
from config import MIN_CHILD_SUPPORT, MAX_CHILD_PER_PARENT, ENABLE_DEPTH_AWARE_RENDER, MAX_DEPTH_RENDER, DEPTH_CAP_LABEL, MIN_LEAF_SUPPORT_RENDER, LOW_SUPPORT_LABEL


def collapse_sparse_children(assign_df: pd.DataFrame):
    tmp = assign_df[assign_df["Consolidated_Path"] != "UNCAT"].copy()
    tmp["Parent"] = tmp["Consolidated_Path"].map(lambda p: split_any(p)[0] if len(split_any(p)) else "UNCAT")
    tmp["Leaf"] = tmp["Consolidated_Path"].map(lambda p: split_any(p)[-1] if len(split_any(p)) else "UNCAT")
    child_support = (tmp.groupby(["Parent", "Leaf"])["Accident ID"].nunique().rename("Support").reset_index())
    child_support["RankInParent"] = child_support.groupby("Parent")["Support"].rank(method="first", ascending=False)
    keep_df = child_support[(child_support["Support"] >= MIN_CHILD_SUPPORT) & (child_support["RankInParent"] <= MAX_CHILD_PER_PARENT)].copy()
    keep_set = set(zip(keep_df["Parent"], keep_df["Leaf"]))
    def collapse_path(path: str):
        if path == "UNCAT": return path
        parts = split_any(path)
        if not parts: return "UNCAT"
        if len(parts) == 1: return parts[0]
        parent, leaf = parts[0], parts[-1]
        if (parent, leaf) in keep_set:
            return path
        return f"{parent} > OTHER"
    assign_df["Consolidated_Path"] = assign_df["Consolidated_Path"].map(collapse_path)
    return assign_df


def depth_aware_render(assign_df: pd.DataFrame):
    assign = assign_df.copy()
    assign["Consolidated_Path_Full"] = assign["Consolidated_Path"]
    tmp = assign[assign["Consolidated_Path_Full"].astype(str) != "UNCAT"].copy()
    tmp["__parts"] = tmp["Consolidated_Path_Full"].map(lambda s: split_any(s))
    tmp["__parent"] = tmp["__parts"].map(lambda z: z[0] if len(z) else "")
    tmp["__leaf"] = tmp["__parts"].map(lambda z: z[-1] if len(z) else "")
    leaf_support = (tmp.groupby(["__parent", "__leaf"])["Accident ID"].nunique().rename("LeafSupport").reset_index())
    tmp = tmp.merge(leaf_support, on=["__parent", "__leaf"], how="left")
    assign = assign.merge(tmp[["Accident ID", "Consolidated_Path_Full", "LeafSupport"]].drop_duplicates(), on=["Accident ID", "Consolidated_Path_Full"], how="left")

    def _allowed_depth(row):
        p = str(row.get("Consolidated_Path_Full", "UNCAT"))
        if p == "UNCAT": return 0
        cos = _fnum(row.get("Cosine", 0.0), 0.0)
        cov = _fnum(row.get("Evidence_Coverage", 0.0), 0.0)
        conf = 0.72 * cos + 0.28 * cov
        d = 1
        if conf >= 0.45 and cov >= 0.10: d = 2
        if conf >= 0.54 and cov >= 0.22: d = 3
        if conf >= 0.62 and cov >= 0.32: d = 4
        max_depth = int(MAX_DEPTH_RENDER)
        return int(min(max(d, 1), max_depth))

    def _render_path(row):
        p = str(row.get("Consolidated_Path_Full", "UNCAT"))
        if p == "UNCAT": return "UNCAT"
        parts = split_any(p)
        if not parts: return "UNCAT"
        d = int(max(1, min(len(parts), row.get("AllowedDepth", 1))))
        out = parts[:d]
        if d < len(parts):
            return " > ".join(out + [DEPTH_CAP_LABEL])
        ls = row.get("LeafSupport", None)
        leaf = str(parts[-1]).strip().lower()
        if len(parts) >= 3 and leaf != "other":
            if (ls is not None) and int(ls) < int(MIN_LEAF_SUPPORT_RENDER):
                return " > ".join(parts[:-1] + [LOW_SUPPORT_LABEL])
        return " > ".join(parts)

    if ENABLE_DEPTH_AWARE_RENDER:
        assign["AllowedDepth"] = assign.apply(_allowed_depth, axis=1)
        assign["Consolidated_Path_Render"] = assign.apply(_render_path, axis=1)
    else:
        assign["AllowedDepth"] = None
        assign["Consolidated_Path_Render"] = assign["Consolidated_Path_Full"]
    return assign


def export_graph(assign_df: pd.DataFrame, dot_filename="categorisation_tree_FINAL.dot", pdf_filename="categorisation_tree_FINAL.pdf"):
    try:
        import networkx as nx
        import shutil
        import subprocess
    except Exception:
        raise
    viz_df = assign_df.copy()
    def clean_path(row):
        p_render = str(row.get("Consolidated_Path_Render", ""))
        p_full = str(row.get("Consolidated_Path_Full", ""))
        if "depth-capped" in p_render:
            target = p_full
        else:
            target = p_render
        if "UNCAT" in target:
            return "Uncategorized"
        return target
    viz_df["Fixed_Path"] = viz_df.apply(clean_path, axis=1)
    G = nx.DiGraph()
    node_counts = {}
    node_direct = {}
    id_col = "Accident ID" if "Accident ID" in viz_df.columns else viz_df.columns[0]
    unique_accidents = viz_df[id_col].nunique()
    ROOT_LABEL = "eMARS Total"
    node_counts[ROOT_LABEL] = unique_accidents
    for _, row in viz_df.iterrows():
        path_str = row["Fixed_Path"]
        if ">" in path_str:
            parts = [p.strip() for p in path_str.split(">")]
        elif "/" in path_str:
            parts = [p.strip() for p in path_str.split("/")]
        else:
            parts = [path_str.strip()]
        parts = parts[:int(MAX_DEPTH_RENDER)]
        current_node = ROOT_LABEL
        for i, part in enumerate(parts):
            if not part: continue
            child_node = part
            if not G.has_edge(current_node, child_node):
                G.add_edge(current_node, child_node)
            node_counts[child_node] = node_counts.get(child_node, 0) + 1
            if i == len(parts) - 1:
                node_direct[child_node] = node_direct.get(child_node, 0) + 1
            current_node = child_node
    def escape_label(s):
        return s.replace('"', '\\"').replace('\n', ' ')
    with open(dot_filename, "w", encoding="utf-8") as f:
        f.write("digraph Tree {\n")
        f.write('  rankdir=LR;\n')
        f.write('  node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10, margin=0.2];\n')
        for node in list(G.nodes()) + [ROOT_LABEL]:
            if node not in node_counts: continue
            count = node_counts[node]
            direct = node_direct.get(node, 0)
            fill = "#ebf3f9"
            if "Uncategorized" in node: fill = "#ffebee"
            if node == ROOT_LABEL: fill = "#dddddd"
            if node == ROOT_LABEL:
                label = f"{node}\nUnique Incidents: {count}"
            else:
                label = f"{node}\nTags: {count}"
                if direct > 0 and direct != count:
                    label += f"\n(Direct: {direct})"
            f.write(f'  "{escape_label(node)}" [label="{escape_label(label)}", fillcolor="{fill}"];\n')
        for u, v in G.edges():
            f.write(f'  "{escape_label(u)}" -> "{escape_label(v)}";\n')
        f.write("}\n")
    dot_exe = shutil.which("dot")
    if dot_exe:
        try:
            subprocess.run([dot_exe, "-Tpdf", dot_filename, "-o", pdf_filename], check=True)
            return dot_filename, pdf_filename
        except Exception:
            return dot_filename, None
    return dot_filename, None
