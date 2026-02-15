# eMARS Consolidated Python package

This folder contains a consolidated, modular conversion of the original Jupyter notebook into a small Python package.

Files:
- `config.py`: configuration constants.
- `data.py`: load eMARS and taxonomy files.
- `embeddings.py`: model load and encoding (requires `sentence-transformers`).
- `assign.py`: hierarchical assignment logic.
- `consolidate.py`: consolidation and disambiguation rules.
- `evidence.py`: evidence-lock filtering.
- `render.py`: collapse children, depth-aware rendering and graph export.
- `main.py`: runner script to execute the full pipeline.
- `requirements.txt`: suggested packages.

Quick run (from the folder containing this package):

```powershell
python -m pip install -r requirements.txt
python -c "from eMARS_consolidated.main import run_all; run_all()"
```

Edit `config.py` to change paths or thresholds.
