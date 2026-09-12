"""Phase 1 README-hygiene checks: the README must explain the callback use
case and link the Ultralytics docs it's demonstrating, and those claims must
match what the notebook actually does -- read as plain JSON rather than
executed (it installs ultralytics and calls model.train(), well out of scope
for this phase's issue)."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
NOTEBOOK = REPO_ROOT / "Ultralutics_Yolo_Custom_Callbacks.ipynb"


def _readme_text() -> str:
    return README.read_text(encoding="utf-8")


def _notebook_source() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in nb["cells"])


def test_readme_names_the_callback_use_case():
    text = _readme_text()
    assert "callback" in text.lower()
    assert "add_callback" in text


def test_readme_links_ultralytics_docs():
    assert "docs.ultralytics.com" in _readme_text()


def test_readme_claims_match_the_notebook_source():
    source = _notebook_source()
    assert "add_callback" in source
    assert "on_train_epoch_end" in source
    assert "YOLO" in source
