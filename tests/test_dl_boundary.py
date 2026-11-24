"""X4 (#75): torch is imported only inside src/pokeagent/dl/."""

from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "pokeagent"


def test_torch_and_dl_imports_stay_inside_the_dl_package():
    for path in SRC.rglob("*.py"):
        if "dl" in path.relative_to(SRC).parts[:1]:
            continue
        text = path.read_text(encoding="utf-8")
        assert "import torch" not in text and "from torch" not in text, path
        assert "pokeagent.dl" not in text, path


def test_streamlit_and_dashboard_imports_stay_inside_the_dashboard_package():
    for path in SRC.rglob("*.py"):
        if "dashboard" in path.relative_to(SRC).parts[:1]:
            continue
        text = path.read_text(encoding="utf-8")
        assert "import streamlit" not in text and "from streamlit" not in text, path
        assert "pokeagent.dashboard" not in text, path
