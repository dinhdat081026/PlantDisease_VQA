"""Check scaffold without importing ML libraries or starting training."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import ABLATIONS, DATASET_MODES, Config, IMAGE_MODEL, TEXT_MODEL

if __name__ == "__main__":
    for relative in ("AGENTS.md", "README.md", "requirements.txt", ".gitignore", "docs/dataset_card.md", "src/data.py", "tests/test_data.py", "outputs/.gitkeep"):
        if not (ROOT / relative).is_file():
            raise SystemExit(f"Missing {relative}")
    for mode in DATASET_MODES:
        for ablation in ABLATIONS:
            Config(dataset_mode=mode, ablation=ablation, class_name="example" if mode == "class" else None)
    print(f"Scaffold configuration OK: {IMAGE_MODEL}; {TEXT_MODEL}")
    print("No training performed; real-data validation still required.")
