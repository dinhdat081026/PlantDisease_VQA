"""Rebuild recorded split and baseline from annotations without reading images."""

import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import SEED
from src.data import build_answer_vocab, build_subset, infer_species, split_by_image
from src.download import image_key, sha256
from src.eda import load_annotations
from src.evaluate import baseline_predictions, fit_question_type_baseline
from src.train import digest_json


def main():
    results = ROOT / "outputs/results"
    source = ROOT / "data/source_inspection/PlantVillageVQA.csv"
    record = json.loads((results / "data_validation_mini_subset.json").read_text(encoding="utf-8"))
    manifest = record["data_manifest"]
    cfg = manifest["split_config"]
    assert SEED == cfg["seed"] == record["seed"]
    assert sha256(source) == manifest["source_sha256"]
    for hashes in (manifest["data_code_sha256"], record["synthetic_tests"]["test_code_sha256"]):
        for name, expected in hashes.items():
            assert sha256(ROOT / name) == expected, name
    raw = load_annotations(source)
    species = infer_species(raw)
    selected = build_subset(raw, cfg["level"], test_policy=cfg["test_policy"],
                            species_by_image=species, rare_min_images=cfg["rare_min_images"],
                            target_images=cfg["target_images"])
    parts = split_by_image(selected, species_by_image=species, rare_min_images=cfg["rare_min_images"],
                           validation_fraction=cfg["validation_fraction"])
    vocabulary = build_answer_vocab(parts["train"], min_count=cfg["min_answer_count"])
    parts["train"] = [row for row in parts["train"] if row["answer"] in vocabulary]
    ids = {name: {image_key(row["image_id"]) for row in rows} for name, rows in parts.items()}
    assert not any(ids[a] & ids[b] for a, b in combinations(ids, 2))
    assert len(set.union(*ids.values())) == manifest["image_count"]
    assert digest_json(vocabulary) == manifest["answer_vocab_sha256"]
    assert {name: digest_json(rows) for name, rows in parts.items()} == manifest["split_fingerprints"]
    print("Raw selected images/QA:", len({image_key(r["image_id"]) for r in selected}), len(selected))
    print("Final images/QA:", {name: (len(ids[name]), len(rows)) for name, rows in parts.items()})
    baseline = fit_question_type_baseline(parts["train"])
    kinds = [row["question_type"] for row in parts["test"]]
    predicted = baseline_predictions(kinds, baseline, vocabulary)
    truth_and_baseline = Counter(
        (row["image_id"], row["question_type"], vocabulary.get(row["answer"], -1), p)
        for row, p in zip(parts["test"], predicted, strict=True)
    )
    with (results / "full_mini_test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        saved = Counter(
            (row["image_id"], row["question_type"], int(row["true_label"]), int(row["baseline_label"]))
            for row in csv.DictReader(stream)
        )
    assert saved == truth_and_baseline
    print("All source/code/test hashes, three split fingerprints, vocabulary and test-label/baseline rows match.")
    print("Baseline:", baseline)


if __name__ == "__main__":
    main()
