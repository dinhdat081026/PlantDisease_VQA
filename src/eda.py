"""Descriptive statistics only; never creates an answer vocabulary or new splits."""
import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path

from .download import REQUIRED_COLUMNS, discover_csv, image_key, index_images, resolve_image, sha256

# Exact labels from the source taxonomy; the four-type selection is project policy.
FACTOID_TYPES = frozenset({"Detailed Verification", "General Health Assessment",
                           "Plant Species Identification", "Visual Attribute Grounding"})
QUESTION_TYPES = FACTOID_TYPES | frozenset({"Existence & Sanity Check", "Specific Disease Identification",
    "Comprehensive Description", "Causal Reasoning", "Counterfactual Reasoning"})


def load_annotations(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(REQUIRED_COLUMNS) or set(reader.fieldnames) != set(REQUIRED_COLUMNS):
            raise ValueError(f"Expected exactly six columns {REQUIRED_COLUMNS}; got {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty annotation CSV")
    for number, row in enumerate(rows, start=2):
        if set(row) != set(REQUIRED_COLUMNS) or any(not isinstance(v, str) or not v.strip() for v in row.values()):
            raise ValueError(f"Malformed or blank field at CSV row {number}")
        if row["question_type"] not in QUESTION_TYPES:
            raise ValueError(f"Unknown question type at CSV row {number}: {row['question_type']}")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Invalid original split at CSV row {number}")
    return rows


def filter_factoid(rows):
    # Do not use answer length alone: short answers also occur in excluded categories.
    return [dict(r) for r in rows if r["question_type"] in FACTOID_TYPES]


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(rows):
    lengths = [len(r["answer"].split()) for r in rows]
    return {"qa_pairs": len(rows), "images": len({image_key(r["image_id"]) for r in rows}),
            "unique_answers": len({r["answer"] for r in rows}),
            "median_answer_words": percentile(lengths, .5), "p95_answer_words": percentile(lengths, .95),
            "max_answer_words": max(lengths, default=0)}


def split_overlap(rows):
    splits = {s: {image_key(r["image_id"]) for r in rows if r["split"] == s} for s in sorted({r["split"] for r in rows})}
    names = sorted(splits)
    return {f"{left}/{right}": sorted(splits[left] & splits[right])
            for i, left in enumerate(names) for right in names[i + 1:]}


def analyze(rows):
    factoid = filter_factoid(rows)
    return {"raw": summarize(rows), "factoid": summarize(factoid),
            "question_types": {t: summarize([r for r in rows if r["question_type"] == t]) for t in sorted(QUESTION_TYPES)},
            "original_splits": {s: summarize([r for r in rows if r["split"] == s]) for s in sorted({r["split"] for r in rows})},
            "overlap_image_ids": split_overlap(rows),
            "duplicate_qa_rows": len(rows) - len({tuple(r[k] for k in REQUIRED_COLUMNS) for r in rows}),
            "note": "All-split descriptive answer counts are not a model vocabulary. Source splits are unchanged."}


def save_figures(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]}):
        counts = Counter(r["question_type"] for r in rows)
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = sorted(counts)
        ax.barh(labels, [counts[t] for t in labels])
        ax.set_xlabel("QA pairs")
        ax.set_title("Question categories (original annotations)")
        fig.savefig(output_dir / "question_types.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, subset, title in zip(axes, (rows, filter_factoid(rows)), ("All questions", "Four factoid categories")):
            counts = Counter(len(r["answer"].split()) for r in subset)
            ax.bar(sorted(counts), [counts[n] for n in sorted(counts)])
            ax.set(xlabel="Answer length (words)", ylabel="QA pairs", title=title)
        fig.savefig(output_dir / "answer_lengths.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=os.environ.get("PLANTVQA_DATA_DIR", "data/PlantVillageVQA"))
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--annotations-only", action="store_true", help="Explicitly skip checking local image presence; does not validate image decoding")
    args = parser.parse_args()
    source = args.csv or discover_csv(args.data_dir)
    before = sha256(source)
    rows = load_annotations(source)
    report = analyze(rows)
    report.update(source_csv=str(source), source_sha256=before, image_presence_checked=not args.annotations_only, image_decoding_checked=False)
    if not args.annotations_only:
        images = index_images(args.data_dir)
        for image_id in {r["image_id"] for r in rows}:
            resolve_image(image_id, images)
        report["image_files"] = len(images)
        report["images_without_qa"] = len(set(images) - {image_key(r["image_id"]) for r in rows})
    if sha256(source) != before:
        raise RuntimeError("Source CSV changed during analysis")
    results = args.output_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "eda.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (results / "question_types.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["question_type", *summarize([])])
        writer.writeheader()
        writer.writerows({"question_type": t, **metrics} for t, metrics in report["question_types"].items())
    save_figures(rows, args.output_dir / "figures")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
