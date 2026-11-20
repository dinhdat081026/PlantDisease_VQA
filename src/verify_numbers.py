"""Compare observations with documented reference numbers without altering data."""
import argparse
import json
import os
from pathlib import Path
from .download import discover_csv, sha256
from .eda import analyze, load_annotations

REFERENCE = {"raw.qa_pairs": 193609, "raw.images": 50073,
             "factoid.qa_pairs": 96825, "factoid.images": 45206}


def verify(rows):
    observed = analyze(rows)
    checks = []
    for key, expected in REFERENCE.items():
        group, metric = key.split(".")
        actual = observed[group][metric]
        checks.append({"metric": key, "expected": expected, "actual": actual, "passed": actual == expected})
    extra = {"nine_question_types": len({r["question_type"] for r in rows}) == 9,
             "no_image_overlap": not any(observed["overlap_image_ids"].values()),
             "factoid_at_most_three_words": observed["factoid"]["max_answer_words"] <= 3}
    passed = all(c["passed"] for c in checks) and all(extra.values())
    return {"passed": passed, "checks": checks, "invariants": extra, "observations": observed,
            "discrepancy_guidance": [] if passed else [
                "Inspect source revision/hash and CSV selection, exact question_type labels and original split values.",
                "Count referenced image IDs separately from image files; filtering can remove all QA for some images.",
                "No cause is confirmed by count differences alone. Do not edit rows to force reference totals."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=os.environ.get("PLANTVQA_DATA_DIR", "data/PlantVillageVQA"))
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/results/verification.json"))
    args = parser.parse_args()
    source = args.csv or discover_csv(args.data_dir)
    if args.output.resolve() == source.resolve():
        parser.error("Output must not overwrite source CSV")
    before = sha256(source)
    report = verify(load_annotations(source))
    if sha256(source) != before:
        raise RuntimeError("Source CSV changed during verification")
    report.update(source_csv=str(source), source_sha256=before,
                  scope="Annotation invariants only; not a training authorization or image decoding check")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
