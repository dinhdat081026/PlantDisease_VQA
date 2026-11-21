"""Rebuild the audit tables and PDF plots from the preserved CSV/JSON artifacts."""

import csv
import json
import math
from collections import Counter
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "outputs/results"
FIGURES = ROOT / "reports/figures"


def read_csv(name):
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (RESULTS / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio(num, den):
    return num / den if den else 0.0


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    from sys import path

    path.insert(0, str(ROOT))
    from src.config import ABLATIONS, DATASET_MODES

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"]})
    history = read_csv("full_mini_history.csv")
    summary_rows = read_csv("summary.csv")
    summary = {(r["level"], r["mode"]): r for r in summary_rows}
    assert len(summary) == len(summary_rows), "Duplicate configuration in summary.csv"
    observed = summary[("mini", "full")]
    best = max(history, key=lambda row: float(row["val_accuracy"]))
    assert [int(row["epoch"]) for row in history] == list(range(1, int(observed["epochs"]) + 1))
    assert best["epoch"] == observed["best_epoch"]
    assert math.isclose(float(best["val_accuracy"]), float(observed["val_accuracy"]))
    fields = ("seed", "epochs", "best_epoch", "val_accuracy", "checkpoint", "test_accuracy", "test_baseline_accuracy")
    plans = [
        {"level": level, "mode": mode, "workspace_results": "verified" if (level, mode) == ("mini", "full") else
         ("summary_present_unverified" if (level, mode) in summary else "missing"),
         **{key: summary.get((level, mode), {}).get(key, "") for key in fields}}
        for level, mode in product(DATASET_MODES, ABLATIONS)
    ]
    write_csv("audit_experiment_summary.csv", plans)

    predictions = read_csv("full_mini_test_predictions.csv")
    report = json.loads((RESULTS / "full_mini_test_metrics.json").read_text(encoding="utf-8"))
    labels = report["metrics"]["confusion_labels"]
    count = len(predictions)
    assert all(-1 <= int(row[key]) < len(labels) - 1 for row in predictions
               for key in ("true_label", "predicted_label", "baseline_label"))
    class_rows = []
    for predictor, key in (("ViT+BERT", "predicted_label"), ("Train majority per type", "baseline_label")):
        pairs = Counter((int(row["true_label"]), int(row[key])) for row in predictions)
        indices = [*range(len(labels) - 1), -1]
        matrix = [[pairs[(truth, pred)] for pred in indices] for truth in indices]
        source = report["metrics" if key == "predicted_label" else "baseline"]
        assert matrix == source["confusion_matrix"]
        assert source["total"] == count
        assert source["correct"] == sum(pairs[(i, i)] for i in indices if i >= 0)
        for i, label in zip(indices, labels):
            support = sum(pairs[(i, j)] for j in indices)
            predicted = sum(pairs[(j, i)] for j in indices)
            tp = pairs[(i, i)] if i >= 0 else 0
            class_rows.append({"predictor": predictor, "label": label, "support": support, "predicted": predicted,
                               "precision": ratio(tp, predicted), "recall": ratio(tp, support),
                               "f1": ratio(2 * tp, support + predicted)})
    write_csv("audit_full_mini_per_class.csv", class_rows)
    print(f"Test rows={count}; unique images={len({r['image_id'] for r in predictions})}")
    for predictor in ("ViT+BERT", "Train majority per type"):
        rows = [r for r in class_rows if r["predictor"] == predictor]
        print(predictor, {key: round(sum(r[key] for r in rows) / len(rows), 9)
                          for key in ("precision", "recall", "f1")})

    epochs = [int(row["epoch"]) for row in history]
    for name, keys, legend, unit in (
        ("training_loss", ("train_loss", "val_loss_known"), ("Train (as logged)", "Validation (known labels)"), "Loss (logged scale)"),
        ("training_accuracy", ("train_accuracy", "val_accuracy"), ("Train", "Validation"), "Accuracy (%)"),
    ):
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        scale = 100 if name.endswith("accuracy") else 1
        for key, label in zip(keys, legend):
            ax.plot(epochs, [scale * float(row[key]) for row in history], marker="o", label=label)
        ax.axvline(int(best["epoch"]), color="0.4", linestyle=":", label="Selected epoch")
        ax.set(title=f"Mini / full — {name.replace('_', ' ').title()}", xlabel="Epoch", ylabel=unit, xticks=epochs)
        ax.grid(axis="y", alpha=.25)
        ax.legend()
        save(fig, name)

    model = report["metrics"]
    baseline = report["baseline"]
    types = list(model["by_question_type"])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for offset, metric, legend in ((-.18, model, "ViT+BERT"), (.18, baseline, "Train majority per type")):
        values = [metric["accuracy"], *(metric["by_question_type"][kind]["accuracy"] for kind in types)]
        ax.bar([x + offset for x in range(len(values))], [100 * v for v in values], width=.36, label=legend)
    ax.set(title="Mini test view — paired accuracy", ylabel="Exact-match accuracy (%)", xlabel="Question type", ylim=(0, 110))
    ax.set_xticks(range(len(types) + 1), ["Overall", *(t.replace(" ", "\n", 1) for t in types)])
    ax.legend()
    save(fig, "test_accuracy_baseline")

    cm = model["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(7, 6))
    grid = ax.imshow(cm, cmap="Blues")
    fig.colorbar(grid, ax=ax, label="QA count")
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            if value:
                ax.text(j, i, value, ha="center", va="center", color="white" if value > grid.norm.vmax / 2 else "black")
    ax.set(title="Mini / full — test confusion matrix", xlabel="Predicted label", ylabel="True label")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    save(fig, "test_confusion_matrix")

    rows = [r for r in class_rows if r["predictor"] == "ViT+BERT"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for offset, metric in ((-.24, "precision"), (0, "recall"), (.24, "f1")):
        ax.bar([x + offset for x in range(len(labels))], [100 * row[metric] for row in rows],
               width=.24, label=metric.upper() if metric == "f1" else metric.title())
    ax.set(title="Mini / full — test scores by answer", xlabel="Answer label (true support)", ylabel="Score (%)", ylim=(0, 114))
    ax.set_xticks(range(len(labels)), [f"{r['label']} ({r['support']})" for r in rows], rotation=45, ha="right")
    ax.legend(ncol=3)
    save(fig, "test_metrics_by_class")


if __name__ == "__main__":
    main()
