"""Checkpoint evaluation; every unknown target (-1) counts as an error."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import csv
import json
import os
from pathlib import Path
import tempfile

import torch
from torch import nn

from .config import SEED

INPUT_KEYS = ("pixel_values", "input_ids", "attention_mask", "token_type_ids")


def move_inputs(batch, device):
    return {key: batch[key].to(device, non_blocking=device.type == "cuda")
            for key in INPUT_KEYS if key in batch}


def amp_context(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def fit_question_type_baseline(train_rows):
    """Fit only final train rows, after rare-answer removal; lexical tie-break."""
    counts = defaultdict(Counter)
    total = Counter()
    for row in train_rows:
        if row.get("split") != "train":
            raise ValueError("Baseline can only be fitted on final train")
        counts[row["question_type"]][row["answer"]] += 1
        total[row["answer"]] += 1
    if not total:
        raise ValueError("Baseline needs nonempty training rows")
    def most_common(counter):
        return min(counter, key=lambda answer: (-counter[answer], answer))
    return {"by_question_type": {kind: most_common(counter) for kind, counter in sorted(counts.items())},
            "fallback": most_common(total), "source": "final_train_only"}


def baseline_predictions(question_types, baseline, answer_to_idx):
    return [answer_to_idx.get(baseline["by_question_type"].get(kind, baseline["fallback"]), -1)
            for kind in question_types]


def compute_metrics(labels, predictions, question_types, answer_to_idx, *, confusion_limit=64):
    labels, predictions, question_types = list(labels), list(predictions), list(question_types)
    if not labels or len(labels) != len(predictions) or len(labels) != len(question_types):
        raise ValueError("Expected nonempty, equally sized labels/predictions/question_types")
    classes = len(answer_to_idx)
    if not classes or sorted(answer_to_idx.values()) != list(range(classes)):
        raise ValueError("Answer indices must be contiguous, nonempty and zero-based")
    if any(int(v) < -1 or int(v) >= classes for v in labels + predictions):
        raise ValueError("Label or prediction outside vocabulary (only -1 may be unknown)")
    counts = defaultdict(lambda: {"correct": 0, "total": 0, "unknown": 0})
    correct = 0
    unknown = 0
    matrix = [[0] * (classes + 1) for _ in range(classes + 1)] if classes <= confusion_limit else None
    for truth, predicted, kind in zip(labels, predictions, question_types):
        truth, predicted = int(truth), int(predicted)
        hit = truth >= 0 and predicted == truth
        correct += int(hit)
        unknown += int(truth == -1)
        counts[kind]["correct"] += int(hit)
        counts[kind]["total"] += 1
        counts[kind]["unknown"] += int(truth == -1)
        if matrix is not None:
            matrix[truth if truth >= 0 else classes][predicted if predicted >= 0 else classes] += 1
    names = [answer for answer, index in sorted(answer_to_idx.items(), key=lambda pair: pair[1])]
    return {"accuracy": correct / len(labels), "exact_match_accuracy": correct / len(labels),
            "correct": correct, "total": len(labels), "unknown": unknown,
            "known_answer_coverage": (len(labels) - unknown) / len(labels),
            "by_question_type": {kind: {**v, "accuracy": v["correct"] / v["total"]} for kind, v in sorted(counts.items())},
            "confusion_matrix": matrix, "confusion_labels": names + ["<UNKNOWN>"] if matrix is not None else None,
            "confusion_note": "Rows=true, columns=predicted; unknown targets remain in accuracy denominator" if matrix is not None else f"Skipped: vocabulary exceeds {confusion_limit} classes"}


@torch.no_grad()
def evaluate_loader(model, loader, answer_to_idx, *, device, baseline=None):
    device = torch.device(device)
    model.eval()
    labels, predictions, kinds, image_ids = [], [], [], []
    loss_sum = 0.0
    known_count = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
    for batch in loader:
        targets = batch["labels"].to(device, dtype=torch.long)
        with amp_context(device):
            logits = model(**move_inputs(batch, device))
        if logits.shape != (targets.numel(), len(answer_to_idx)) or targets.ndim != 1:
            raise ValueError("Expected logits (B,C) and labels (B,)")
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite evaluation logits")
        loss_sum += criterion(logits.float(), targets).item()
        known_count += int((targets >= 0).sum().item())
        labels.extend(targets.cpu().tolist())
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        kinds.extend(batch["question_type"])
        image_ids.extend(batch["image_id"])
    metrics = compute_metrics(labels, predictions, kinds, answer_to_idx)
    metrics["loss_known_answers"] = loss_sum / known_count if known_count else None
    baseline_labels = baseline_predictions(kinds, baseline, answer_to_idx) if baseline is not None else None
    baseline_metrics = compute_metrics(labels, baseline_labels, kinds, answer_to_idx) if baseline_labels is not None else None
    idx_to_answer = {v: k for k, v in answer_to_idx.items()}
    rows = [{"image_id": image, "question_type": kind, "true_label": truth, "predicted_label": predicted,
             "predicted_answer": idx_to_answer[predicted], "correct": truth >= 0 and truth == predicted,
             "baseline_label": baseline_labels[i] if baseline_labels is not None else ""}
            for i, (image, kind, truth, predicted) in enumerate(zip(image_ids, kinds, labels, predictions))]
    return {"metrics": metrics, "baseline": baseline_metrics, "predictions": rows}


def load_checkpoint(path):
    # Own checkpoints contain tensors and primitive metadata only; no arbitrary classes.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state_dict", "config", "answer_to_idx", "mode", "tokenizer", "preprocessing",
                "epoch", "seed", "validation_metric", "data_manifest", "baseline", "data_validation"}
    if not isinstance(checkpoint, dict) or not required <= checkpoint.keys():
        raise ValueError("Incomplete VQA checkpoint")
    if checkpoint["seed"] != SEED or checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint seed/schema")
    vocab = checkpoint["answer_to_idx"]
    if not vocab or sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("Invalid checkpoint vocabulary")
    if checkpoint["mode"] != checkpoint["config"].get("mode"):
        raise ValueError("Checkpoint mode/config mismatch")
    return checkpoint


def restore_tokenizer(metadata, directory):
    from transformers import AutoTokenizer
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in metadata["files"].items():
        if Path(name).name != name or name in {"", ".", ".."} or ":" in name or "\\" in name:
            raise ValueError("Unsafe tokenizer artifact name")
        (directory / name).write_bytes(content)
    return AutoTokenizer.from_pretrained(directory, local_files_only=True)


def save_evaluation(result, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stem}_metrics.json").write_text(json.dumps({k: v for k, v in result.items() if k != "predictions"}, indent=2), encoding="utf-8")
    with (output_dir / f"{stem}_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["predictions"][0]))
        writer.writeheader()
        writer.writerows(result["predictions"])
    with (output_dir / f"{stem}_question_types.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["question_type", "correct", "total", "unknown", "accuracy", "baseline_accuracy"])
        writer.writeheader()
        for kind, values in result["metrics"]["by_question_type"].items():
            writer.writerow({"question_type": kind, **values,
                "baseline_accuracy": result["baseline"]["by_question_type"][kind]["accuracy"] if result["baseline"] is not None else ""})
    if result["metrics"]["confusion_matrix"] is not None:
        with (output_dir / f"{stem}_confusion_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["true/predicted", *result["metrics"]["confusion_labels"]])
            writer.writerows([name, *row] for name, row in zip(result["metrics"]["confusion_labels"], result["metrics"]["confusion_matrix"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--data_dir", default=os.environ.get("PLANTVQA_DATA_DIR"))
    parser.add_argument("--csv", type=Path, help="Relocated annotation file; content hash must match checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/results"))
    args = parser.parse_args(argv)
    from .train import (build_pipeline, pipeline_manifest, select_device, update_summary)
    from .download import discover_csv, sha256
    from .models import ViTBERT
    checkpoint = load_checkpoint(args.ckpt)
    config = dict(checkpoint["config"])
    if args.data_dir:
        config["data_dir"] = args.data_dir
    source = args.csv or Path(config["source_csv"])
    if args.csv is None and not source.is_file() and args.data_dir:
        source = discover_csv(config["data_dir"])
    if sha256(source) != checkpoint["data_manifest"]["source_sha256"]:
        raise ValueError("Source annotations differ from checkpoint")
    device = select_device(args.device)
    with tempfile.TemporaryDirectory() as directory:
        tokenizer = restore_tokenizer(checkpoint["tokenizer"], directory)
        pipeline, images = build_pipeline(config, source, tokenizer=tokenizer,
                                           preprocessing=checkpoint["preprocessing"], evaluation=True)
        if pipeline.answer_vocab != checkpoint["answer_to_idx"]:
            raise ValueError("Rebuilt train answer vocabulary differs from checkpoint")
        manifest = pipeline_manifest(pipeline, source, config, images, decode_images=False)
        if manifest != checkpoint["data_manifest"]:
            raise ValueError("Data, image content, split or preprocessing fingerprint differs from checkpoint")
        model = ViTBERT(len(checkpoint["answer_to_idx"]), checkpoint["mode"], pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        result = evaluate_loader(model, pipeline.loaders["test"], checkpoint["answer_to_idx"],
                                 device=device, baseline=checkpoint["baseline"])
        result.update(checkpoint=str(args.ckpt), epoch=checkpoint["epoch"], seed=checkpoint["seed"],
                      mode=checkpoint["mode"], test_policy=config["test_policy"],
                      source_sha256=manifest["source_sha256"], test_view=pipeline.report["test_note"])
        save_evaluation(result, args.output_dir, args.ckpt.stem + "_test")
        update_summary(args.output_dir / "summary.csv", config, {"best_epoch": checkpoint["epoch"],
            "val_accuracy": checkpoint["validation_metric"]["value"], "checkpoint": str(args.ckpt),
            "test_accuracy": result["metrics"]["accuracy"],
            "test_baseline_accuracy": result["baseline"]["accuracy"]})
        print(json.dumps({"metrics": result["metrics"], "baseline": result["baseline"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
