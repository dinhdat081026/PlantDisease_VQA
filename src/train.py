"""Four-epoch VQA training; real CLI training requires a fresh data-validation artifact.

Use --validate-only to run pipeline pytest checks, decode selected real images
and write evidence. This path never constructs ViTBERT or an optimizer.
Default test_policy preserves the full original factoid test view; mini/class
budgets that cannot preserve it require an explicit --test-policy subset.
"""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
# Disable unused Transformers backends before any project/HF module can import them.
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn

from .config import ABLATIONS, IMAGE_MODEL, SEED, TEXT_MODEL
from .data import build_answer_vocab, create_dataloaders, filter_factoid
from .download import discover_csv, image_key, index_images, sha256
from .eda import load_annotations
from .evaluate import amp_context, evaluate_loader, fit_question_type_baseline, move_inputs


@dataclass
class TrainConfig:
    level: str = "class"
    mode: str = "full"
    epochs: int = 4
    encoder_lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = .01
    max_grad_norm: float = 1.0
    batch_size: int = 16
    num_workers: int = 0
    max_length: int = 64
    rare_min_images: int = 5
    target_images: int | None = None
    test_policy: str = "preserve"
    data_dir: str = "data/PlantVillageVQA"
    source_csv: str = ""
    output_dir: str = "outputs"
    gradient_accumulation_steps: int = 1

    def __post_init__(self):
        if self.level not in {"mini", "class", "full"} or self.mode not in ABLATIONS:
            raise ValueError("Invalid level/mode")
        if self.test_policy not in {"preserve", "subset"}:
            raise ValueError("Invalid test policy")
        if self.epochs < 1 or self.batch_size < 1 or self.num_workers < 0 or self.max_length < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("Invalid training dimensions")
        if any(not math.isfinite(v) or v <= 0 for v in (self.encoder_lr, self.head_lr, self.max_grad_norm)):
            raise ValueError("Learning rates and gradient clipping must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Invalid weight decay")


def select_device(name="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested device unavailable; supported devices are CPU/CUDA")
    return device


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_optimizer(model, config):
    groups = {}
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError("Parameter occurs more than once")
        seen.add(id(parameter))
        family = "encoder" if name.startswith(("image_encoder.", "text_encoder.")) else "head"
        decay = parameter.ndim > 1 and not name.endswith(".bias")
        key = (family, decay)
        groups.setdefault(key, []).append(parameter)
    if not groups:
        raise ValueError("No trainable parameters")
    return torch.optim.AdamW([{"params": parameters,
        "lr": config.encoder_lr if family == "encoder" else config.head_lr,
        "weight_decay": config.weight_decay if decay else 0.0,
        "group_name": family + ("_decay" if decay else "_no_decay")}
        for (family, decay), parameters in sorted(groups.items())])


def train_one_epoch(model, loader, optimizer, *, device, scaler, max_grad_norm=1.0, accumulation_steps=1):
    device = torch.device(device)
    model.train()  # ViTBERT.train keeps frozen encoders in eval mode.
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum, optimizer_steps = 0, 0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    num_batches = len(loader)
    for batch_index, batch in enumerate(loader):
        labels = batch["labels"].to(device, dtype=torch.long)
        if labels.ndim != 1 or not labels.numel() or (labels < 0).any():
            raise ValueError("Train labels must be nonempty, known, nonnegative class indices")
        with amp_context(device):
            logits = model(**move_inputs(batch, device))
            loss = criterion(logits.float(), labels)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        scaler.scale(loss / accumulation_steps).backward()
        should_step = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == num_batches
        if should_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # GradScaler skips optimizer.step on overflow and lowers its scale.
            if scaler.get_scale() >= previous_scale:
                optimizer_steps += 1
            optimizer.zero_grad(set_to_none=True)
        count = labels.numel()
        total += count
        loss_sum += loss.item() * accumulation_steps * count
        correct += int((logits.detach().argmax(-1) == labels).sum().item())
    if not total:
        raise ValueError("Empty training loader")
    return {"loss": loss_sum / total, "accuracy": correct / total, "total": total, "optimizer_steps": optimizer_steps}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def preprocessing_metadata(config):
    from timm.data import resolve_data_config
    from timm.models import get_pretrained_cfg
    cfg = get_pretrained_cfg(IMAGE_MODEL)
    if cfg is None:
        raise ValueError("Missing pretrained image config")
    return {"image_model": IMAGE_MODEL, "text_model": TEXT_MODEL,
            "data_config": resolve_data_config(pretrained_cfg=cfg.to_dict()),
            "train_augmentation": {"hflip": .5, "vflip": 0.0, "color_jitter": .4,
                                   "scale": (.08, 1.0), "ratio": (3 / 4, 4 / 3), "re_prob": 0.0},
            "image_format": "normalized", "max_length": config["max_length"]}


def build_pipeline(config, source, *, tokenizer=None, preprocessing=None, evaluation=False):
    from timm.data import create_transform
    preprocessing = preprocessing or preprocessing_metadata(config)
    if preprocessing["image_format"] != "normalized" or preprocessing["max_length"] != config["max_length"]:
        raise ValueError("Preprocessing and token length contract mismatch")
    image_config = preprocessing["data_config"]
    eval_transform = create_transform(**image_config, is_training=False)
    train_transform = eval_transform if evaluation else create_transform(
        **image_config, **preprocessing["train_augmentation"], is_training=True)
    images = index_images(config["data_dir"])
    pipeline = create_dataloaders(load_annotations(source), image_index=images,
        level=config["level"], test_policy=config["test_policy"], target_images=config["target_images"],
        rare_min_images=config["rare_min_images"], batch_size=config["batch_size"],
        num_workers=config["num_workers"], max_length=config["max_length"], tokenizer=tokenizer,
        train_transform=train_transform, eval_transform=eval_transform,
        pin_memory=torch.cuda.is_available())
    return pipeline, images


def pipeline_manifest(pipeline, source, config, images, *, decode_images=False):
    """Content hashes bind evidence/checkpoints to annotations, images and exact views."""
    ids = sorted({image_key(r["image_id"]) for rows in pipeline.splits.values() for r in rows})
    image_hashes = []
    for key in ids:
        path = images[key]
        if decode_images:
            from PIL import Image
            with Image.open(path) as image:
                image.convert("RGB").load()
        image_hashes.append((key, sha256(path)))
    root = Path(__file__).resolve().parents[1]
    data_code = {name: sha256(root / name) for name in ("src/config.py", "src/data.py", "src/download.py", "src/eda.py")}
    tokenizer = pipeline.loaders["train"].dataset.tokenizer
    return {"source_sha256": sha256(source), "image_fingerprint": digest_json(image_hashes),
            "image_count": len(ids), "data_code_sha256": data_code,
            "split_fingerprints": {name: digest_json(rows) for name, rows in pipeline.splits.items()},
            "answer_vocab_sha256": digest_json(pipeline.answer_vocab),
            "tokenizer_vocab_sha256": digest_json(tokenizer.get_vocab()),
            "tokenizer_settings": {"do_lower_case": getattr(tokenizer, "do_lower_case", None),
                "padding_side": tokenizer.padding_side, "truncation_side": tokenizer.truncation_side,
                "special_tokens_map": tokenizer.special_tokens_map},
            "split_config": {**{k: config[k] for k in ("level", "test_policy", "rare_min_images", "target_images", "max_length")},
                             "seed": SEED, "validation_fraction": .15, "min_answer_count": 5},
            "eval_transform": repr(pipeline.loaders["val"].dataset.transform)}


def check_pipeline_invariants(pipeline, source_rows, test_policy):
    sets = {s: {image_key(r["image_id"]) for r in rows} for s, rows in pipeline.splits.items()}
    if not all(sets.values()) or any(sets[a] & sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Empty split or image leakage")
    if build_answer_vocab(pipeline.splits["train"]) != pipeline.answer_vocab:
        raise ValueError("Invalid train-only answer vocabulary")
    if any(r["answer"] not in pipeline.answer_vocab for r in pipeline.splits["train"]):
        raise ValueError("Rare/unknown train answers remain")
    source_test = [r for r in filter_factoid(source_rows) if r["split"] == "test"]
    if test_policy == "preserve" and pipeline.splits["test"] != source_test:
        raise ValueError("Original factoid test view was changed")
    source_membership = {image_key(r["image_id"]): r["split"] for r in source_rows}
    for name, rows in pipeline.splits.items():
        expected = "test" if name == "test" else "train"
        if any(source_membership[image_key(r["image_id"])] != expected for r in rows):
            raise ValueError("Official train/test membership changed")
    return True


def validate_only(pipeline, source, config, images, output, *, test_files=None):
    root = Path(__file__).resolve().parents[1]
    output = Path(output)
    if output.resolve() == Path(source).resolve() or output.resolve().is_relative_to(Path(config["data_dir"]).resolve()):
        raise ValueError("Validation evidence must not overwrite or live inside source data")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "seed": SEED}
    # Invalidate old evidence BEFORE rechecking so a failed run cannot leave a green artifact.
    output.write_text(json.dumps(report), encoding="utf-8")
    check_pipeline_invariants(pipeline, load_annotations(source), config["test_policy"])
    manifest = pipeline_manifest(pipeline, source, config, images, decode_images=True)
    junit = output.with_suffix(".pytest.xml")
    junit.unlink(missing_ok=True)
    test_files = list(test_files or ["tests/test_data.py", "tests/test_dataset_pipeline.py"])
    env = dict(os.environ, PLANTVQA_CSV=str(Path(source).resolve()))
    command = [sys.executable, "-m", "pytest", *test_files, "-v", "--junitxml", str(junit.resolve())]
    completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    output.with_suffix(".pytest.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    suites = list(ET.parse(junit).getroot().iter("testsuite")) if junit.is_file() else []
    totals = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    tests_passed = completed.returncode == 0 and totals["tests"] > 0 and not any(totals[k] for k in ("failures", "errors", "skipped"))
    report.update(passed=tests_passed, data_manifest=manifest,
        synthetic_tests={"passed": tests_passed, "executed": True, "command": command, **totals,
                         "test_code_sha256": {name: sha256(root / name) for name in test_files}},
        real_data={"passed": True, "images_decoded": True, "invariants_checked": True})
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not tests_passed:
        raise RuntimeError(f"Data tests did not pass without skips; inspect {output}")
    return report


def require_validation(report, manifest):
    synthetic = report.get("synthetic_tests", {})
    real = report.get("real_data", {})
    if not (report.get("passed") is True and report.get("seed") == SEED
            and synthetic.get("passed") is True and synthetic.get("executed") is True
            and synthetic.get("tests", 0) > 0 and not any(synthetic.get(k, 0) for k in ("failures", "errors", "skipped"))
            and real.get("passed") is True and real.get("images_decoded") is True
            and real.get("invariants_checked") is True and report.get("data_manifest") == manifest):
        raise RuntimeError("Training blocked: absent, failed or stale real-data/test evidence; run --validate-only first")
    root = Path(__file__).resolve().parents[1]
    hashes = synthetic.get("test_code_sha256", {})
    expected = ("tests/test_data.py", "tests/test_dataset_pipeline.py")
    if any(hashes.get(name) != sha256(root / name) for name in expected):
        raise RuntimeError("Training blocked: data tests changed since validation")


def serialize_tokenizer(tokenizer):
    with tempfile.TemporaryDirectory() as directory:
        tokenizer.save_pretrained(directory)
        files = {p.name: p.read_bytes() for p in Path(directory).iterdir() if p.is_file()}
    if not files:
        raise ValueError("Tokenizer serialization produced no files")
    return {"name_or_path": str(tokenizer.name_or_path), "class": type(tokenizer).__name__, "files": files}


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def update_summary(path, config, values, *, require_existing=False, allow_update=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = []
    key = (config["level"], config["mode"], str(SEED))
    matches = [i for i, r in enumerate(rows) if (r.get("level"), r.get("mode"), r.get("seed")) == key]
    if len(matches) > 1 or (matches and not allow_update) or (require_existing and not matches):
        raise ValueError("Duplicate/missing experiment summary row")
    record = {"level": config["level"], "mode": config["mode"], "seed": SEED, **values}
    if matches:
        rows[matches[0]].update(record)
    else:
        rows.append(record)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", encoding="utf-8", newline="", delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fit(model, train_loader, val_loader, *, config, answer_to_idx, tokenizer,
        preprocessing, baseline, data_manifest, data_validation, device, resume=False):
    """Only train+val loaders accepted: test cannot select epochs/checkpoints."""
    require_validation(data_validation, data_manifest)
    device = torch.device(device)
    if getattr(model, "mode", config.mode) != config.mode:
        raise ValueError("Model mode differs from experiment config")
    settings = asdict(config)
    root = Path(config.output_dir)
    stem = f"{config.mode}_{config.level}"
    checkpoint_path = root / "models" / f"{stem}.pt"
    last_path = root / "models" / f"{stem}_last.pt"
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    marker = results / f"{stem}_run.json"
    if not resume and (checkpoint_path.exists() or last_path.exists() or marker.exists()):
        raise FileExistsError("Experiment already started/completed; refusing to repeat this configuration. Use --resume with its compatible _last.pt checkpoint")
    if resume and not last_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint missing: {last_path}")
    summary_path = results / "summary.csv"
    if not resume and summary_path.is_file():
        with summary_path.open(encoding="utf-8", newline="") as stream:
            if any((r.get("level"), r.get("mode"), r.get("seed")) == (config.level, config.mode, str(SEED)) for r in csv.DictReader(stream)):
                raise FileExistsError("Experiment already present in summary")
    if not resume:
        with marker.open("x", encoding="utf-8") as stream:
            json.dump({"config": settings, "seed": SEED, "started_utc": datetime.now(timezone.utc).isoformat()}, stream, indent=2)
    seed_everything()
    model.to(device)
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    tokenizer_metadata = serialize_tokenizer(tokenizer)
    best_accuracy, best_epoch, start_epoch, history = -math.inf, None, 1, []
    if resume:
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("schema_version") != 2 or state.get("config") != settings or state.get("mode") != config.mode or state.get("seed") != SEED or state.get("data_manifest") != data_manifest:
            raise ValueError("Resume checkpoint does not match config, seed, mode or dataset")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = state["epoch"] + 1
        best_accuracy, best_epoch, history = state["best_accuracy"], state["best_epoch"], list(state["history"])
        torch.set_rng_state(state["torch_rng_state"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        if device.type == "cuda" and state["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        if start_epoch > config.epochs:
            raise ValueError("Resume checkpoint already completed requested epochs")
    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device=device, scaler=scaler, max_grad_norm=config.max_grad_norm, accumulation_steps=config.gradient_accumulation_steps)
        validation = evaluate_loader(model, val_loader, answer_to_idx, device=device, baseline=baseline)
        if train_metrics["optimizer_steps"] > 0:
            scheduler.step()
        accuracy = validation["metrics"]["accuracy"]
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("Invalid validation accuracy")
        history.append({"epoch": epoch, "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
                        "val_loss_known": validation["metrics"]["loss_known_answers"], "val_accuracy": accuracy,
                        "val_unknown": validation["metrics"]["unknown"]})
        if accuracy > best_accuracy:  # Strict improvement: ties keep the earlier epoch.
            best_accuracy, best_epoch = accuracy, epoch
            save_checkpoint(checkpoint_path, {"schema_version": 1,
                "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "config": settings, "answer_to_idx": dict(answer_to_idx), "mode": config.mode,
                "tokenizer": tokenizer_metadata, "preprocessing": preprocessing, "epoch": epoch,
                "seed": SEED, "validation_metric": {"name": "exact_match_accuracy", "value": accuracy,
                    "total": validation["metrics"]["total"], "unknown": validation["metrics"]["unknown"]},
                "data_manifest": data_manifest, "data_validation": data_validation, "baseline": baseline,
                "torch_version": str(torch.__version__)})
        save_checkpoint(last_path, {"schema_version": 2, "config": settings, "mode": config.mode,
            "seed": SEED, "epoch": epoch, "best_accuracy": best_accuracy, "best_epoch": best_epoch,
            "history": history, "data_manifest": data_manifest,
            "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None})
        with (results / f"{stem}_history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    update_summary(summary_path, settings, {"epochs": config.epochs, "best_epoch": best_epoch,
        "val_accuracy": best_accuracy, "checkpoint": str(checkpoint_path)}, allow_update=False)
    return {"checkpoint": str(checkpoint_path), "best_epoch": best_epoch, "val_accuracy": best_accuracy, "history": history}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("mini", "class", "full"), default="class")
    parser.add_argument("--mode", choices=ABLATIONS, default="full")
    parser.add_argument("--data_dir", default=os.environ.get("PLANTVQA_DATA_DIR", "data/PlantVillageVQA"))
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--test-policy", choices=("preserve", "subset"), default="preserve")
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    source = args.csv or discover_csv(args.data_dir)
    config = TrainConfig(level=args.level, mode=args.mode, data_dir=str(Path(args.data_dir).resolve()),
        source_csv=str(source.resolve()), batch_size=args.batch_size, num_workers=args.num_workers,
        encoder_lr=args.encoder_lr, head_lr=args.head_lr, output_dir=args.output_dir, test_policy=args.test_policy,
        gradient_accumulation_steps=args.gradient_accumulation_steps)
    settings = asdict(config)
    report_path = args.validation_report or Path(config.output_dir) / "results" / f"data_validation_{config.level}_{config.test_policy}.json"
    if not args.validate_only and not report_path.is_file():
        parser.error(f"Missing {report_path}; run the same command with --validate-only first")
    seed_everything()
    preprocessing = preprocessing_metadata(settings)
    pipeline, images = build_pipeline(settings, source, preprocessing=preprocessing)
    if args.validate_only:
        report = validate_only(pipeline, source, settings, images, report_path)
        print(json.dumps(report, indent=2))
        return 0
    evidence = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = pipeline_manifest(pipeline, source, settings, images)
    require_validation(evidence, manifest)
    from .models import ViTBERT
    device = select_device(args.device)
    model = ViTBERT(len(pipeline.answer_vocab), config.mode, pretrained=True)
    if model.image_data_config is not None and digest_json(model.image_data_config) != digest_json(preprocessing["data_config"]):
        raise ValueError("Model and pipeline image preprocessing differ")
    baseline = fit_question_type_baseline(pipeline.splits["train"])
    result = fit(model, pipeline.loaders["train"], pipeline.loaders["val"], config=config, resume=args.resume,
        answer_to_idx=pipeline.answer_vocab, tokenizer=pipeline.loaders["train"].dataset.tokenizer,
        preprocessing=preprocessing, baseline=baseline, data_manifest=manifest, data_validation=evidence, device=device)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
