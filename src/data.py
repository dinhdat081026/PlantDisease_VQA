"""Image-group data pipeline. No source file writes and no model training.

Test policy is explicit: 'preserve' retains all supplied factoid test rows;
'subset' samples a factoid evaluation view but NEVER changes official membership.
The original annotation file is never modified. Validation size is ceil(15% of
all selected images), drawn exclusively from original train. Sizes are counted
before the train-only rare-answer removal; report both stages.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import IMAGE_MODEL, SEED, TEXT_MODEL
from .download import image_key, index_images, resolve_image
from .eda import FACTOID_TYPES

UNKNOWN_LABEL = -1
SUBSET_TARGETS = {"mini": 1000, "class": 8000}
SPECIES = ("Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", "Peach",
           "Pepper bell", "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato")


def _rows(data):
    if hasattr(data, "to_dict"):
        data = data.to_dict(orient="records")
    return [dict(row) for row in data]


def validate_records(rows):
    if not rows:
        raise ValueError("Expected nonempty QA records")
    for row in rows:
        for key in ("image_id", "question", "answer"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing or invalid {key}")


def filter_factoid(data):
    rows = [r for r in _rows(data) if r.get("question_type") in FACTOID_TYPES]
    if any(len(r.get("answer", "").split()) > 3 for r in rows):
        raise ValueError("Factoid answer exceeds three words; investigate source rather than silently dropping it")
    if rows:
        validate_records(rows)
    return rows


def _image_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[image_key(row["image_id"])].append(row)
    return dict(groups)


def _check_official_splits(rows):
    groups = _image_groups(rows)
    for key, group in groups.items():
        labels = {r.get("split") for r in group}
        if not labels <= {"train", "test"}:
            raise ValueError(f"Expected original train/test only: {key}: {labels}")
        if len(labels) != 1:
            raise ValueError(f"Image leakage in source: {key}")
    return groups


def infer_species(data, species_by_image=None):
    """Metadata for strata, never an answer vocabulary. Negative QA proves no host.

    Accept explicit metadata when available. Otherwise use exact species answers
    or a positive species-identification question mentioning one known crop.
    Conflicting positive evidence is an error. Missing evidence maps to other.
    """
    rows = _rows(data)
    explicit = {image_key(k): str(v) for k, v in (species_by_image or {}).items()}
    canonical = {s.casefold(): s for s in SPECIES}
    patterns = {s: re.compile(r"(?<![a-z])" + (r"(?:pepper(?: bell)?|bell pepper)" if s == "Pepper bell" else re.escape(s.casefold())) + r"(?![a-z])") for s in SPECIES}
    result = {}
    for key, group in _image_groups(rows).items():
        if key in explicit:
            result[key] = explicit[key]
            continue
        evidence = {str(r["species"]).strip() for r in group if r.get("species")}
        for row in group:
            if row.get("question_type") != "Plant Species Identification":
                continue
            answer = row["answer"].strip().casefold()
            if answer in canonical:
                evidence.add(canonical[answer])
            elif answer == "yes":
                matches = {s for s, pattern in patterns.items() if pattern.search(row["question"].casefold())}
                if len(matches) == 1:
                    evidence.update(matches)
        if len(evidence) > 1:
            raise ValueError(f"Conflicting species evidence for {key}: {sorted(evidence)}")
        result[key] = next(iter(evidence), "other")
    return result


def _strata(ids, species, rare_min_images):
    if rare_min_images < 1:
        raise ValueError("rare_min_images must be positive")
    counts = Counter(species.get(i, "other") for i in ids)
    return {i: species.get(i, "other") if counts[species.get(i, "other")] >= rare_min_images else "other" for i in ids}


def _sample_images(ids, count, strata, purpose):
    """Exact proportional quotas (largest remainder), stable seeded hash ranking.

    No row-level sampling. Sorting plus hash ranking makes selections invariant
    to source row order, Python hash randomization and DataFrame index values.
    """
    ids = sorted(set(ids))
    if not 0 <= count <= len(ids):
        raise ValueError(f"Requested {count} images from {len(ids)} available")
    if count == 0:
        return set()
    groups = defaultdict(list)
    for key in ids:
        groups[strata[key]].append(key)
    quotas = {s: len(g) * count // len(ids) for s, g in groups.items()}
    remaining = count - sum(quotas.values())
    order = sorted(groups, key=lambda s: (-(len(groups[s]) * count % len(ids)), s))
    for s in order[:remaining]:
        quotas[s] += 1
    selected = set()
    for s, group in sorted(groups.items()):
        def rank(key):
            return hashlib.sha256(f"{SEED}:{purpose}:{s}:{key}".encode()).digest(), key
        selected.update(sorted(group, key=rank)[:quotas[s]])
    return selected


def build_subset(data, level="full", *, species_by_image=None, rare_min_images=5,
                 target_images=None, test_policy="preserve"):
    """Return copied factoid rows; keep all QA for each selected image.

    With preserve policy, the target is a TOTAL image budget and must accommodate
    all original factoid test images. Incompatible mini/class budgets raise an
    explicit error rather than discarding test images. With subset policy,
    sampling is stratified jointly by original split and species.
    """
    if level not in {"mini", "class", "full"}:
        raise ValueError("level must be mini, class or full")
    if test_policy not in {"preserve", "subset"}:
        raise ValueError("test_policy must be preserve or subset")
    source = _rows(data)
    validate_records(source)
    _check_official_splits(source)
    rows = filter_factoid(source)
    if not rows:
        raise ValueError("No factoid QA available")
    groups = _check_official_splits(rows)
    species = infer_species(source, species_by_image)
    strata = _strata(groups, species, rare_min_images)
    if level == "full":
        if target_images is not None:
            raise ValueError("full cannot be given a target_images budget")
        return rows
    count = SUBSET_TARGETS[level] if target_images is None else target_images
    if not isinstance(count, int) or count < 1:
        raise ValueError("target_images must be a positive integer")
    count = min(count, len(groups))
    if test_policy == "subset":
        joint = {i: groups[i][0]["split"] + ":" + strata[i] for i in groups}
        chosen = _sample_images(groups, count, joint, "subset:" + level)
    else:
        test_ids = {i for i, g in groups.items() if g[0]["split"] == "test"}
        train_ids = set(groups) - test_ids
        if count <= len(test_ids):
            raise ValueError(f"{level} budget {count} cannot preserve {len(test_ids)} original factoid test images and retain train; explicitly choose test_policy='subset' for a sampled evaluation view")
        chosen = test_ids | _sample_images(train_ids, count - len(test_ids), strata, "subset:" + level)
    return [r for r in rows if image_key(r["image_id"]) in chosen]


def split_by_image(data, *, validation_fraction=0.15, species_by_image=None, rare_min_images=5):
    rows = _rows(data)
    validate_records(rows)
    groups = _check_official_splits(rows)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    train_ids = {i for i, g in groups.items() if g[0]["split"] == "train"}
    n_val = math.ceil(validation_fraction * len(groups))
    if not 0 < n_val < len(train_ids):
        raise ValueError(f"Validation requires {n_val} images from {len(train_ids)} original train images; insufficient images to leave training nonempty")
    species = infer_species(rows, species_by_image)
    strata = _strata(train_ids, species, rare_min_images)
    validation_ids = _sample_images(train_ids, n_val, strata, "validation")
    result = {"train": [], "val": [], "test": []}
    for row in rows:
        if row["split"] == "test":
            result["test"].append(row)
        elif image_key(row["image_id"]) in validation_ids:
            result["val"].append({**row, "source_split": "train", "split": "val"})
        else:
            result["train"].append(row)
    return result


def build_answer_vocab(train_rows, *, min_count=5):
    rows = _rows(train_rows)
    validate_records(rows)
    if any(r.get("split", "train") != "train" for r in rows):
        raise ValueError("Vocabulary must receive only final training rows")
    if min_count < 1:
        raise ValueError("min_count must be positive")
    counts = Counter(r["answer"] for r in rows)
    return {answer: index for index, answer in enumerate(sorted(a for a, n in counts.items() if n >= min_count))}


def encode_answer(answer, answer_vocab):
    return answer_vocab.get(answer, UNKNOWN_LABEL)


def _validate_vocab(vocab):
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("Vocabulary labels must be contiguous nonnegative integers")


class PlantVQADataset(Dataset):
    def __init__(self, rows, *, answer_vocab, tokenizer, image_index=None,
                 data_dir=None, transform=None, max_length=64, is_training=False):
        self.rows = _rows(rows)
        if self.rows:
            validate_records(self.rows)
        self.answer_vocab = dict(answer_vocab)
        _validate_vocab(self.answer_vocab)
        self.tokenizer = tokenizer
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.max_length = max_length
        if image_index is None and data_dir is None:
            raise ValueError("Provide image_index or data_dir")
        self.image_index = image_index if image_index is not None else index_images(data_dir)
        if transform is None:
            from timm.data import create_transform, resolve_data_config
            from timm.models import get_pretrained_cfg
            config = get_pretrained_cfg(IMAGE_MODEL)
            if config is None:
                raise ValueError(f"No preprocessing config for {IMAGE_MODEL}")
            # Metadata only: do not construct ViT or download its weights.
            data_config = resolve_data_config(pretrained_cfg=config.to_dict())
            transform = create_transform(**data_config, is_training=is_training)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        row = self.rows[index]
        path = resolve_image(row["image_id"], self.image_index)
        with Image.open(path) as image:
            pixels = self.transform(image.convert("RGB"))
        if not isinstance(pixels, torch.Tensor) or pixels.shape != (3, 224, 224):
            raise ValueError("ViT transform must return a (3,224,224) tensor")
        pixels = pixels.to(dtype=torch.float32)
        if not torch.isfinite(pixels).all():
            raise ValueError("Nonfinite image tensor")
        tokens = self.tokenizer(row["question"], truncation=True, padding="max_length",
                                max_length=self.max_length, return_tensors="pt")
        result = {"pixel_values": pixels,
                  "labels": torch.tensor(encode_answer(row["answer"], self.answer_vocab), dtype=torch.long),
                  "image_id": row["image_id"], "question_type": row["question_type"]}
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key not in tokens:
                if key == "token_type_ids":
                    result[key] = torch.zeros(self.max_length, dtype=torch.long)
                    continue
                raise ValueError(f"Tokenizer missing {key}")
            value = torch.as_tensor(tokens[key], dtype=torch.long)
            if value.shape != (1, self.max_length):
                raise ValueError(f"Tokenizer {key} must have shape (1,max_length)")
            result[key] = value[0]
        return result


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class DataPipeline:
    loaders: dict
    splits: dict
    answer_vocab: dict
    report: dict


def classification_accuracy(predictions, labels):
    """Every unknown label counts as wrong, including a predicted -1. No masking denominator."""
    if predictions.shape != labels.shape or labels.numel() == 0:
        raise ValueError("Expected nonempty predictions and labels with equal shape")
    return ((predictions == labels) & (labels >= 0)).sum().item() / labels.numel()


def create_dataloaders(data, *, data_dir=None, image_index=None, level="full",
                       test_policy="preserve", species_by_image=None, rare_min_images=5,
                       target_images=None, batch_size=16, num_workers=0, max_length=64,
                       tokenizer=None, train_transform=None, eval_transform=None,
                       pin_memory=False):
    if batch_size < 1 or num_workers < 0:
        raise ValueError("Invalid batch_size or num_workers")
    # Seed parent-process augmentations as well as loader shuffle and workers.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    source = _rows(data)
    species = infer_species(source, species_by_image)
    subset = build_subset(source, level, test_policy=test_policy, species_by_image=species,
                          rare_min_images=rare_min_images, target_images=target_images)
    splits = split_by_image(subset, species_by_image=species, rare_min_images=rare_min_images)
    before = {s: {"qa_pairs": len(rows), "images": len(_image_groups(rows))} for s, rows in splits.items()}
    vocab = build_answer_vocab(splits["train"])
    if not vocab:
        raise ValueError("No answers occur at least five times in final train")
    splits["train"] = [r for r in splits["train"] if r["answer"] in vocab]
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    if image_index is None:
        if data_dir is None:
            raise ValueError("Provide data_dir or image_index")
        image_index = index_images(data_dir)
    missing = [i for i in _image_groups(subset) if i not in image_index]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} selected images; first: {missing[:5]}")
    loaders = {}
    for name, rows in splits.items():
        dataset = PlantVQADataset(rows, answer_vocab=vocab, tokenizer=tokenizer,
            image_index=image_index, max_length=max_length, is_training=name == "train",
            transform=train_transform if name == "train" else eval_transform)
        generator = torch.Generator().manual_seed(SEED)
        loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=name == "train",
            num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=_seed_worker,
            generator=generator, drop_last=False)
    report = {"level": level, "test_policy": test_policy, "seed": SEED,
              "validation_fraction_of_total": 0.15, "vocabulary_size": len(vocab),
              "before_rare_answer_removal": before,
              "after_rare_answer_removal": {s: {"qa_pairs": len(rows), "images": len(_image_groups(rows))} for s, rows in splits.items()},
              "removed_training_qa": before["train"]["qa_pairs"] - len(splits["train"]),
              "unknown_answers": {s: sum(r["answer"] not in vocab for r in rows) for s, rows in splits.items()},
              "species_unknown_images": sum(species.get(i, "other") == "other" for i in _image_groups(subset)),
              "test_note": "All supplied factoid test rows preserved" if test_policy == "preserve" else "Sampled factoid evaluation view; official train/test membership preserved"}
    return DataPipeline(loaders=loaders, splits=splits, answer_vocab=vocab, report=report)
