"""Offline pipeline tests. Real CSV checks skip only when no source is available.

Tests use a tiny LOCAL BERT vocabulary, never a pretrained-model download.
The sampled test policy is always explicit in tests; production defaults preserve
all original factoid test rows. Reference drift is reported separately from
algorithm invariants, not concealed by manipulating rows.
"""
from collections import Counter
import copy
import csv
import math
import os
from pathlib import Path
import warnings

import pytest
import torch
from PIL import Image
from transformers import BertTokenizer

from src.config import SEED
from src.data import (FACTOID_TYPES, UNKNOWN_LABEL, PlantVQADataset,
    build_answer_vocab, build_subset, classification_accuracy, create_dataloaders,
    encode_answer, filter_factoid, infer_species, split_by_image)
from src.download import image_key, index_images


def qa(image, *, split="train", answer="Yes", species="Tomato", kind="Detailed Verification", question="Is the leaf healthy?"):
    return dict(image_id=f"image_{image}.JPG", split=split, answer=answer,
                species=species, question_type=kind, question=question,
                image_path="stale/path.jpg")


@pytest.fixture
def rows():
    result = []
    for i in range(100):
        for j in range(3):
            result.append(qa(i, split="train" if i < 80 else "test",
                             species="Tomato" if i % 2 else "Apple", answer="Yes" if j else "No"))
    return result


def ids(rows):
    return {image_key(r["image_id"]) for r in rows}


@pytest.fixture
def tokenizer(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nis\nthe\nleaf\nhealthy\n?\n", encoding="utf-8")
    return BertTokenizer(vocab_file=str(vocab), do_lower_case=True)


def test_filter_exact_four_categories_and_no_mutation(rows):
    raw = rows + [qa(101, kind="Existence & Sanity Check"), qa(102, kind="Causal Reasoning")]
    for i, category in enumerate(sorted(FACTOID_TYPES)):
        raw.append(qa(200 + i, kind=category))
    before = copy.deepcopy(raw)
    result = filter_factoid(raw)
    assert {r["question_type"] for r in result} == FACTOID_TYPES
    assert len(result) == len(rows) + 4
    assert raw == before
    with pytest.raises(ValueError, match="exceeds three"):
        filter_factoid([qa(1, answer="one two three four")])


def test_split_disjoint_complete_test_preserved_validation_from_train(rows):
    original = copy.deepcopy(rows)
    split = split_by_image(rows)
    train_ids, val_ids, test_ids = (ids(split[s]) for s in ("train", "val", "test"))
    assert not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)
    assert train_ids | val_ids | test_ids == ids(rows)
    assert len(val_ids) == math.ceil(.15 * len(ids(rows)))
    assert split["test"] == [r for r in rows if r["split"] == "test"]
    assert val_ids <= ids([r for r in rows if r["split"] == "train"])
    assert len(split["train"]) + len(split["val"]) + len(split["test"]) == len(rows)
    assert rows == original
    for part in split.values():
        assert all(sum(r["image_id"] == key for r in part) == 3 for key in {r["image_id"] for r in part})


def test_deterministic_seed_and_input_order_independence(rows):
    first = split_by_image(rows)
    assert first == split_by_image(rows)
    reversed_split = split_by_image(list(reversed(rows)))
    assert {s: ids(v) for s, v in first.items()} == {s: ids(v) for s, v in reversed_split.items()}
    a = build_subset(rows, "mini", target_images=40, test_policy="subset")
    b = build_subset(list(reversed(rows)), "mini", target_images=40, test_policy="subset")
    assert ids(a) == ids(b)
    assert isinstance(SEED, int)


def test_stratified_selection_and_rare_species(rows):
    subset = build_subset(rows, "class", target_images=40, test_policy="subset")
    unique = {image_key(r["image_id"]): r["species"] for r in subset}
    assert Counter(unique.values()) == {"Apple": 20, "Tomato": 20}
    # Tiny strata and grouped 'other' must not crash stratified selection.
    rare = rows + [qa(1001, species="rare_a"), qa(1002, species="rare_b")]
    result = build_subset(rare, "mini", target_images=30, test_policy="subset", rare_min_images=5)
    assert len(ids(result)) == 30
    assert len(result) == sum(1 for r in rare if image_key(r["image_id"]) in ids(result))


def test_species_evidence_never_infers_from_negative_question():
    examples = [qa(1, kind="Plant Species Identification", answer="No", question="Is this an Apple leaf?"),
                qa(2, kind="Plant Species Identification", answer="Yes", question="Is this a Tomato leaf?"),
                qa(3, kind="Plant Species Identification", answer="Pepper bell")]
    for row in examples:
        row.pop("species")
    result = infer_species(examples)
    assert result == {"image_1": "other", "image_2": "Tomato", "image_3": "Pepper bell"}
    assert infer_species(examples, {"image_1.JPG": "Corn"})["image_1"] == "Corn"


def test_default_subset_preserves_every_test_row_and_membership(rows):
    subset = build_subset(rows, "mini", target_images=40)
    assert [r for r in subset if r["split"] == "test"] == [r for r in rows if r["split"] == "test"]
    with pytest.raises(ValueError, match="cannot preserve"):
        build_subset(rows, "mini", target_images=10)
    view = build_subset(rows, "mini", target_images=40, test_policy="subset")
    original_split = {r["image_id"]: r["split"] for r in rows}
    assert all(r["split"] == original_split[r["image_id"]] for r in view)


def test_source_leakage_and_impossible_validation_rejected(rows):
    with pytest.raises(ValueError, match="leakage"):
        split_by_image(rows + [qa(0, split="test")])
    with pytest.raises(ValueError, match="insufficient"):
        split_by_image([qa(0)] + [qa(i, split="test") for i in range(1, 20)])
    with pytest.raises(ValueError):
        split_by_image(rows, validation_fraction=1)


def test_vocabulary_train_only_frequency_threshold_unknown_accuracy():
    train = [qa(i, answer="frequent") for i in range(5)] + [qa(i + 10, answer="rare") for i in range(4)]
    original = copy.deepcopy(train)
    vocab = build_answer_vocab(train)
    assert vocab == {"frequent": 0}
    assert train == original
    assert encode_answer("rare", vocab) == UNKNOWN_LABEL == -1
    assert encode_answer("test_only", vocab) == -1
    with pytest.raises(ValueError, match="only final training"):
        build_answer_vocab(train + [qa(99, split="test", answer="test_only")])
    # Even predicting -1 must not receive credit; unknowns stay in denominator.
    assert classification_accuracy(torch.tensor([0, 0, -1]), torch.tensor([0, -1, -1])) == pytest.approx(1 / 3)


@pytest.mark.parametrize("extension", ["jpg", "JPG", "jpeg", "png"])
def test_dataset_real_image_vit_and_local_bert_tensor_contract(tmp_path, tokenizer, extension):
    folder = tmp_path / "actual_images"
    folder.mkdir()
    Image.new("L", (280, 250), color=128).save(folder / f"image_1.{extension}")
    dataset = PlantVQADataset([qa(1)], answer_vocab={"Yes": 0}, tokenizer=tokenizer,
        image_index=index_images(folder), max_length=16)
    item = dataset[0]
    assert item["pixel_values"].shape == (3, 224, 224)
    assert item["pixel_values"].dtype == torch.float32
    assert torch.isfinite(item["pixel_values"]).all()
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        assert item[key].shape == (16,)
        assert item[key].dtype == torch.long
    assert item["input_ids"][0] == tokenizer.cls_token_id
    assert item["labels"].shape == torch.Size([])
    assert item["labels"].dtype == torch.long
    assert item["labels"].item() == 0
    unknown = PlantVQADataset([qa(1, answer="unseen")], answer_vocab={"Yes": 0}, tokenizer=tokenizer,
        image_index=index_images(folder), max_length=16)
    assert unknown[0]["labels"].item() == -1


def test_dataloaders_remove_rare_train_only_and_keep_unknown_eval(tmp_path, rows, tokenizer):
    # Each image has two common answers. A unique training-only answer is rare.
    rows = copy.deepcopy(rows)
    rows.append({**rows[0], "answer": "train_rare"})
    for row in rows:
        if row["split"] == "test":
            row["answer"] = "test_only"
    for key in ids(rows):
        Image.new("RGB", (224, 224), color="green").save(tmp_path / f"{key}.jpg")
    pipeline = create_dataloaders(rows, data_dir=tmp_path, tokenizer=tokenizer, batch_size=8)
    assert "test_only" not in pipeline.answer_vocab
    assert "train_rare" not in pipeline.answer_vocab
    assert all(r["answer"] in pipeline.answer_vocab for r in pipeline.splits["train"])
    assert pipeline.splits["test"] == [r for r in rows if r["split"] == "test"]
    assert pipeline.report["removed_training_qa"] in (0, 1)  # rare image may land in val
    for name, loader in pipeline.loaders.items():
        assert len(loader.dataset) == len(pipeline.splits[name])
        batch = next(iter(loader))
        assert batch["pixel_values"].shape[1:] == (3, 224, 224)
        assert batch["input_ids"].shape[1:] == (64,)
        assert batch["labels"].dtype == torch.long
        if name == "test":
            assert (batch["labels"] == -1).all()
    assert pipeline.report["seed"] == SEED


@pytest.fixture(scope="module")
def actual_rows():
    root = Path(__file__).resolve().parents[1]
    path = Path(os.environ.get("PLANTVQA_CSV", root / "data/source_inspection/PlantVillageVQA.csv"))
    if not path.is_file():
        pytest.skip("Real CSV absent: set PLANTVQA_CSV to validate source counts")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_actual_full_reference_counts_and_original_test(actual_rows):
    assert len(actual_rows) == 193609
    assert len(ids(actual_rows)) == 50073
    full = build_subset(actual_rows, "full")
    assert (len(full), len(ids(full))) == (96825, 45206)
    split = split_by_image(full, species_by_image=infer_species(actual_rows))
    assert len(ids(split["val"])) == math.ceil(.15 * len(ids(full)))
    assert split["test"] == [r for r in full if r["split"] == "test"]
    assert not (ids(split["train"]) & ids(split["val"]) or ids(split["train"]) & ids(split["test"]) or ids(split["val"]) & ids(split["test"]))


@pytest.mark.parametrize("level,target,reference_images,reference_qa", [
    ("mini", 1000, 995, 2102), ("class", 8000, 7999, 17064)])
def test_actual_subset_budget_and_reference_discrepancy(actual_rows, level, target, reference_images, reference_qa):
    # Entire-test preservation cannot produce the smaller documented totals.
    with pytest.raises(ValueError, match="cannot preserve"):
        build_subset(actual_rows, level)
    # Explicit sampled evaluation view: exact target quotas, unchanged source splits.
    view = build_subset(actual_rows, level, test_policy="subset")
    assert len(ids(view)) == target
    source = {r["image_id"]: r["split"] for r in actual_rows}
    assert all(source[r["image_id"]] == r["split"] for r in view)
    selected_ids = ids(view)
    assert len(view) == sum(r["question_type"] in FACTOID_TYPES and image_key(r["image_id"]) in selected_ids for r in actual_rows)
    if (len(ids(view)), len(view)) != (reference_images, reference_qa):
        warnings.warn(f"{level}: observed {len(ids(view))} images/{len(view)} QA differs from documented {reference_images}/{reference_qa}; exact quotas and sampled-test policy differ from an undocumented reference sampler. No row repairs performed.", UserWarning)
