"""Synthetic smoke tests only. No real ViT/BERT download or full training run."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import csv
import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from src.config import SEED
from src.download import sha256
from src.evaluate import (baseline_predictions, compute_metrics, evaluate_loader,
    fit_question_type_baseline, load_checkpoint, restore_tokenizer, save_evaluation)
from src.train import (TrainConfig, build_optimizer, fit, require_validation,
    serialize_tokenizer, train_one_epoch, update_summary)


class TinyDataset(Dataset):
    def __init__(self, labels=(0, 1, 0, 1)):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"pixel_values": torch.tensor([float(index), 1.0, .5, -.5]),
                "input_ids": torch.tensor([1, 2, 0]), "attention_mask": torch.tensor([1, 1, 0]),
                "token_type_ids": torch.zeros(3, dtype=torch.long),
                "labels": torch.tensor(self.labels[index]), "image_id": f"image_{index}.jpg",
                "question_type": "kind_a" if index % 2 == 0 else "kind_b"}


class TinyModel(nn.Module):
    def __init__(self, frozen=False):
        super().__init__()
        self.mode = "frozen" if frozen else "full"
        self.image_encoder = nn.Sequential(nn.Linear(4, 4), nn.Dropout(.2))
        self.text_encoder = nn.Embedding(4, 4)
        self.image_projection = nn.Linear(4, 4)
        self.fusion = nn.Linear(4, 2)
        if frozen:
            self.image_encoder.requires_grad_(False)
            self.text_encoder.requires_grad_(False)

    def forward(self, pixel_values, input_ids, **kwargs):
        encoded = self.image_encoder(pixel_values) + self.text_encoder(input_ids)[:, 0]
        logits = self.fusion(torch.relu(self.image_projection(encoded)))
        self.output_dtype = logits.dtype
        self.grad_enabled = torch.is_grad_enabled()
        return logits


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(SEED)


@pytest.fixture
def tokenizer(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nleaf\n", encoding="utf-8")
    return BertTokenizer(vocab_file=str(vocab), do_lower_case=True)


def evidence(manifest):
    root = Path(__file__).resolve().parents[1]
    return {"passed": True, "seed": SEED, "data_manifest": manifest,
        "synthetic_tests": {"passed": True, "executed": True, "tests": 1,
            "failures": 0, "errors": 0, "skipped": 0,
            "test_code_sha256": {name: sha256(root / name) for name in ("tests/test_data.py", "tests/test_dataset_pipeline.py")}},
        "real_data": {"passed": True, "images_decoded": True, "invariants_checked": True}}


@pytest.mark.parametrize("frozen", [False, True])
def test_adamw_groups_cover_each_trainable_parameter_once(frozen):
    model = TinyModel(frozen)
    config = TrainConfig()
    optimizer = build_optimizer(model, config)
    assigned = [p for group in optimizer.param_groups for p in group["params"]]
    assert len({id(p) for p in assigned}) == len(assigned)
    assert {id(p) for p in assigned} == {id(p) for p in model.parameters() if p.requires_grad}
    for group in optimizer.param_groups:
        expected_lr = config.encoder_lr if group["group_name"].startswith("encoder") else config.head_lr
        assert group["lr"] == expected_lr
        if group["group_name"].endswith("no_decay"):
            assert group["weight_decay"] == 0
    assert any(g["group_name"].startswith("encoder") for g in optimizer.param_groups) == (not frozen)


def test_one_cpu_smoke_step_updates_parameters_without_amp():
    model = TinyModel()
    loader = DataLoader(TinyDataset(), batch_size=4)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = build_optimizer(model, TrainConfig())
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    result = train_one_epoch(model, loader, optimizer, device="cpu", scaler=scaler)
    assert result["total"] == 4 and 0 <= result["accuracy"] <= 1
    assert result["loss"] > 0
    assert model.output_dtype == torch.float32
    assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_unknown_train_labels_are_rejected():
    model = TinyModel()
    optimizer = build_optimizer(model, TrainConfig())
    with pytest.raises(ValueError, match="known"):
        train_one_epoch(model, DataLoader(TinyDataset((-1,)), batch_size=1), optimizer,
                        device="cpu", scaler=torch.amp.GradScaler("cuda", enabled=False))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable: float16 AMP smoke requires GPU")
def test_cuda_float16_amp_one_synthetic_batch():
    model = TinyModel().cuda()
    optimizer = build_optimizer(model, TrainConfig())
    result = train_one_epoch(model, DataLoader(TinyDataset(), batch_size=4), optimizer,
        device="cuda", scaler=torch.amp.GradScaler("cuda"))
    assert result["total"] == 4
    assert model.output_dtype == torch.float16


def test_unknown_labels_always_wrong_and_confusion_includes_them():
    metrics = compute_metrics([0, 1, -1, -1], [0, 0, 0, -1], ["a", "a", "b", "b"], {"No": 0, "Yes": 1})
    assert metrics["accuracy"] == metrics["exact_match_accuracy"] == .25
    assert metrics["total"] == 4 and metrics["unknown"] == 2
    assert metrics["by_question_type"]["b"]["accuracy"] == 0
    assert metrics["by_question_type"]["a"]["accuracy"] == .5
    assert sum(map(sum, metrics["confusion_matrix"])) == 4
    assert metrics["confusion_matrix"][-1][-1] == 1  # matching -1 still earns no credit
    assert metrics["confusion_labels"][-1] == "<UNKNOWN>"
    skipped = compute_metrics([0], [0], ["a"], {"No": 0, "Yes": 1}, confusion_limit=1)
    assert skipped["confusion_matrix"] is None


def test_baseline_train_only_ties_and_unseen_question_type():
    rows = [{"split": "train", "question_type": "a", "answer": answer} for answer in ("Yes", "No", "No", "Yes")]
    baseline = fit_question_type_baseline(rows)
    assert baseline["by_question_type"] == {"a": "No"}
    assert baseline["fallback"] == "No"
    assert baseline_predictions(["a", "unseen"], baseline, {"No": 0, "Yes": 1}) == [0, 0]
    with pytest.raises(ValueError, match="only"):
        fit_question_type_baseline(rows + [{"split": "test", "question_type": "a", "answer": "TEST_LEAK"}])


def test_evaluation_all_unknown_loss_is_none_and_no_grad(tmp_path):
    model = TinyModel()
    result = evaluate_loader(model, DataLoader(TinyDataset((-1, -1)), batch_size=2),
        {"No": 0, "Yes": 1}, device="cpu",
        baseline={"by_question_type": {}, "fallback": "No"})
    assert not model.training
    assert not model.grad_enabled
    assert result["metrics"]["accuracy"] == 0
    assert result["metrics"]["loss_known_answers"] is None
    assert result["metrics"]["unknown"] == 2
    assert result["baseline"]["accuracy"] == 0
    assert all(p.grad is None for p in model.parameters())
    save_evaluation(result, tmp_path, "smoke")
    assert (tmp_path / "smoke_metrics.json").is_file()
    assert (tmp_path / "smoke_question_types.csv").is_file()
    assert (tmp_path / "smoke_confusion_matrix.csv").is_file()
    with (tmp_path / "smoke_predictions.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2


@pytest.mark.parametrize("defect", ["absent", "manifest", "synthetic", "images", "skipped", "test_code"])
def test_gate_blocks_failed_or_stale_validation(defect):
    manifest = {"source_sha256": "synthetic_fixture"}
    report = evidence(manifest)
    if defect == "absent":
        report = {}
    elif defect == "manifest":
        report["data_manifest"] = {"source_sha256": "stale"}
    elif defect == "synthetic":
        report["synthetic_tests"]["executed"] = False
    elif defect == "images":
        report["real_data"]["images_decoded"] = False
    elif defect == "skipped":
        report["synthetic_tests"]["skipped"] = 1
    else:
        report["synthetic_tests"]["test_code_sha256"] = {}
    with pytest.raises(RuntimeError, match="blocked"):
        require_validation(report, manifest)


def test_fit_selects_val_best_checkpoint_keeps_earliest_tie_and_roundtrips(tmp_path, monkeypatch, tokenizer):
    import src.train as training
    config = TrainConfig(epochs=3, output_dir=str(tmp_path))
    model = TinyModel()
    train_loader, val_loader = object(), object()
    train_calls, val_calls = [], []
    scores = iter([.25, .75, .75])
    def fake_train(current_model, loader, *args, **kwargs):
        assert loader is train_loader
        train_calls.append(loader)
        with torch.no_grad():
            current_model.fusion.bias.fill_(len(train_calls))
        return {"loss": 1.0, "accuracy": .5, "total": 4, "optimizer_steps": 1}
    def fake_val(current_model, loader, *args, **kwargs):
        assert loader is val_loader
        val_calls.append(loader)
        return {"metrics": {"accuracy": next(scores), "loss_known_answers": 1.0, "total": 4, "unknown": 1}}
    monkeypatch.setattr(training, "train_one_epoch", fake_train)
    monkeypatch.setattr(training, "evaluate_loader", fake_val)
    manifest = {"source_sha256": "synthetic_fixture"}
    kwargs = dict(config=config, answer_to_idx={"No": 0, "Yes": 1}, tokenizer=tokenizer,
        preprocessing={"image_format": "normalized", "data_config": {"mean": [.5] * 3}},
        baseline={"by_question_type": {}, "fallback": "No"}, data_manifest=manifest,
        data_validation=evidence(manifest), device="cpu")
    result = fit(model, train_loader, val_loader, **kwargs)
    assert len(train_calls) == len(val_calls) == 3
    assert result["best_epoch"] == 2
    checkpoint = load_checkpoint(result["checkpoint"])
    assert checkpoint["validation_metric"]["value"] == .75
    assert checkpoint["epoch"] == 2 and checkpoint["seed"] == SEED
    assert torch.equal(checkpoint["model_state_dict"]["fusion.bias"], torch.full((2,), 2.0))
    assert torch.equal(model.fusion.bias, torch.full((2,), 3.0))
    restored = TinyModel()
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    restored_tokenizer = restore_tokenizer(checkpoint["tokenizer"], tmp_path / "restored_tokenizer")
    assert restored_tokenizer.get_vocab() == tokenizer.get_vocab()
    with (tmp_path / "results/summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and int(rows[0]["seed"]) == SEED
    update_summary(tmp_path / "results/summary.csv", asdict(config), {"test_accuracy": .5})
    with (tmp_path / "results/summary.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 1
    with pytest.raises(FileExistsError, match="repeat"):
        fit(model, train_loader, val_loader, **kwargs)


def test_fit_never_enters_training_without_data_evidence(tmp_path, tokenizer):
    with pytest.raises(RuntimeError, match="blocked"):
        fit(TinyModel(), object(), object(), config=TrainConfig(output_dir=str(tmp_path)),
            answer_to_idx={"No": 0, "Yes": 1}, tokenizer=tokenizer, preprocessing={},
            baseline={}, data_manifest={}, data_validation={}, device="cpu")
    assert not (tmp_path / "models").exists()


def test_tokenizer_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="Unsafe"):
        restore_tokenizer({"files": {"../outside.txt": b"bad"}}, tmp_path)


def test_scheduler_does_not_step_when_amp_skips_every_optimizer_step(tmp_path, monkeypatch, tokenizer):
    import src.train as training
    config = TrainConfig(epochs=1, output_dir=str(tmp_path))
    model = TinyModel()
    manifest = {"source_sha256": "synthetic_fixture"}
    monkeypatch.setattr(training, "train_one_epoch", lambda *a, **k: {"loss": 1.0, "accuracy": .5, "total": 4, "optimizer_steps": 0})
    monkeypatch.setattr(training, "evaluate_loader", lambda *a, **k: {"metrics": {"accuracy": .25, "loss_known_answers": 1.0, "total": 4, "unknown": 0}})
    stepped = []
    original_factory = torch.optim.lr_scheduler.LambdaLR
    def make_scheduler(*args, **kwargs):
        scheduler = original_factory(*args, **kwargs)
        original_step = scheduler.step
        def observed_step(*step_args, **step_kwargs):
            stepped.append(True)
            return original_step(*step_args, **step_kwargs)
        scheduler.step = observed_step
        return scheduler
    monkeypatch.setattr(training.torch.optim.lr_scheduler, "LambdaLR", make_scheduler)
    kwargs = dict(config=config, answer_to_idx={"No": 0, "Yes": 1}, tokenizer=tokenizer,
        preprocessing={"image_format": "normalized"}, baseline={"by_question_type": {}, "fallback": "No"},
        data_manifest=manifest, data_validation=evidence(manifest), device="cpu")
    training.fit(model, object(), object(), **kwargs)
    assert stepped == []
