"""Synthetic invariants; real-source counts are checked by src.verify_numbers."""
import copy
import csv
import stat
import zipfile

import pytest

from src.download import (REQUIRED_COLUMNS, discover_csv, index_images, resolve_image,
                          safe_extract, sha256)
from src.eda import (FACTOID_TYPES, QUESTION_TYPES, analyze, filter_factoid,
                     load_annotations, percentile, split_overlap)
from src.verify_numbers import verify


def record(**overrides):
    row = dict(image_id="image_1.JPG", question_type="Detailed Verification",
               question="Is this diseased?", answer="Yes", image_path="wrong/train/image_1.jpg", split="train")
    row.update(overrides)
    return row


def write_csv(path, rows, fields=REQUIRED_COLUMNS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("extension", ["jpg", "JPG", "jpeg", "png"])
def test_find_image_by_id_not_stale_annotation_path(tmp_path, extension):
    image = tmp_path / "unexpected" / f"image_1.{extension}"
    image.parent.mkdir()
    image.write_bytes(b"fixture-not-an-image")
    index = index_images(tmp_path)
    assert resolve_image("image_1.JPG", index) == image.resolve()
    assert resolve_image("image_1", index) == image.resolve()
    with pytest.raises(FileNotFoundError):
        resolve_image("unknown.jpg", index)


def test_ambiguous_image_id_is_not_silently_selected(tmp_path):
    (tmp_path / "image_1.jpg").touch()
    (tmp_path / "image_1.png").touch()
    with pytest.raises(ValueError, match="Ambiguous"):
        index_images(tmp_path)


def test_csv_discovery_schema_and_source_preservation(tmp_path):
    source = tmp_path / "arbitrary" / "labels.csv"
    rows = [record(question="A comma, a quote \" and a newline\n?"), record(image_id="image_2.png", split="test")]
    write_csv(source, rows)
    before = sha256(source)
    assert discover_csv(tmp_path) == source
    loaded = load_annotations(source)
    assert loaded == rows
    original = copy.deepcopy(loaded)
    analyze(loaded)
    verify(loaded)
    assert loaded == original
    assert sha256(source) == before
    second = tmp_path / "another.csv"
    write_csv(second, rows)
    with pytest.raises(ValueError, match="Expected one"):
        discover_csv(tmp_path)


@pytest.mark.parametrize("field,value", [("answer", ""), ("question_type", "unknown"), ("split", "invalid")])
def test_bad_csv_rejected(tmp_path, field, value):
    path = tmp_path / "bad.csv"
    write_csv(path, [record(**{field: value})])
    with pytest.raises(ValueError):
        load_annotations(path)


def test_six_columns_required(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("image_id,question,answer\nx,q,a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="six columns"):
        load_annotations(path)


def test_exact_four_types_not_answer_length_heuristic():
    rows = [record(question_type=t, answer="Yes") for t in QUESTION_TYPES]
    original = copy.deepcopy(rows)
    kept = filter_factoid(rows)
    assert len(kept) == 4
    assert {r["question_type"] for r in kept} == FACTOID_TYPES
    assert rows == original
    assert "Existence & Sanity Check" not in {r["question_type"] for r in kept}


def test_long_factoid_answer_is_reported_not_silently_deleted():
    rows = [record(answer="one two three four")]
    assert len(filter_factoid(rows)) == 1
    report = verify(rows)
    assert not report["invariants"]["factoid_at_most_three_words"]


def test_overlap_is_detected_without_repartitioning():
    rows = [record(), record(image_id="image_1.jpg", split="test")]
    assert split_overlap(rows) == {"test/train": ["image_1"]}
    assert not verify(rows)["invariants"]["no_image_overlap"]


def test_statistics_and_linear_percentiles():
    rows = [record(answer=a) for a in ("Yes", "No", "two words", "one two three four")]
    result = analyze(rows)["raw"]
    assert result["qa_pairs"] == 4
    assert result["images"] == 1
    assert result["unique_answers"] == 4
    assert result["median_answer_words"] == 1.5
    assert result["p95_answer_words"] == pytest.approx(3.7)
    assert percentile([], .95) is None


def test_reference_mismatch_is_explicit_and_does_not_fix_data():
    rows = [record()]
    report = verify(rows)
    assert not report["passed"]
    assert report["checks"][0]["actual"] == 1
    assert report["checks"][0]["expected"] != 1
    assert report["discrepancy_guidance"]
    assert len(rows) == 1


@pytest.mark.parametrize("name", ["../escape.txt", "/absolute.txt", "C:/escape.txt", "folder\\escape.txt"])
def test_unsafe_zip_rejected_before_extraction(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("safe.txt", "safe")
        zipped.writestr(name, "unsafe")
    target = tmp_path / "extracted"
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(archive, target)
    assert not (target / "safe.txt").exists()


def test_zip_symlink_and_overwrite_rejected(tmp_path):
    archive = tmp_path / "link.zip"
    link = zipfile.ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(link, "outside")
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "extracted")
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("nested/source.csv", "original")
    target = tmp_path / "extracted"
    safe_extract(archive, target)
    with pytest.raises(FileExistsError):
        safe_extract(archive, target)
    assert (target / "nested/source.csv").read_text() == "original"
