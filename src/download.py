"""Discover the Hub tree and extract archives without trusting paths from the card."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

REPO_ID = "SyedNazmusSakib/PlantVillageVQA"
REQUIRED_COLUMNS = ("image_id", "question_type", "question", "answer", "image_path", "split")
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive, destination):
    """Validate all members first; reject traversal, links and overwrites."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            name = PurePosixPath(member.filename)
            target = (destination / member.filename).resolve()
            if (name.is_absolute() or ".." in name.parts or "\\" in member.filename
                    or ":" in member.filename or not target.is_relative_to(destination)
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            if target.exists() and not member.is_dir():
                raise FileExistsError(target)
        zipped.extractall(destination)


def discover_csv(data_dir):
    candidates = []
    for path in sorted(Path(data_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                fields = csv.DictReader(stream).fieldnames or []
            if set(REQUIRED_COLUMNS).issubset(fields):
                candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f"Expected one source annotation CSV, found {candidates}; supply --csv explicitly")
    return candidates[0]


def image_key(value):
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    path = Path(name)
    return path.stem if path.suffix.lower() in IMAGE_EXTENSIONS else name


def index_images(data_dir):
    index = {}
    for path in sorted(Path(data_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            key = image_key(path.name)
            if key in index:
                raise ValueError(f"Ambiguous image_id {key}: {index[key]} and {path}")
            index[key] = path.resolve()
    return index


def resolve_image(image_id, image_index):
    try:
        return image_index[image_key(image_id)]
    except KeyError as exc:
        raise FileNotFoundError(f"No image for image_id={image_id}") from exc


def download_dataset(data_dir, revision="main"):
    from huggingface_hub import HfApi, snapshot_download
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    commit = api.dataset_info(REPO_ID, revision=revision).sha
    tree = api.list_repo_files(REPO_ID, repo_type="dataset", revision=commit)
    # Discover paths from the actual tree, not advertised folder names.
    selected = [p for p in tree if Path(p).suffix.lower() in IMAGE_EXTENSIONS | {".zip", ".csv", ".json", ".md"}]
    if not selected:
        raise ValueError("No supported dataset assets in repository tree")
    snapshot = Path(snapshot_download(REPO_ID, repo_type="dataset", revision=commit, allow_patterns=selected))
    target = data_dir / commit
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing source: {target}")
    with tempfile.TemporaryDirectory(dir=data_dir) as temporary:
        staging = Path(temporary) / "source"
        staging.mkdir()
        hashes = {}
        for relative in selected:
            source = snapshot / relative
            hashes[relative] = sha256(source)
            if source.suffix.lower() == ".zip":
                safe_extract(source, staging)
            else:
                dest = staging / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    raise FileExistsError(dest)
                shutil.copyfile(source, dest)
        csv_path = discover_csv(staging)
        images = index_images(staging)
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            ids = {r["image_id"] for r in csv.DictReader(stream)}
        for image_id in ids:
            resolve_image(image_id, images)
        manifest = {"repo_id": REPO_ID, "revision": commit, "repo_files": tree,
                    "download_sha256": hashes, "csv": str(csv_path.relative_to(staging)),
                    "csv_sha256": sha256(csv_path), "image_files": len(images),
                    "referenced_images": len(ids), "images_without_qa": len(set(images) - {image_key(i) for i in ids})}
        (staging / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        shutil.move(str(staging), str(target))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=os.environ.get("PLANTVQA_DATA_DIR", "data/PlantVillageVQA"))
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    print(download_dataset(args.data_dir, args.revision))


if __name__ == "__main__":
    main()
