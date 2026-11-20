# Google Colab training guide

Use VS Code on Windows for source edits and tests; use Colab for CUDA execution. Notebook: notebooks/train_colab.ipynb. It runs cells only when you explicitly select them. Set a GitHub URL/ref or upload a project ZIP in the first cell. Pin the Git commit SHA for repeatability.

## GPU and dependencies
Choose Colab NVIDIA T4. The notebook checks torch.cuda.is_available() before installing packages. It preserves the existing Colab PyTorch/torchvision, filters those packages from requirements installation and installs the remaining project requirements plus editable source. This exact environment has not been run here. CUDA uses AMP float16 with GradScaler; CPU uses float32. Seed comes only from src/config.py (59). No bf16 or FlashAttention 2.

## Storage and data
Dataset and temporary outputs use /content/data and /content/outputs. The notebook mounts Drive only for durable backups at MyDrive/PlantDiseaseVQA. It discovers/downloads PlantVillageVQA using the project CLI or reads an archive uploaded into /content. Do not train from Drive storage.

## Manual cells, tests and gate
Cells cover source checkout/upload, GPU/dependencies, Drive, data, EDA/tests, validation, each requested training configuration, evaluation and backups. No all-benchmark loop runs automatically. The EDA/test cell runs data/model/train-evaluate smoke suites. Each training cell first calls the source validator, which decodes images and runs required data pytest cases before creating evidence. If tests fail, stop and fix them; do not edit the validation report manually.

## Test view
Full factoid test contains 8,959 images, which exceeds mini/class budgets 1,000/8,000. Default policy preserves the original test and rejects incompatible mini/class sizes. The notebook explicitly sets test_policy subset for mini/class examples; those results describe a sampled view, not the full official test. Full can use preserve. Source image membership and CSV remain unchanged.

## Memory
Baseline microbatch is 16, gradient accumulation 1. Notebook uses microbatch 4 and accumulation 4, same nominal effective batch 16 with lower activation memory. If T4 runs out of memory, use 2/8, then 1/16. These settings affect dropout grouping and should be recorded. Notebook sets DataLoader workers to 2; use 0 if worker startup fails.

## Training and resume
Run one notebook cell per intended configuration: mini/full, class/full, class/frozen, class/question_only, class/image_only, full/full. Each is an explicit call; there is no automatic benchmark sweep. CLI supports --output_dir and --resume. Best validation checkpoint is outputs/models/<mode>_<level>.pt; resumable epoch state is outputs/models/<mode>_<level>_last.pt. Resume verifies level, mode, seed, data manifest and complete config; restores model, AdamW, scheduler, GradScaler, epoch/history and Python/NumPy/PyTorch/CUDA RNG states. It rejects missing, incompatible or completed checkpoints. Keep run names distinct.

Checkpoints also store tokenizer files, preprocessing, train-only answer vocabulary, validation selection metric and train-only question-type baseline. Unknown validation/test labels map to -1 and count as incorrect. Test is not used for checkpoint selection.

## Evaluation and backup
The evaluation cell calls python -m src.evaluate --ckpt <checkpoint>. Evaluation checks the dataset and split fingerprints, then writes exact-match accuracy, per-question-type accuracy, confusion matrix, predictions and train-only baseline metrics under /content/outputs/results. After each run it copies checkpoints/results/figures to Drive; source dataset remains under /content.

## Verification status
Python commands in the current Windows workspace fail to start (Access is denied/EPERM). Pytest, compileall, CLI help, notebook execution, CUDA AMP and Drive mounting are unverified here. No training or accuracy result is claimed; the first manual Colab run is still required.
