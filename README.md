# Plant Disease Visual Question Answering

Python, PyTorch, timm, Hugging Face Transformers and Streamlit.

## Current implementation
Data download/EDA, image-group data pipeline, ViTBERT, training and evaluation code are implemented. The Streamlit app remains a placeholder. Native Python process execution is blocked in this session: pytest and compileall have not run successfully. No full training, trained weights or measured model performance are claimed. See outputs/results/*implementation_report.json and docs/dataset_card.md for evidence and limitations.

The original specification PDF was absent during setup. Dataset source and actual archive structure are documented in docs/dataset_card.md. requirements.txt contains baseline pins; installation/compatibility remain unverified here.

## Setup and tests
Use Python 3.10+ and install an appropriate PyTorch CPU/CUDA build. The target GPU is Colab T4. Tests require the pinned ML libraries; offline model tests mock backbones and data tests use a local BERT vocabulary. The CUDA AMP smoke test skips when CUDA is unavailable.

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests/test_data.py tests/test_dataset_pipeline.py tests/test_models.py tests/test_train_evaluate.py -v
python -m compileall -q src scripts tests
~~~

## Dataset and EDA
~~~powershell
python -m src.download --data_dir data/PlantVillageVQA
python -m src.eda --data_dir data/PlantVillageVQA
python -m src.verify_numbers --data_dir data/PlantVillageVQA
~~~

Download discovers actual Hub assets and an immutable revision, checks archive paths, refuses source overwrites and records hashes. Set PLANTVQA_DATA_DIR or pass --data_dir. If multiple CSVs/revisions exist, explicitly pass --csv to EDA/train/evaluate.

CSV fields: image_id, question_type, question, answer, image_path, split. Resolve images by ID through the recursive image index, accepting jpg/JPG/jpeg/png. Never trust the stale source image_path folder layout. EDA writes JSON/CSV under outputs/results and configures PDF plots under outputs/figures.

Keep exactly four factoid categories: Detailed Verification, General Health Assessment, Plant Species Identification and Visual Attribute Grounding. This is the project policy, not an upstream card prescription. Word-length violations are errors rather than silent deletion.

All image selection/partitioning lives in src/data.py. Validation is ceil(15% of selected images), taken exclusively from original train. Species evidence uses direct species answers or positive identification questions; unknown or rare species go to other. Negative questions do not identify the true species.

Default test policy is preserve: retain all supplied original factoid test rows. The source has 8959 factoid test images, so this policy is incompatible with mini/class total budgets of 1000/8000 images. The explicit --test-policy subset option selects a factoid evaluation view, preserving each selected image's original train/test membership and leaving the original file untouched. Its exact quota sampler does not reproduce the undocumented reference mini/class sampler; differences are recorded in outputs/results/data_pipeline_inspection.json. Do not describe a sampled view as the complete official test.

After splitting, vocabulary uses only train answers with at least five QA occurrences. Remove rarer train rows; retain every selected validation/test row and encode unseen answers as -1. Report counts before and after rare-answer removal. Unknown labels always count as wrong, even if a baseline predicts -1.

## ViTBERT
Backbones are vit_base_patch16_224 and bert-base-uncased. Each outputs 768 features and has its own Linear(768,512) projection. Fusion is Linear(1024,512), ReLU, Dropout(0.5), Linear(512,num_classes). Modes: full, frozen, question_only, image_only. Disabled branches are not constructed; their fusion slots are zero-filled. Frozen encoders stay in eval and no_grad while projections/head train.

Dataset tensors are already normalized using the registered pretrained ViT config. ViTBERT's default image_format=normalized passes them through; explicit uint8/unit_float formats normalize raw RGB tensors before the encoder. Do not normalize twice. BERT uses the final CLS hidden state.

## Validate before training
The train CLI has a mandatory validation gate tied to CSV/image hashes, exact split rows/vocabulary, tokenizer settings, pipeline code and test code. --validate-only runs data pytest checks, checks invariants and decodes selected real images; it never constructs ViTBERT or AdamW. Synthetic success alone does not unlock training. Failed tests or skipped data checks cannot produce a passing artifact. Missing/stale evidence blocks training.

For the complete factoid test view, use level full with the preserve default. The following class example explicitly opts into the sampled test view:

~~~powershell
python -m src.train --level class --mode full --test-policy subset --validate-only
python -m src.train --level class --mode full --test-policy subset
~~~

The requested base CLI is available as python -m src.train --level class --mode full; it intentionally blocks until valid evidence exists, and class's preserve-policy budget conflict must be resolved explicitly. Never silently discard original test rows.

## Training behavior
Default: exactly four epochs, validation after each epoch, strict best-val-accuracy selection (ties retain the earlier epoch). AdamW has encoder/head learning rates of 2e-5/1e-3, with bias/one-dimensional parameters excluded from weight decay. Frozen parameters are excluded from the optimizer. CUDA uses autocast float16 and GradScaler; CPU smoke tests use float32. No bf16 or FlashAttention 2 is used.

Train uses CrossEntropyLoss and rejects unknown training labels. Validation/test CE loss is reported only on known answers; exact-match accuracy keeps all QA in its denominator. Training accepts train and val loaders only; test accuracy never selects checkpoints. An existing run marker/checkpoint/summary entry prevents repeating a configuration with the same centralized seed.

Outputs:
- outputs/models/<mode>_<level>.pt: best model_state_dict, config, answer_to_idx, mode, embedded tokenizer files, preprocessing metadata, epoch, seed, validation metric, train-only baseline and data/validation fingerprints.
- outputs/results/<mode>_<level>_history.csv: per-epoch train/val metrics.
- outputs/results/summary.csv: one row per level/mode/seed, updated with evaluation metrics rather than duplicated.
- outputs/results/data_validation_<level>_<policy>.json plus pytest XML/log: validation evidence.

## Evaluation
~~~powershell
python -m src.evaluate --ckpt outputs/models/full_class.pt
~~~

Pass --data_dir and --csv if relocating data. Evaluation loads checkpoints with weights_only=True, restores the embedded tokenizer locally, reconstructs transforms from saved metadata and validates dataset/split/vocabulary/image fingerprints before inference. It initializes backbones without downloading pretrained weights and strictly loads checkpoint weights.

Evaluation writes <checkpoint_stem>_test_metrics.json, predictions.csv, question_types.csv and confusion_matrix.csv under outputs/results. Exact-match accuracy is unnormalized answer-string identity via the checkpoint vocabulary. The confusion matrix includes an unknown-target row/column when vocabulary size is at most 64; larger matrices are explicitly skipped. The most-common-answer-per-question-type baseline and its fallback are fitted only from final train; no test answer counts are used.

The commands above document intended execution. They were not run successfully in this restricted session. CPU/CUDA runtime behavior and numerical results remain unverified until smoke tests execute in a working Python environment. No full training has run.
