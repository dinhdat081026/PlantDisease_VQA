# Data implementation report

## Implemented
- src/download.py: inspect immutable Hub tree, download discovered assets, safe ZIP extraction, hash provenance, discover CSV by schema, resolve images by ID across supported extensions; reject missing or ambiguous matches and source overwrites.
- src/eda.py: validate six-column CSV; exact four-type filter; raw/filtered/category/split statistics; image overlap; CSV/JSON outputs and matplotlib PDF figure generation.
- src/verify_numbers.py: compute reference comparisons and invariants; write observed values and fail on discrepancies, without modifying annotations.
- tests/test_dataset_pipeline.py: synthetic pytest cases for image resolution, ambiguous IDs, CSV validation, quoted/newline fields, immutable input, filter semantics, lengths/percentiles, overlap, count mismatches and safe ZIP extraction.
- requirements.txt: baseline dependency pins added; no installation/compatibility verification claimed.

## Actual work executed
Node.js fetch read the Hub API and source card. ZIP central-directory ranges established actual paths and file counts; the CSV payload was fetched and decompressed. Node.js computed actual statistics and ten reference/invariant checks; all ten were true. Saved CSV SHA-256: a96f8b5306bac41e171ad36af7ed3b2c0d0fd4ca4d3b2eacc06617ef8fafc06d. Original CSV bytes were checked unchanged after analysis. Source revision: a025a438308eff9a7289231ebc09e27e8ec6678c.

Evidence: outputs/results/archive_inventory.json, source_provenance.json, source_dataset_card.md, observed_numbers.json, observed_question_types.csv and inspection_checks.json. The independent inspection algorithm is saved in scripts/inspect_annotations.mjs; its standalone execution was not tested. Only the source CSV is present locally at data/source_inspection/PlantVillageVQA.csv; ZIP images were not downloaded or decoded. This separate location avoids confusing the inspection payload with a complete download.

## Attempts and limitations
- Native python -m pytest -q: process failed to spawn (EPERM), no tests executed.
- Native python -m compileall -q src scripts tests: process failed to spawn (EPERM), no compilation executed.
- Native PowerShell inventory: CreateProcessAsUserW Access is denied.
- Attempted in-process Pyodide fallback: initialization failed with Cannot determine runtime environment; no Python tests executed there either.
- Python download, EDA, verify_numbers and unittest commands have NOT executed. Node.js observations are explicitly separate from their results.
- No figures have been generated. src.eda is configured to save question_types.pdf and answer_lengths.pdf to outputs/figures/ when Python execution is available.
- No model training, answer vocabulary construction, new data splits or image decoding were performed. Real-image validation and passing Python tests remain required before training.

## Next executable checks
~~~powershell
python -m pytest -q
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
python -m src.download --data_dir data/PlantVillageVQA
python -m src.eda --data_dir data/PlantVillageVQA
python -m src.verify_numbers --data_dir data/PlantVillageVQA
~~~

If multiple complete revisions are downloaded, discovery intentionally fails rather than guessing: supply --csv pointing to the desired source revision. All reference totals matched; the 55448 versus 50073 image count difference is explained by 5375 archive images without annotated QA. No reference numbers were forced into observations.
