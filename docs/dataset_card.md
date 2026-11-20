# PlantVillageVQA dataset card

## Source and revision
- Hugging Face: https://huggingface.co/datasets/SyedNazmusSakib/PlantVillageVQA
- Inspected revision: a025a438308eff9a7289231ebc09e27e8ec6678c.
- License declared by the upstream card: CC BY 4.0; cite the dataset and PlantVillage corpus.
- Actual Hub tree contains README.md, .gitattributes and PlantVillageVQA.zip (870561715 bytes).
- Actual ZIP contains PlantVillageVQA/PlantVillageVQA.csv, a JSON mirror and PlantVillageVQA/Images/ with mixed .jpg/.JPG extensions. Do not assume the advertised lowercase folder or image numbering.
- Source card is preserved in outputs/results/source_dataset_card.md. No project_spec.pdf was supplied.

## CSV and image resolution
Six columns: image_id, question_type, question, answer, image_path, split. Source image_id includes its extension. Stored image_path examples point to images/train/... although the archive is flat under Images/. Resolve image_id by filename stem through an actual recursive image index, accepting case-insensitive jpg/jpeg/png extensions. Reject ambiguous stems and missing files rather than guessing.

The original CSV is retained byte-for-byte under ignored data/. During this inspection only the CSV was extracted through HTTP ZIP ranges; images were checked against ZIP directory entries, not downloaded or decoded. The download CLI implements complete download/extraction separately.

## Observations calculated from the actual CSV
| Scope | QA pairs | Referenced images | Unique answers | Median answer words | P95 answer words | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Original | 193609 | 50073 | 125 | 1 | 13 | 18 |
| Four factoid types | 96825 | 45206 | 16 | 1 | 1 | 2 |

Word counts use whitespace tokenization; median and P95 use linear interpolation. Descriptive unique-answer counts cover all source rows and must never be reused as a classifier vocabulary. Model answer vocabulary is built from the final train partition only in src/data.py.

## Taxonomy
The upstream card defines nine types. The project's explicit four-type factoid policy retains Detailed Verification, General Health Assessment, Plant Species Identification and Visual Attribute Grounding. The upstream card itself does not prescribe this four-type filter. Select by exact type labels; do not select every short answer. Report length violations instead of silently deleting them.

| Question type | QA pairs | Unique answers | Median words | P95 words |
|---|---:|---:|---:|---:|
| Causal Reasoning | 26223 | 50 | 9 | 13 |
| Comprehensive Description | 27809 | 52 | 9 | 13 |
| Counterfactual Reasoning | 9981 | 36 | 16 | 18 |
| Detailed Verification | 14089 | 2 | 1 | 1 |
| Existence & Sanity Check | 2845 | 2 | 1 | 9 |
| General Health Assessment | 32099 | 2 | 1 | 1 |
| Plant Species Identification | 34151 | 16 | 1 | 1 |
| Specific Disease Identification | 29926 | 50 | 9 | 13 |
| Visual Attribute Grounding | 16486 | 2 | 1 | 1 |

## Original partitions and preservation
The actual source has train and test, no val: train 40066 images / 154977 QA; test 10007 images / 38632 QA. Train/test image overlap: 0.

Filtering is an analysis/training view; it does not alter the official test CSV. The factoid test view contains 8959 images / 19212 QA and must be labeled as a filtered evaluation view, not the entire official test. All future partitioning remains exclusively in src/data.py.

## Count reconciliation
All four project reference totals match the inspected CSV. The upstream card's 55448 images equals the actual ZIP image-file count, while 50073 is the count of distinct images referenced by annotations. There are 5375 archive images without QA and no annotation image IDs missing from the ZIP inventory. Filtering removes all retained QA for 4867 referenced images. No rows were added, edited or removed to force reference totals.

## Limitations
The inspection used Node.js because native Python execution is blocked. This is neither a pytest result nor training authorization. Actual image decoding, quality, source duplicates, licensing applicability and field generalization still require validation. PlantVillage's controlled imagery and synthetic QA templates can limit performance outside this dataset.
