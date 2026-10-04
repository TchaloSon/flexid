# FLEXID: French Legal Explainable Inference Dataset

FLEXID is a benchmark for controlled natural language inference in French civil law. Its 1,002 instances pair a normative premise with a factual hypothesis and one of three labels: `entailment`, `contradiction`, or `neutral`. Non-neutral instances include a continuous extractive rationale in the premise. The corpus combines 240 pedagogically grounded factual scenarios with 762 norm-first controlled scenarios.

Repository: https://github.com/TchalosonForResearch/flexid

This supplementary package contains the preparation, training, evaluation and audit scripts, together with source and attribution notices. Corpus files, human annotations, model predictions, trained weights and historical run environments are separate experimental artifacts; they are not embedded in this scripts package.

## Installation and execution

The scripts use Python 3.10+ syntax. Machine-learning runs also require compatible versions of the packages in [requirements.txt](requirements.txt). This file declares dependencies; it is not a frozen environment from the original experiments. The desktop annotation interface uses Tkinter, supplied separately on some Python installations.

Run commands from the FLEXID project root, with `scripts/` and the corpus directory `data/` at that level:

```bash
python -m pip install -r requirements.txt
```

DeepSeek inference reads `DEEPSEEK_API_KEY` from the environment. Credentials are not included in the source files. Local scoring of saved predictions does not require an API key. Remote inference uses the model and decoding configuration defined in each script; a later API call is a new run and may differ from an earlier saved prediction.

## Corpus and official split

The preparation sequence is:

```bash
python scripts/shuffle_flexid_deterministic.py
python scripts/tokenize_flexid_premise.py
python scripts/create_flexid_group_split.py
```

The deterministic shuffle produces `data/flexid_shuffled.jsonl`; tokenization produces `data/flexid_shuffled_tokenized.jsonl`. The official split is stored in `data/flexid_exact_group_split/`, with 701 training, 151 validation and 150 test instances. The test contains 50 instances per label.

`create_flexid_group_split.py` forms connected components using equality of the input legal-reference strings or equality of premise text. Legal-reference canonicalisation is an upstream corpus-preparation step: the split script consumes the resulting strings and does not canonicalise them itself. Split membership, group counts and checks are written to its output files. Counts of legal authorities and connected components are different quantities.

The partial-input audit has its own grouping and resampling procedure. Its results are not evaluations on the official 150-instance test set.

## Independent human annotation

`select_flexid_kappa_rationales.py` selects 180 instances using seed 2026, with 60 instances per label and distinct normalised legal-reference keys. It exports a blind annotator file, aligned gold file and selection audit. Its matching and selection constraints do not by themselves establish that the selected cases were unseen by an annotator; the independent study is identified by the actual IDs in the completed annotation files.

Independent label and rationale agreement is computed with:

```bash
python scripts/calculate_iaa.py
```

The inputs are `data/flexid_kappa_rational_first_annotator.jsonl` and `data/flexid_kappa_rational_second_annotator.jsonl`. Instances are aligned by ID. The script reports raw agreement, Cohen's kappa, bootstrap intervals, rationale overlap and joint agreement. Its `END_TOKEN_IS_EXCLUSIVE` setting is `False`.

## Supervised baselines

```bash
python scripts/train_camembert_judibert.py
```

Both CamemBERT-base and JuriBERT-base use seeds 2026, 2027 and 2028 on the official split. Run signatures include split-file hashes and training settings. `train_camembert_joint_rationales.py` is an additional joint CamemBERT implementation; it is distinct from the mDeBERTa baseline.

The joint mDeBERTa baseline is run with:

```bash
python scripts/mdeberta_join_label_rationales.py train --train data/flexid_exact_group_split/train.jsonl --validation data/flexid_exact_group_split/validation.jsonl --test data/flexid_exact_group_split/test.jsonl --official-split-dir data/flexid_exact_group_split --output data/results_mdeberta_joint --seeds 2026 2027 2028
```

Training selects checkpoints using validation data. The official test is used for evaluation. Model revisions, numerical precision, effective batch settings and software versions recorded by a run are part of its provenance.

## DeepSeek evaluation

### Official 150-instance test

The default execution recalculates metrics from the saved predictions without sending API requests:

```bash
python scripts/deepseek_official_test_150.py --score-only
```

It reads `data/results_deepseek_official_test_v3/deepseek_official_test_150_predictions.jsonl` and the official test. Missing predictions stop the recalculation. Where an inference manifest is present, the test-file hash and split protocol must match it.

New outputs in the same directory are:

- `deepseek_official_test_150_metrics_aligned.json`;
- `deepseek_official_test_150_scored_aligned.jsonl`;
- `deepseek_official_test_150_confusion_aligned.csv`.

The input predictions and earlier output filenames are preserved. Scoring provenance records file hashes and the available inference manifest. The scoring protocol is `FLEXID-DEEPSEEK-MDEBERTA-ALIGNED-v1`.

For a new inference run, or to generate missing predictions using the configured API:

```bash
python scripts/deepseek_official_test_150.py --annotate
```

### Matched 180-instance analysis

`annotate_flexid_deepseekapi_180.py` generates zero-shot predictions; `annotate_flexid_deepseekapi_few_shot_180.py` implements the few-shot condition. `metrics_calculator_human_and_deepseek.py` evaluates annotation or prediction files against the sample's reference annotations. It is not the independent human-to-human IAA calculator. `deepseek_paired_prompt_ablation_boostrap.py` performs the paired zero-shot/few-shot analysis on aligned IDs.

The 180-instance analysis and the official 150-instance test have different purposes and denominators.

## Rationale conventions and comparability

Evaluation tokens are maximal non-whitespace sequences (`\S+`); punctuation attached to a word remains part of its token. Serialized token indices start at 1 and both bounds are inclusive: `{s, s+1, ..., e}`. Character offsets start at 0, with inclusive start and exclusive end: `[a,b)`.

| Evaluation | Cases entering mean token-F1 and IoU | Exact-match criterion |
|---|---|---|
| Independent human IAA | Both annotators assign a non-neutral label, including non-neutral label disagreements | Same token span |
| mDeBERTa | Correct non-neutral label | Same character boundaries |
| DeepSeek official test, aligned scoring | Correct non-neutral label | Same character boundaries; token exact match is also reported |
| Reference-versus-prediction analysis on 180 instances | Correct non-neutral label | Same token span |

DeepSeek's predicted character boundaries are reconstructed from the start of its first selected token and the exclusive end of its last selected token. They are not adjusted to match the gold. DeepSeek therefore selects whole whitespace tokens; mDeBERTa can produce finer character boundaries through its subword tokenizer.

In the aligned official-test outputs, `exact_match` and `character_exact_match` denote Character EM; `token_exact_match` remains separately available. The same distinction applies to joint exact-match fields. Conditional scores follow the same eligibility rule for the two official-test models, but the eligible instances can still differ because their label predictions differ.

Joint IoU@0.50 is evaluated over the full evaluation set. A non-neutral success requires the correct label and token IoU of at least 0.50. A neutral success requires the correct neutral label with no rationale. This provides a common end-to-end measure on the official test.

## Script inventory

| Script | Role |
|---|---|
| `annotation_tool.py` | Desktop interface for corpus construction and annotation |
| `shuffle_flexid_deterministic.py` | Deterministic corpus ordering |
| `tokenize_flexid_premise.py` | Whitespace tokenization and character-to-token mapping |
| `select_flexid_kappa_rationales.py` | Balanced 180-instance sample, blind export and audit |
| `calculate_iaa.py` | Independent human-to-human agreement |
| `create_flexid_group_split.py` | Official group-disjoint split and checks |
| `train_camembert_judibert.py` | Three-seed label-only encoder baselines |
| `mdeberta_join_label_rationales.py` | Joint mDeBERTa training and evaluation |
| `train_camembert_joint_rationales.py` | Additional joint CamemBERT implementation |
| `deepseek_official_test_150.py` | Official-test inference and aligned local scoring |
| `annotate_flexid_deepseekapi_180.py` | Zero-shot predictions on the 180-instance sample |
| `annotate_flexid_deepseekapi_few_shot_180.py` | Few-shot predictions on the 180-instance sample |
| `metrics_calculator_human_and_deepseek.py` | Reference-based label and rationale evaluation |
| `deepseek_paired_prompt_ablation_boostrap.py` | Paired prompting comparison and uncertainty |
| `premise_hypothesis_classifier.py` | Full-pair, partial-input and shuffled-premise audit |
| `name_sensitivity.py` | Name-identity sensitivity analysis |
| `gender_counterfactual_sensitivity_audit.py` | Grammatical-gender counterfactual audit |
| `deepseek_name_gender_sensitivity.py` | DeepSeek name and gender sensitivity analysis |
| `shuffle_and_create_batches.py` | Additional shuffling and batch-export utility |

## Sources, citation and licensing

[ATTRIBUTION.md](ATTRIBUTION.md) contains the dataset citation. [pratical_cases_source.md](pratical_cases_source.md) lists the teaching, assessment and case-law references, distinguishing pedagogical materials from official legal sources.

Original FLEXID dataset contributions and documentation are licensed under CC BY 4.0, within the scope of [LICENSE.md](LICENSE.md). Third-party material is not relicensed. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) distinguishes DILA/Légifrance, Cour de cassation/Judilibre, Conseil d'État/ArianeWeb and external pedagogical sources. The dataset license does not grant a software license for the scripts.

FLEXID is a research resource, not legal advice or an authoritative consolidated statement of current law.
