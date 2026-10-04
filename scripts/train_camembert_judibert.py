#!/usr/bin/env python3


from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


# ============================================================================
# CONFIGURATION
# ============================================================================

SPLIT_RELATIVE_DIRECTORY = (
    Path("data") / "flexid_exact_group_split"
)

SPLIT_PROTOCOL_VERSION = "FLEXID-EXACT-GROUP-SPLIT-v3"
TRAINING_SCRIPT_VERSION = "FLEXID-ENCODER-TRAIN-v3.0"

EXPECTED_INSTANCE_COUNT = 1002

EXPECTED_SPLIT_SIZES = {
    "train": 701,
    "validation": 151,
    "test": 150,
}

MODELS = (
    {
        "key": "camembert_base",
        "display_name": "CamemBERT-base",
        "model_id": "almanach/camembert-base",
    },
    {
        "key": "juribert_base",
        "display_name": "JuriBERT-base",
        "model_id": "dascim/juribert-base",
    },
)

LABELS = ("entailment", "contradiction", "neutral")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

TRAINING_SEEDS_BY_MODEL = {
    "camembert_base": (2026, 2027, 2028),
    "juribert_base": (2026, 2027, 2028),
}

OUTPUT_DIRECTORY_NAME = "results_encoder_baselines_exact_group_v3"

MAX_LENGTH = 512
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_TRAIN_EPOCHS = 8
WARMUP_RATIO = 0.10

PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1

# Si le GPU manque de VRAM avec batch=16, le script retente automatiquement
# avec batch=8 et accumulation=2, donc le batch effectif reste 16.
AUTO_OOM_BATCH_FALLBACK = True
FALLBACK_TRAIN_BATCH_SIZE = 8
FALLBACK_EVAL_BATCH_SIZE = 16
FALLBACK_GRADIENT_ACCUMULATION_STEPS = 2

EARLY_STOPPING_PATIENCE = 2
SAVE_TOTAL_LIMIT = 1

# Le script ignore les runs déjà terminés et valides.
SKIP_COMPLETED_RUNS = True



# ============================================================================
# UTILITAIRES GÉNÉRAUX
# ============================================================================

def utc_timestamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    temporary.replace(path)


def normalise_label(value: Any) -> str:
    label = str(value).strip().casefold()

    aliases = {
        "entailment": "entailment",
        "entails": "entailment",
        "implication": "entailment",
        "contradiction": "contradiction",
        "contradict": "contradiction",
        "neutral": "neutral",
        "neutre": "neutral",
    }

    return aliases.get(label, label)


def find_project_root_and_split_dir() -> tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent

    candidates = (
        script_dir,
        script_dir.parent,
    )

    checked: list[Path] = []

    for candidate in candidates:
        split_dir = candidate / SPLIT_RELATIVE_DIRECTORY
        checked.append(split_dir)

        required = (
            split_dir / "train.jsonl",
            split_dir / "validation.jsonl",
            split_dir / "test.jsonl",
            split_dir / "split_summary.json",
            split_dir / "split_membership.jsonl",
        )

        if all(path.is_file() for path in required):
            return candidate, split_dir

    rendered = "\n".join(f"  - {path}" for path in checked)

    raise FileNotFoundError(
        "Split exact-group v3 introuvable. Dossiers vérifiés :\n"
        f"{rendered}\n"
        "Place le script à la racine de FLEXID_FINAL ou dans scripts/."
    )


# ============================================================================
# CHARGEMENT ET VALIDATION DE FLEXID
# ============================================================================

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    text = path.read_text(encoding="utf-8-sig").strip()

    if not text:
        raise ValueError(f"Fichier vide : {path}")

    if text.startswith("["):
        data = json.loads(text)

        if not isinstance(data, list):
            raise ValueError(
                f"{path}: le JSON doit contenir une liste d'objets."
            )

        records = data

    else:
        records: list[dict[str, Any]] = []

        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}, ligne {line_number}: JSONL invalide : {exc}"
                ) from exc

            records.append(record)

    if not records:
        raise ValueError(f"Aucune instance trouvée dans {path}")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError(
            f"{path}: chaque instance doit être un objet JSON."
        )

    return records


def prepare_split_record(
    record: dict[str, Any],
    *,
    split_name: str,
    position: int,
) -> dict[str, Any]:
    required_fields = (
        "id",
        "premise",
        "hypothesis_facts",
        "label",
    )

    missing = [
        field
        for field in required_fields
        if field not in record
    ]

    if missing:
        raise ValueError(
            f"{split_name}, instance #{position}: champs absents : "
            + ", ".join(missing)
        )

    record_id = record["id"]
    premise = record["premise"]
    hypothesis_facts = record["hypothesis_facts"]
    label = record["label"]

    if not isinstance(record_id, str) or record_id == "":
        raise ValueError(
            f"{split_name}, instance #{position}: id invalide."
        )

    if not isinstance(premise, str) or premise == "":
        raise ValueError(
            f"{record_id}: premise doit être une chaîne non vide."
        )

    if (
        not isinstance(hypothesis_facts, str)
        or hypothesis_facts == ""
    ):
        raise ValueError(
            f"{record_id}: hypothesis_facts doit être une chaîne non vide."
        )

    if label not in LABEL_TO_ID:
        raise ValueError(
            f"{record_id}: label exact invalide {label!r}."
        )

    return {
        "id": record_id,
        "premise": premise,
        "hypothesis_facts": hypothesis_facts,
        "label": label,
        "label_id": LABEL_TO_ID[label],
    }


def label_distribution(
    records: Sequence[dict[str, Any]],
) -> dict[str, int]:
    counts = Counter(record["label"] for record in records)

    return {
        label: counts[label]
        for label in LABELS
    }


def load_and_validate_fixed_splits(
    split_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[int]],
    dict[str, Any],
]:
    split_names = ("train", "validation", "test")

    summary = json.loads(
        (split_dir / "split_summary.json").read_text(
            encoding="utf-8"
        )
    )

    protocol_version = summary.get("protocol_version")

    if protocol_version != SPLIT_PROTOCOL_VERSION:
        raise ValueError(
            "Version de split inattendue : "
            f"{protocol_version!r}. Version exigée : "
            f"{SPLIT_PROTOCOL_VERSION!r}."
        )

    records_by_split: dict[str, list[dict[str, Any]]] = {}
    all_ids: set[str] = set()

    for split_name in split_names:
        raw_records = load_jsonl(
            split_dir / f"{split_name}.jsonl"
        )

        expected_size = EXPECTED_SPLIT_SIZES[split_name]

        if len(raw_records) != expected_size:
            raise ValueError(
                f"{split_name}: {len(raw_records)} instances trouvées, "
                f"{expected_size} attendues."
            )

        prepared_records = [
            prepare_split_record(
                record,
                split_name=split_name,
                position=position,
            )
            for position, record in enumerate(
                raw_records,
                start=1,
            )
        ]

        split_ids = {
            record["id"]
            for record in prepared_records
        }

        if len(split_ids) != len(prepared_records):
            raise ValueError(
                f"{split_name}: identifiants dupliqués."
            )

        overlap = all_ids & split_ids

        if overlap:
            raise ValueError(
                "Chevauchement d'identifiants entre les splits : "
                f"{sorted(overlap)[:20]}"
            )

        all_ids.update(split_ids)
        records_by_split[split_name] = prepared_records

    if len(all_ids) != EXPECTED_INSTANCE_COUNT:
        raise ValueError(
            f"Les splits couvrent {len(all_ids)} identifiants ; "
            f"{EXPECTED_INSTANCE_COUNT} attendus."
        )

    stratification = summary.get("stratification", {})
    actual_instances = stratification.get(
        "actual_instances",
        {},
    )
    actual_labels = stratification.get(
        "actual_labels",
        {},
    )

    for split_name in split_names:
        observed_size = len(records_by_split[split_name])

        if actual_instances.get(split_name) != observed_size:
            raise ValueError(
                f"split_summary.json est incohérent pour {split_name}."
            )

        observed_distribution = label_distribution(
            records_by_split[split_name]
        )

        if actual_labels.get(split_name) != observed_distribution:
            raise ValueError(
                "Distribution des labels différente du résumé pour "
                f"{split_name}: {observed_distribution}"
            )

    membership_rows = load_jsonl(
        split_dir / "split_membership.jsonl"
    )

    membership_by_id: dict[str, str] = {}

    for row in membership_rows:
        record_id = row.get("id")
        split_name = row.get("split")

        if (
            not isinstance(record_id, str)
            or split_name not in split_names
        ):
            raise ValueError(
                "Ligne invalide dans split_membership.jsonl."
            )

        if record_id in membership_by_id:
            raise ValueError(
                f"ID dupliqué dans split_membership : {record_id}"
            )

        membership_by_id[record_id] = split_name

    if set(membership_by_id) != all_ids:
        raise ValueError(
            "split_membership.jsonl ne couvre pas exactement les splits."
        )

    for split_name in split_names:
        for record in records_by_split[split_name]:
            if membership_by_id[record["id"]] != split_name:
                raise ValueError(
                    f"Affectation incohérente pour {record['id']}."
                )

    records: list[dict[str, Any]] = []
    split: dict[str, list[int]] = {}

    for split_name in split_names:
        start_index = len(records)
        records.extend(records_by_split[split_name])
        end_index = len(records)

        split[split_name] = list(
            range(start_index, end_index)
        )

    return records, split, summary


# ============================================================================
# DÉPENDANCES
# ============================================================================

def import_training_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        import transformers
        import datasets
        import sklearn

        from datasets import Dataset
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            precision_recall_fscore_support,
        )
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
        from transformers.trainer_utils import get_last_checkpoint

    except ImportError as exc:
        raise RuntimeError(
            "Dépendances absentes. Installe-les avec :\n"
            "  python -m pip install -U torch transformers datasets "
            "accelerate scikit-learn sentencepiece safetensors"
        ) from exc

    return {
        "np": np,
        "torch": torch,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version,
        "Dataset": Dataset,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "precision_recall_fscore_support": (
            precision_recall_fscore_support
        ),
        "AutoModelForSequenceClassification": (
            AutoModelForSequenceClassification
        ),
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "EarlyStoppingCallback": EarlyStoppingCallback,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
        "get_last_checkpoint": get_last_checkpoint,
    }


# ============================================================================
# DONNÉES ET MÉTRIQUES
# ============================================================================

def records_for_indices(
    records: list[dict[str, Any]],
    indices: Sequence[int],
) -> list[dict[str, Any]]:
    return [
        {
            "id": records[index]["id"],
            "premise": records[index]["premise"],
            "hypothesis_facts": records[index]["hypothesis_facts"],
            "labels": records[index]["label_id"],
        }
        for index in indices
    ]


def build_compute_metrics(
    dependencies: dict[str, Any],
) -> Callable[[Any], dict[str, float]]:
    np = dependencies["np"]
    accuracy_score = dependencies["accuracy_score"]
    precision_recall_fscore_support = dependencies[
        "precision_recall_fscore_support"
    ]

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        predictions = eval_prediction.predictions

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predicted_ids = np.argmax(predictions, axis=-1)
        gold_ids = eval_prediction.label_ids

        accuracy = accuracy_score(gold_ids, predicted_ids)

        macro_precision, macro_recall, macro_f1, _ = (
            precision_recall_fscore_support(
                gold_ids,
                predicted_ids,
                average="macro",
                zero_division=0,
            )
        )

        weighted_precision, weighted_recall, weighted_f1, _ = (
            precision_recall_fscore_support(
                gold_ids,
                predicted_ids,
                average="weighted",
                zero_division=0,
            )
        )

        return {
            "accuracy": float(accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_precision),
            "weighted_recall": float(weighted_recall),
            "weighted_f1": float(weighted_f1),
        }

    return compute_metrics


def softmax_rows(logits: Any, np: Any) -> Any:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def detailed_test_metrics(
    gold_ids: Sequence[int],
    predicted_ids: Sequence[int],
    dependencies: dict[str, Any],
) -> dict[str, Any]:
    accuracy_score = dependencies["accuracy_score"]
    confusion_matrix = dependencies["confusion_matrix"]
    precision_recall_fscore_support = dependencies[
        "precision_recall_fscore_support"
    ]

    accuracy = float(
        accuracy_score(gold_ids, predicted_ids)
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            gold_ids,
            predicted_ids,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            gold_ids,
            predicted_ids,
            average="weighted",
            zero_division=0,
        )
    )

    per_precision, per_recall, per_f1, per_support = (
        precision_recall_fscore_support(
            gold_ids,
            predicted_ids,
            labels=list(range(len(LABELS))),
            average=None,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        gold_ids,
        predicted_ids,
        labels=list(range(len(LABELS))),
    )

    return {
        "accuracy": accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            label: {
                "precision": float(per_precision[index]),
                "recall": float(per_recall[index]),
                "f1": float(per_f1[index]),
                "support": int(per_support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": {
            "row_labels_gold": list(LABELS),
            "column_labels_prediction": list(LABELS),
            "values": matrix.astype(int).tolist(),
        },
    }


def write_confusion_matrix_csv(
    path: Path,
    matrix: Sequence[Sequence[int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold\\prediction", *LABELS])

        for label, row in zip(LABELS, matrix):
            writer.writerow([label, *row])


# ============================================================================
# COMPATIBILITÉ TRANSFORMERS
# ============================================================================

def build_training_arguments(
    TrainingArguments: Any,
    *,
    run_dir: Path,
    seed: int,
    use_fp16: bool,
    use_bf16: bool,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
) -> Any:
    """
    Construit TrainingArguments en fonction de la signature réellement
    installée.

    Transformers a renommé ou supprimé certains paramètres selon les
    versions. Seuls les paramètres explicitement acceptés par la classe
    locale sont transmis.
    """
    signature = inspect.signature(TrainingArguments.__init__)
    parameters = signature.parameters

    candidate_kwargs: dict[str, Any] = {
        "output_dir": str(run_dir / "checkpoints"),
        "do_train": True,
        "do_eval": True,
        "do_predict": True,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "warmup_ratio": WARMUP_RATIO,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "seed": seed,
        "data_seed": seed,
        "report_to": "none",
        "fp16": use_fp16,
        "bf16": use_bf16,
        "dataloader_num_workers": 0,
        "remove_unused_columns": True,
        "full_determinism": True,
    }

    if "eval_strategy" in parameters:
        candidate_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in parameters:
        candidate_kwargs["evaluation_strategy"] = "epoch"
    else:
        raise RuntimeError(
            "Version de Transformers incompatible : "
            "TrainingArguments ne propose ni eval_strategy "
            "ni evaluation_strategy."
        )

    supported_kwargs = {
        name: value
        for name, value in candidate_kwargs.items()
        if name in parameters
    }

    ignored_kwargs = sorted(
        set(candidate_kwargs) - set(supported_kwargs)
    )

    if "output_dir" not in supported_kwargs:
        raise RuntimeError(
            "TrainingArguments ne reconnaît pas output_dir."
        )

    if ignored_kwargs:
        print(
            "    Paramètres TrainingArguments non disponibles dans "
            "cette version, ignorés : "
            + ", ".join(ignored_kwargs)
        )

    return TrainingArguments(**supported_kwargs)


def trainer_tokenizer_argument(
    Trainer: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    parameters = inspect.signature(Trainer.__init__).parameters

    if "processing_class" in parameters:
        return {"processing_class": tokenizer}

    if "tokenizer" in parameters:
        return {"tokenizer": tokenizer}

    return {}


# ============================================================================
# ENTRAÎNEMENT D'UN RUN
# ============================================================================

def train_single_run(
    *,
    model_spec: dict[str, str],
    seed: int,
    records: list[dict[str, Any]],
    split: dict[str, list[int]],
    split_hashes: dict[str, str],
    output_root: Path,
    dependencies: dict[str, Any],
) -> dict[str, Any]:
    np = dependencies["np"]
    torch = dependencies["torch"]
    Dataset = dependencies["Dataset"]
    AutoTokenizer = dependencies["AutoTokenizer"]
    AutoModelForSequenceClassification = dependencies[
        "AutoModelForSequenceClassification"
    ]
    DataCollatorWithPadding = dependencies[
        "DataCollatorWithPadding"
    ]
    EarlyStoppingCallback = dependencies[
        "EarlyStoppingCallback"
    ]
    Trainer = dependencies["Trainer"]
    TrainingArguments = dependencies["TrainingArguments"]
    set_seed = dependencies["set_seed"]
    get_last_checkpoint = dependencies["get_last_checkpoint"]

    model_dir = output_root / model_spec["key"]
    run_dir = model_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    completed_metrics_file = run_dir / "test_metrics.json"

    run_signature = {
        "training_script_version": TRAINING_SCRIPT_VERSION,
        "model_key": model_spec["key"],
        "model_id": model_spec["model_id"],
        "seed": seed,
        "split_file_sha256": split_hashes,
        "training_configuration": {
            "max_length": MAX_LENGTH,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_train_epochs_max": NUM_TRAIN_EPOCHS,
            "warmup_ratio": WARMUP_RATIO,
            "preferred_per_device_train_batch_size": (
                PER_DEVICE_TRAIN_BATCH_SIZE
            ),
            "preferred_per_device_eval_batch_size": (
                PER_DEVICE_EVAL_BATCH_SIZE
            ),
            "preferred_gradient_accumulation_steps": (
                GRADIENT_ACCUMULATION_STEPS
            ),
            "auto_oom_batch_fallback": AUTO_OOM_BATCH_FALLBACK,
            "fallback_per_device_train_batch_size": (
                FALLBACK_TRAIN_BATCH_SIZE
            ),
            "fallback_per_device_eval_batch_size": (
                FALLBACK_EVAL_BATCH_SIZE
            ),
            "fallback_gradient_accumulation_steps": (
                FALLBACK_GRADIENT_ACCUMULATION_STEPS
            ),
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        },
    }

    if SKIP_COMPLETED_RUNS and completed_metrics_file.is_file():
        existing = json.loads(
            completed_metrics_file.read_text(encoding="utf-8")
        )

        if existing.get("run_signature") == run_signature:
            print(
                f"  Run déjà terminé et compatible, ignoré : "
                f"{model_spec['display_name']} — seed {seed}"
            )
            return existing

        print(
            f"  Résultat existant incompatible avec le protocole "
            f"courant ; nouveau run : "
            f"{model_spec['display_name']} — seed {seed}"
        )

    set_seed(seed)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_spec["model_id"],
            use_fast=True,
        )
        tokenizer_backend = "fast"
    except Exception:
        print(
            "    Tokenizer rapide indisponible ; "
            "tentative avec le tokenizer lent."
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_spec["model_id"],
            use_fast=False,
        )
        tokenizer_backend = "slow"

    model_max_length = getattr(
        tokenizer,
        "model_max_length",
        MAX_LENGTH,
    )

    effective_max_length = min(
        MAX_LENGTH,
        model_max_length
        if isinstance(model_max_length, int)
        and model_max_length < 1_000_000
        else MAX_LENGTH,
    )

    raw_datasets = {
        split_name: Dataset.from_list(
            records_for_indices(records, indices)
        )
        for split_name, indices in split.items()
    }

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["premise"],
            batch["hypothesis_facts"],
            truncation="only_first",
            max_length=effective_max_length,
        )

    tokenized = {
        split_name: dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=[
                "premise",
                "hypothesis_facts",
            ],
            desc=f"Tokenisation {model_spec['display_name']} {split_name}",
        )
        for split_name, dataset in raw_datasets.items()
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        model_spec["model_id"],
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )

    model.config.problem_type = "single_label_classification"

    cuda_available = bool(torch.cuda.is_available())
    bf16_supported = bool(
        cuda_available
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )

    use_bf16 = bf16_supported
    use_fp16 = cuda_available and not use_bf16

    def make_trainer_and_args(
        train_batch_size: int,
        eval_batch_size: int,
        gradient_accumulation_steps: int,
    ) -> tuple[Any, Any]:
        training_args_local = build_training_arguments(
            TrainingArguments,
            run_dir=run_dir,
            seed=seed,
            use_fp16=use_fp16,
            use_bf16=use_bf16,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        trainer_local = Trainer(
            model=model,
            args=training_args_local,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=DataCollatorWithPadding(
                tokenizer=tokenizer,
                pad_to_multiple_of=8 if cuda_available else None,
            ),
            compute_metrics=build_compute_metrics(dependencies),
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=EARLY_STOPPING_PATIENCE
                )
            ],
            **trainer_tokenizer_argument(Trainer, tokenizer),
        )

        return trainer_local, training_args_local

    active_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE
    active_eval_batch_size = PER_DEVICE_EVAL_BATCH_SIZE
    active_gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS
    batch_fallback_used = False

    trainer, training_args = make_trainer_and_args(
        active_train_batch_size,
        active_eval_batch_size,
        active_gradient_accumulation_steps,
    )

    checkpoint_dir = Path(training_args.output_dir)
    last_checkpoint = (
        get_last_checkpoint(str(checkpoint_dir))
        if checkpoint_dir.exists()
        else None
    )

    print(
        f"  Entraînement {model_spec['display_name']} — seed {seed}"
    )
    print(
        f"    précision mixte : "
        f"{'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}"
    )
    print(
        f"    batch train/eval : "
        f"{active_train_batch_size}/{active_eval_batch_size} | "
        f"accumulation={active_gradient_accumulation_steps} | "
        f"batch effectif="
        f"{active_train_batch_size * active_gradient_accumulation_steps}"
    )

    try:
        trainer.train(
            resume_from_checkpoint=last_checkpoint
            if last_checkpoint
            else None
        )
    except RuntimeError as exc:
        oom_message = str(exc).casefold()
        is_oom = (
            "out of memory" in oom_message
            or "cuda error: out of memory" in oom_message
        )

        if (
            not AUTO_OOM_BATCH_FALLBACK
            or not cuda_available
            or not is_oom
        ):
            raise

        print(
            "    CUDA OOM détecté : nouveau départ avec "
            f"batch={FALLBACK_TRAIN_BATCH_SIZE}, "
            f"eval_batch={FALLBACK_EVAL_BATCH_SIZE}, "
            f"accumulation="
            f"{FALLBACK_GRADIENT_ACCUMULATION_STEPS}."
        )

        torch.cuda.empty_cache()

        # Un run OOM partiellement créé ne doit pas être repris avec une autre
        # configuration de batch. Les checkpoints incomplets sont supprimés.
        if checkpoint_dir.exists():
            import shutil
            shutil.rmtree(checkpoint_dir)

        # Réinitialise le seed et recharge le modèle pour repartir proprement.
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_spec["model_id"],
            num_labels=len(LABELS),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            ignore_mismatched_sizes=True,
        )
        model.config.problem_type = "single_label_classification"

        active_train_batch_size = FALLBACK_TRAIN_BATCH_SIZE
        active_eval_batch_size = FALLBACK_EVAL_BATCH_SIZE
        active_gradient_accumulation_steps = (
            FALLBACK_GRADIENT_ACCUMULATION_STEPS
        )
        batch_fallback_used = True

        trainer, training_args = make_trainer_and_args(
            active_train_batch_size,
            active_eval_batch_size,
            active_gradient_accumulation_steps,
        )

        trainer.train(resume_from_checkpoint=None)

    validation_metrics = trainer.evaluate(
        eval_dataset=tokenized["validation"],
        metric_key_prefix="validation",
    )

    test_prediction = trainer.predict(
        tokenized["test"],
        metric_key_prefix="test",
    )

    logits = test_prediction.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    predicted_ids = np.argmax(logits, axis=-1)
    gold_ids = test_prediction.label_ids
    probabilities = softmax_rows(logits, np)

    test_metrics = detailed_test_metrics(
        gold_ids,
        predicted_ids,
        dependencies,
    )

    test_records = records_for_indices(
        records,
        split["test"],
    )

    prediction_rows = []

    for position, record in enumerate(test_records):
        predicted_id = int(predicted_ids[position])
        gold_id = int(gold_ids[position])

        prediction_rows.append(
            {
                "id": record["id"],
                "gold_label": ID_TO_LABEL[gold_id],
                "predicted_label": ID_TO_LABEL[predicted_id],
                "correct": gold_id == predicted_id,
                "probabilities": {
                    label: float(
                        probabilities[position][label_index]
                    )
                    for label_index, label in enumerate(LABELS)
                },
            }
        )

    run_result = {
        "created_at_utc": utc_timestamp(),
        "run_signature": run_signature,
        "model_key": model_spec["key"],
        "model_display_name": model_spec["display_name"],
        "model_id": model_spec["model_id"],
        "seed": seed,
        "effective_max_length": effective_max_length,
        "tokenizer_backend": tokenizer_backend,
        "training_configuration": {
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_train_epochs_max": NUM_TRAIN_EPOCHS,
            "warmup_ratio": WARMUP_RATIO,
            "per_device_train_batch_size": (
                active_train_batch_size
            ),
            "per_device_eval_batch_size": (
                active_eval_batch_size
            ),
            "gradient_accumulation_steps": (
                active_gradient_accumulation_steps
            ),
            "effective_train_batch_size": (
                active_train_batch_size
                * active_gradient_accumulation_steps
            ),
            "batch_fallback_used": batch_fallback_used,
            "early_stopping_patience": (
                EARLY_STOPPING_PATIENCE
            ),
            "precision": (
                "bf16"
                if use_bf16
                else "fp16"
                if use_fp16
                else "fp32"
            ),
        },
        "split_sizes": {
            split_name: len(indices)
            for split_name, indices in split.items()
        },
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_metric": trainer.state.best_metric,
        "validation_metrics": {
            key: (
                float(value)
                if isinstance(value, (int, float))
                else value
            )
            for key, value in validation_metrics.items()
        },
        "test": test_metrics,
    }

    write_json(
        completed_metrics_file,
        run_result,
    )

    write_jsonl(
        run_dir / "test_predictions.jsonl",
        prediction_rows,
    )

    write_confusion_matrix_csv(
        run_dir / "test_confusion_matrix.csv",
        test_metrics["confusion_matrix"]["values"],
    )

    trainer.save_model(str(run_dir / "best_model"))
    tokenizer.save_pretrained(str(run_dir / "best_model"))

    return run_result


# ============================================================================
# AGRÉGATION
# ============================================================================

AGGREGATE_METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
)


def aggregate_model_runs(
    model_spec: dict[str, str],
    run_results: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "model_key": model_spec["key"],
        "model_display_name": model_spec["display_name"],
        "model_id": model_spec["model_id"],
        "seeds": [
            result["seed"]
            for result in run_results
        ],
        "runs": len(run_results),
        "metrics": {},
    }

    for metric in AGGREGATE_METRICS:
        values = [
            float(result["test"][metric])
            for result in run_results
        ]

        aggregate["metrics"][metric] = {
            "mean": statistics.mean(values),
            "std": (
                statistics.stdev(values)
                if len(values) > 1
                else 0.0
            ),
            "min": min(values),
            "max": max(values),
            "values": values,
        }

    aggregate["per_class_f1"] = {}

    for label in LABELS:
        values = [
            float(result["test"]["per_class"][label]["f1"])
            for result in run_results
        ]

        aggregate["per_class_f1"][label] = {
            "mean": statistics.mean(values),
            "std": (
                statistics.stdev(values)
                if len(values) > 1
                else 0.0
            ),
            "values": values,
        }

    return aggregate


def comparison_row(
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    metrics = aggregate["metrics"]

    return {
        "model": aggregate["model_display_name"],
        "model_id": aggregate["model_id"],
        "runs": aggregate["runs"],
        "accuracy_mean": metrics["accuracy"]["mean"],
        "accuracy_std": metrics["accuracy"]["std"],
        "macro_f1_mean": metrics["macro_f1"]["mean"],
        "macro_f1_std": metrics["macro_f1"]["std"],
        "macro_precision_mean": metrics["macro_precision"]["mean"],
        "macro_recall_mean": metrics["macro_recall"]["mean"],
        "weighted_f1_mean": metrics["weighted_f1"]["mean"],
        "entailment_f1_mean": (
            aggregate["per_class_f1"]["entailment"]["mean"]
        ),
        "contradiction_f1_mean": (
            aggregate["per_class_f1"]["contradiction"]["mean"]
        ),
        "neutral_f1_mean": (
            aggregate["per_class_f1"]["neutral"]["mean"]
        ),
    }


def write_comparison_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("Aucun résultat à écrire.")

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    project_root, split_dir = find_project_root_and_split_dir()
    data_dir = project_root / "data"
    output_root = data_dir / OUTPUT_DIRECTORY_NAME
    output_root.mkdir(parents=True, exist_ok=True)

    records, split, split_summary = load_and_validate_fixed_splits(
        split_dir
    )

    dependencies = import_training_dependencies()
    torch = dependencies["torch"]

    split_hashes = {
        split_name: sha256_file(
            split_dir / f"{split_name}.jsonl"
        )
        for split_name in ("train", "validation", "test")
    }

    training_protocol = {
        "created_at_utc": utc_timestamp(),
        "training_script_version": TRAINING_SCRIPT_VERSION,
        "software": {
            "python": dependencies["python_version"],
            "numpy": dependencies["numpy_version"],
            "torch": dependencies["torch_version"],
            "transformers": dependencies["transformers_version"],
            "datasets": dependencies["datasets_version"],
            "scikit_learn": dependencies["sklearn_version"],
        },
        "task": "three-class NLI label classification",
        "rationales_predicted": False,
        "split_protocol_version": SPLIT_PROTOCOL_VERSION,
        "split_source_sha256": split_summary[
            "source"
        ]["sha256"],
        "split_file_sha256": split_hashes,
        "split_sizes": {
            split_name: len(indices)
            for split_name, indices in split.items()
        },
        "split_label_distribution": {
            split_name: label_distribution(
                [
                    records[index]
                    for index in indices
                ]
            )
            for split_name, indices in split.items()
        },
        "models": list(MODELS),
        "training_seeds_by_model": {
            model_key: list(seeds)
            for model_key, seeds in TRAINING_SEEDS_BY_MODEL.items()
        },
        "input_fields": [
            "premise",
            "hypothesis_facts",
        ],
        "ignored_fields": [
            "law_ref",
            "meta.law_ref",
            "rationale_start",
            "rationale_end",
            "rationale_text",
            "tokenized_premise",
        ],
        "training_configuration": {
            "max_length": MAX_LENGTH,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_train_epochs_max": NUM_TRAIN_EPOCHS,
            "warmup_ratio": WARMUP_RATIO,
            "preferred_per_device_train_batch_size": (
                PER_DEVICE_TRAIN_BATCH_SIZE
            ),
            "preferred_per_device_eval_batch_size": (
                PER_DEVICE_EVAL_BATCH_SIZE
            ),
            "preferred_gradient_accumulation_steps": (
                GRADIENT_ACCUMULATION_STEPS
            ),
            "preferred_effective_train_batch_size": (
                PER_DEVICE_TRAIN_BATCH_SIZE
                * GRADIENT_ACCUMULATION_STEPS
            ),
            "auto_oom_batch_fallback": AUTO_OOM_BATCH_FALLBACK,
            "fallback_per_device_train_batch_size": (
                FALLBACK_TRAIN_BATCH_SIZE
            ),
            "fallback_per_device_eval_batch_size": (
                FALLBACK_EVAL_BATCH_SIZE
            ),
            "fallback_gradient_accumulation_steps": (
                FALLBACK_GRADIENT_ACCUMULATION_STEPS
            ),
            "fallback_effective_train_batch_size": (
                FALLBACK_TRAIN_BATCH_SIZE
                * FALLBACK_GRADIENT_ACCUMULATION_STEPS
            ),
            "early_stopping_patience": (
                EARLY_STOPPING_PATIENCE
            ),
        },
    }

    write_json(
        output_root / "training_protocol.json",
        training_protocol,
    )

    print("Baselines supervisées FLEXID — split exact-group v3")
    print(f"Version script           : {TRAINING_SCRIPT_VERSION}")
    print(f"Projet                   : {project_root}")
    print(f"Split                    : {split_dir}")
    print(
        f"Protocole split          : "
        f"{SPLIT_PROTOCOL_VERSION}"
    )
    print(
        "Tailles                  : "
        f"train={len(split['train'])}, "
        f"validation={len(split['validation'])}, "
        f"test={len(split['test'])}"
    )
    print("Prédiction               : labels NLI uniquement")
    print("Rationales               : non prédites")
    print(
        "Python                   : "
        f"{dependencies['python_version'].split()[0]}"
    )
    print(
        "PyTorch                  : "
        f"{dependencies['torch_version']}"
    )
    print(
        "Transformers             : "
        f"{dependencies['transformers_version']}"
    )
    print(
        "Datasets                 : "
        f"{dependencies['datasets_version']}"
    )
    print(
        "scikit-learn             : "
        f"{dependencies['sklearn_version']}"
    )
    print(
        "NumPy                    : "
        f"{dependencies['numpy_version']}"
    )
    print(
        "GPU disponible          : "
        f"{'oui' if torch.cuda.is_available() else 'non'}"
    )

    if torch.cuda.is_available():
        print(
            "GPU                      : "
            f"{torch.cuda.get_device_name(0)}"
        )
    else:
        print(
            "AVERTISSEMENT            : entraînement sur CPU, "
            "très lent pour l'ensemble des runs."
        )

    all_aggregates: list[dict[str, Any]] = []

    for model_spec in MODELS:
        print("")
        print("=" * 72)
        print(
            f"{model_spec['display_name']} "
            f"({model_spec['model_id']})"
        )
        print("=" * 72)

        run_results = []

        for seed in TRAINING_SEEDS_BY_MODEL[model_spec["key"]]:
            run_results.append(
                train_single_run(
                    model_spec=model_spec,
                    seed=seed,
                    records=records,
                    split=split,
                    split_hashes=split_hashes,
                    output_root=output_root,
                    dependencies=dependencies,
                )
            )

        aggregate = aggregate_model_runs(
            model_spec,
            run_results,
        )
        all_aggregates.append(aggregate)

        model_dir = output_root / model_spec["key"]

        write_json(
            model_dir / "aggregate_results.json",
            aggregate,
        )

        print(
            f"  Macro-F1 test : "
            f"{aggregate['metrics']['macro_f1']['mean']:.4f} "
            f"± {aggregate['metrics']['macro_f1']['std']:.4f}"
        )
        print(
            f"  Accuracy test : "
            f"{aggregate['metrics']['accuracy']['mean']:.4f} "
            f"± {aggregate['metrics']['accuracy']['std']:.4f}"
        )

    comparison_rows = [
        comparison_row(aggregate)
        for aggregate in all_aggregates
    ]

    write_comparison_csv(
        output_root / "encoder_baselines_comparison.csv",
        comparison_rows,
    )

    write_json(
        output_root / "encoder_baselines_comparison.json",
        {
            "created_at_utc": utc_timestamp(),
            "split_protocol_version": SPLIT_PROTOCOL_VERSION,
            "split_source_sha256": split_summary[
                "source"
            ]["sha256"],
            "split_file_sha256": split_hashes,
            "training_seeds_by_model": {
                model_key: list(seeds)
                for model_key, seeds in TRAINING_SEEDS_BY_MODEL.items()
            },
            "rationales_predicted": False,
            "models": all_aggregates,
        },
    )

    print("")
    print("=" * 72)
    print("ENTRAÎNEMENTS TERMINÉS")
    print("=" * 72)
    print(f"Résultats                : {output_root}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nInterruption demandée. Les checkpoints déjà créés sont "
            "conservés et seront repris au prochain lancement.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        raise SystemExit(1)
