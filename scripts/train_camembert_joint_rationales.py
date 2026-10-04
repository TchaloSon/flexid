#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


# ============================================================================
# CONFIGURATION DE L'EXPÉRIENCE
# ============================================================================

SPLIT_RELATIVE_DIRECTORY = Path("data") / "flexid_exact_group_split"
SPLIT_PROTOCOL_VERSION = "FLEXID-EXACT-GROUP-SPLIT-v3"
TRAINING_SCRIPT_VERSION = "FLEXID-CAMEMBERT-JOINT-RATIONALE-v2.0-BALANCED"

MODEL_ID = "almanach/camembert-base"
MODEL_DISPLAY_NAME = "CamemBERT-base joint NLI + rationale span"

EXPECTED_SPLIT_SIZES = {
    "train": 701,
    "validation": 151,
    "test": 150,
}
EXPECTED_INSTANCE_COUNT = sum(EXPECTED_SPLIT_SIZES.values())

LABELS = ("entailment", "contradiction", "neutral")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
NON_NEUTRAL_LABELS = {"entailment", "contradiction"}

SEEDS = (2026, 2027, 2028)
OUTPUT_DIRECTORY_NAME = "results_camembert_joint_rationale_v3_balanced"

MAX_LENGTH = 512
DOC_STRIDE = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_TRAIN_EPOCHS = 8
WARMUP_RATIO = 0.10
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0

# Équilibrage multi-tâche sans hyperparamètre ajusté sur le test :
#   total = CE(label) + span_CE_normalisée
#
# La CE de classification choisit parmi 3 labels (~log(3) à l'initialisation),
# tandis que la CE start/end choisit parmi plusieurs centaines de positions
# (~log(L)). Additionner les deux CE brutes fait mécaniquement dominer la
# tâche span. Pour chaque exemple span, on divise donc sa CE start/end par
# log(nombre de tokens de prémisse admissibles), soit la CE attendue d'un
# choix uniforme. La CE label reste exactement sur son échelle standard.
SPAN_LOSS_NORMALIZATION = "random_entropy_log_valid_premise_tokens"

# Sélection du checkpoint uniquement sur validation.
# Le Macro-F1 NLI est primaire, comme dans le baseline classification-only ;
# Joint IoU@0.50 sert uniquement de tie-break. Cela évite de sélectionner un
# checkpoint qui sacrifie l'inférence pour gagner quelques spans.
EARLY_STOPPING_PATIENCE = 2

# Un seul checkpoint temporaire par seed est conservé pendant le run.
# Il est supprimé après production des résultats finaux par défaut.
KEEP_BEST_CHECKPOINT = False
MIN_FREE_DISK_GB = 2.0

# Ignore un seed déjà terminé si sa signature expérimentale est identique.
SKIP_COMPLETED_RUNS = True

# Tokenisation utilisée UNIQUEMENT pour mesurer les rationales et rester
# comparable au protocole DeepSeek : tokens non blancs de la prémisse.
EVAL_TOKEN_PATTERN = re.compile(r"\S+")


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
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL invalide dans {path}, ligne {line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{path}, ligne {line_number}: un objet JSON est attendu."
                )
            rows.append(payload)
    return rows


def find_project_root() -> tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    current_dir = Path.cwd().resolve()

    candidates: list[Path] = []
    for start in (script_dir, current_dir):
        candidates.append(start)
        candidates.extend(start.parents)

    seen: set[Path] = set()
    checked: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        split_dir = candidate / SPLIT_RELATIVE_DIRECTORY
        checked.append(split_dir)
        required = [
            split_dir / "train.jsonl",
            split_dir / "validation.jsonl",
            split_dir / "test.jsonl",
            split_dir / "split_summary.json",
            split_dir / "split_membership.jsonl",
        ]
        if all(path.is_file() for path in required):
            return candidate, split_dir

    raise FileNotFoundError(
        "Impossible de trouver le split officiel FLEXID. Chemins testés :\n  - "
        + "\n  - ".join(str(path) for path in checked[:12])
    )


def set_global_seed(seed: int, torch: Any, np: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_free_disk(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < MIN_FREE_DISK_GB:
        raise RuntimeError(
            f"Espace disque libre insuffisant : {free_gb:.2f} Go. "
            f"Au moins {MIN_FREE_DISK_GB:.1f} Go sont exigés pour ce run."
        )


# ============================================================================
# VALIDATION DU SPLIT OFFICIEL
# ============================================================================


def validate_split_records(
    records: list[dict[str, Any]],
    *,
    split_name: str,
) -> list[dict[str, Any]]:
    expected_size = EXPECTED_SPLIT_SIZES[split_name]
    if len(records) != expected_size:
        raise ValueError(
            f"{split_name}: {len(records)} instances, {expected_size} attendues."
        )

    prepared: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for position, record in enumerate(records, start=1):
        required = ("id", "premise", "hypothesis_facts", "label")
        missing = [field for field in required if field not in record]
        if missing:
            raise ValueError(
                f"{split_name} #{position}: champs absents : {', '.join(missing)}"
            )

        record_id = record["id"]
        premise = record["premise"]
        hypothesis = record["hypothesis_facts"]
        label = record["label"]

        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{split_name} #{position}: id invalide.")
        if record_id in seen_ids:
            raise ValueError(f"{split_name}: id dupliqué {record_id}.")
        seen_ids.add(record_id)

        if not isinstance(premise, str) or not premise:
            raise ValueError(f"{record_id}: premise vide ou invalide.")
        if not isinstance(hypothesis, str) or not hypothesis:
            raise ValueError(f"{record_id}: hypothesis_facts vide ou invalide.")
        if label not in LABEL_TO_ID:
            raise ValueError(f"{record_id}: label invalide {label!r}.")

        rationale_start: int | None = None
        rationale_end: int | None = None

        if label in NON_NEUTRAL_LABELS:
            if "rationale_start" not in record or "rationale_end" not in record:
                raise ValueError(
                    f"{record_id}: rationale_start/rationale_end absents."
                )
            rationale_start = int(record["rationale_start"])
            rationale_end = int(record["rationale_end"])
            if not (0 <= rationale_start < rationale_end <= len(premise)):
                raise ValueError(
                    f"{record_id}: span caractère invalide "
                    f"[{rationale_start}, {rationale_end})."
                )
            rationale_text = record.get("rationale_text")
            if isinstance(rationale_text, str) and rationale_text:
                if premise[rationale_start:rationale_end] != rationale_text:
                    raise ValueError(
                        f"{record_id}: rationale_text ne correspond pas "
                        "exactement aux offsets caractères."
                    )

        prepared.append(
            {
                "id": record_id,
                "premise": premise,
                "hypothesis_facts": hypothesis,
                "label": label,
                "label_id": LABEL_TO_ID[label],
                "rationale_start": rationale_start,
                "rationale_end": rationale_end,
            }
        )

    return prepared


def load_and_validate_official_splits(
    split_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    summary_path = split_dir / "split_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = summary.get("protocol_version")
    if protocol != SPLIT_PROTOCOL_VERSION:
        raise ValueError(
            f"Protocole inattendu {protocol!r}; attendu {SPLIT_PROTOCOL_VERSION!r}."
        )

    records_by_split: dict[str, list[dict[str, Any]]] = {}
    all_ids: set[str] = set()

    for split_name in ("train", "validation", "test"):
        rows = validate_split_records(
            load_jsonl(split_dir / f"{split_name}.jsonl"),
            split_name=split_name,
        )
        ids = {row["id"] for row in rows}
        overlap = all_ids & ids
        if overlap:
            raise ValueError(
                f"Chevauchement d'IDs entre splits : {sorted(overlap)[:20]}"
            )
        all_ids.update(ids)
        records_by_split[split_name] = rows

    if len(all_ids) != EXPECTED_INSTANCE_COUNT:
        raise ValueError(
            f"Les splits couvrent {len(all_ids)} IDs ; "
            f"{EXPECTED_INSTANCE_COUNT} attendus."
        )

    membership = load_jsonl(split_dir / "split_membership.jsonl")
    membership_map = {str(row.get("id", "")): row.get("split") for row in membership}
    if len(membership_map) != EXPECTED_INSTANCE_COUNT:
        raise ValueError("split_membership.jsonl ne contient pas 1002 IDs uniques.")

    for split_name, rows in records_by_split.items():
        for row in rows:
            if membership_map.get(row["id"]) != split_name:
                raise ValueError(
                    f"{row['id']}: incohérence entre {split_name}.jsonl et "
                    "split_membership.jsonl."
                )

    return records_by_split, summary


# ============================================================================
# TOKENISATION ET ALIGNEMENT DES SPANS
# ============================================================================


def whitespace_tokens(premise: str) -> list[dict[str, Any]]:
    return [
        {
            "id": index,
            "text": match.group(0),
            "start_char": match.start(),
            "end_char": match.end(),
        }
        for index, match in enumerate(EVAL_TOKEN_PATTERN.finditer(premise), start=1)
    ]


def char_span_to_whitespace_span(
    *,
    premise: str,
    start_char: int,
    end_char: int,
    record_id: str,
) -> tuple[int, int]:
    tokens = whitespace_tokens(premise)
    selected = [
        token
        for token in tokens
        if token["end_char"] > start_char and token["start_char"] < end_char
    ]
    if not selected:
        raise ValueError(
            f"{record_id}: le span [{start_char}, {end_char}) ne recouvre "
            "aucun token d'évaluation."
        )
    return selected[0]["id"], selected[-1]["id"]


def preprocess_split(
    records: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    split_name: str,
    effective_max_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Tokenise chaque instance en fenêtres chevauchantes de la prémisse.

    La fenêtre est déterminée sans utiliser le rationale gold : le tokenizer
    parcourt toute la prémisse avec un stride fixe et répète l'hypothèse dans
    chaque fenêtre. Les offsets gold servent uniquement à définir la loss span
    sur les fenêtres qui contiennent intégralement le rationale.

    La loss de classification reçoit le label global sur toutes les fenêtres,
    avec un poids 1/N afin que chaque instance pèse autant, quel que soit son
    nombre de fenêtres. La loss span est répartie également entre les fenêtres
    contenant intégralement le rationale gold.
    """
    model_examples: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []

    total_windows = 0
    multi_window_instances = 0
    max_windows = 0

    for instance_index, record in enumerate(records):
        encoded = tokenizer(
            record["premise"],
            record["hypothesis_facts"],
            truncation="only_first",
            max_length=effective_max_length,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        if not hasattr(encoded, "sequence_ids"):
            raise RuntimeError(
                "Le tokenizer doit être un tokenizer rapide avec sequence_ids()."
            )

        input_windows = encoded["input_ids"]
        if not input_windows:
            raise ValueError(f"{record['id']}: aucune fenêtre tokenisée.")

        window_count = len(input_windows)
        total_windows += window_count
        max_windows = max(max_windows, window_count)
        if window_count > 1:
            multi_window_instances += 1

        gold_ws_start: int | None = None
        gold_ws_end: int | None = None
        if record["label"] in NON_NEUTRAL_LABELS:
            gold_ws_start, gold_ws_end = char_span_to_whitespace_span(
                premise=record["premise"],
                start_char=int(record["rationale_start"]),
                end_char=int(record["rationale_end"]),
                record_id=record["id"],
            )

        window_payloads: list[dict[str, Any]] = []
        gold_window_indices: list[int] = []

        for local_window_index in range(window_count):
            sequence_ids = encoded.sequence_ids(local_window_index)
            offsets = encoded["offset_mapping"][local_window_index]
            premise_mask = [1 if sid == 0 else 0 for sid in sequence_ids]
            if not any(premise_mask):
                raise ValueError(
                    f"{record['id']}: aucune partie de prémisse dans la fenêtre "
                    f"{local_window_index}."
                )

            start_position = -100
            end_position = -100
            fully_contains_gold = False

            if record["label"] in NON_NEUTRAL_LABELS:
                start_char = int(record["rationale_start"])
                end_char = int(record["rationale_end"])
                overlapping_positions = [
                    index
                    for index, (sid, offset) in enumerate(zip(sequence_ids, offsets))
                    if sid == 0
                    and int(offset[1]) > start_char
                    and int(offset[0]) < end_char
                ]

                if overlapping_positions:
                    candidate_start = overlapping_positions[0]
                    candidate_end = overlapping_positions[-1]
                    visible_start = int(offsets[candidate_start][0])
                    visible_end = int(offsets[candidate_end][1])
                    uncovered_left = record["premise"][start_char:visible_start]
                    uncovered_right = record["premise"][visible_end:end_char]
                    if not uncovered_left.strip() and not uncovered_right.strip():
                        fully_contains_gold = True
                        start_position = candidate_start
                        end_position = candidate_end
                        gold_window_indices.append(local_window_index)

            window_payloads.append(
                {
                    "local_window_index": local_window_index,
                    "input_ids": encoded["input_ids"][local_window_index],
                    "attention_mask": encoded["attention_mask"][local_window_index],
                    "token_type_ids": (
                        encoded["token_type_ids"][local_window_index]
                        if "token_type_ids" in encoded
                        else None
                    ),
                    "premise_mask": premise_mask,
                    "offset_mapping": [
                        (int(start), int(end)) for start, end in offsets
                    ],
                    "sequence_ids": sequence_ids,
                    "start_position": start_position,
                    "end_position": end_position,
                    "fully_contains_gold": fully_contains_gold,
                }
            )

        if record["label"] in NON_NEUTRAL_LABELS and not gold_window_indices:
            raise ValueError(
                f"{record['id']}: aucune fenêtre ne contient intégralement le "
                f"rationale gold à MAX_LENGTH={effective_max_length}, "
                f"DOC_STRIDE={DOC_STRIDE}."
            )

        label_weight = 1.0 / window_count
        span_weight = (
            1.0 / len(gold_window_indices)
            if gold_window_indices
            else 0.0
        )

        for payload in window_payloads:
            flat_window_index = len(model_examples)
            model_example: dict[str, Any] = {
                "input_ids": payload["input_ids"],
                "attention_mask": payload["attention_mask"],
                "premise_mask": payload["premise_mask"],
                "labels": record["label_id"],
                "label_loss_weight": label_weight,
                "start_positions": payload["start_position"],
                "end_positions": payload["end_position"],
                "span_loss_weight": (
                    span_weight if payload["fully_contains_gold"] else 0.0
                ),
                "example_index": flat_window_index,
            }
            if payload["token_type_ids"] is not None:
                model_example["token_type_ids"] = payload["token_type_ids"]

            model_examples.append(model_example)
            metadata.append(
                {
                    **record,
                    "instance_index": instance_index,
                    "window_index": flat_window_index,
                    "local_window_index": payload["local_window_index"],
                    "window_count": window_count,
                    "offset_mapping": payload["offset_mapping"],
                    "sequence_ids": payload["sequence_ids"],
                    "premise_mask": payload["premise_mask"],
                    "gold_model_start": (
                        payload["start_position"]
                        if payload["start_position"] != -100
                        else None
                    ),
                    "gold_model_end": (
                        payload["end_position"]
                        if payload["end_position"] != -100
                        else None
                    ),
                    "gold_ws_start": gold_ws_start,
                    "gold_ws_end": gold_ws_end,
                    "fully_contains_gold": payload["fully_contains_gold"],
                }
            )

    print(
        f"Tokenisation {split_name}: {len(records)} instances -> "
        f"{total_windows} fenêtres (MAX_LENGTH={effective_max_length}, "
        f"stride={DOC_STRIDE}); multi-fenêtres={multi_window_instances}, "
        f"max={max_windows}"
    )
    return model_examples, metadata


# ============================================================================
# DATASET / COLLATOR
# ============================================================================


class ListDataset:
    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class JointCollator:
    def __init__(self, tokenizer: Any, torch: Any) -> None:
        self.tokenizer = tokenizer
        self.torch = torch

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        tokenizer_features: list[dict[str, Any]] = []
        premise_masks: list[list[int]] = []
        labels: list[int] = []
        label_weights: list[float] = []
        starts: list[int] = []
        ends: list[int] = []
        span_weights: list[float] = []
        indices: list[int] = []

        for feature in features:
            item = {
                key: feature[key]
                for key in ("input_ids", "attention_mask", "token_type_ids")
                if key in feature
            }
            tokenizer_features.append(item)
            premise_masks.append(list(feature["premise_mask"]))
            labels.append(int(feature["labels"]))
            label_weights.append(float(feature["label_loss_weight"]))
            starts.append(int(feature["start_positions"]))
            ends.append(int(feature["end_positions"]))
            span_weights.append(float(feature["span_loss_weight"]))
            indices.append(int(feature["example_index"]))

        batch = self.tokenizer.pad(
            tokenizer_features,
            padding=True,
            return_tensors="pt",
        )
        sequence_length = int(batch["input_ids"].shape[1])
        padding_side = getattr(self.tokenizer, "padding_side", "right")

        padded_masks: list[list[int]] = []
        adjusted_starts: list[int] = []
        adjusted_ends: list[int] = []

        for mask, start, end in zip(premise_masks, starts, ends):
            pad_amount = sequence_length - len(mask)
            if pad_amount < 0:
                raise RuntimeError("Longueur de padding négative.")

            if padding_side == "left":
                padded_mask = [0] * pad_amount + mask
                if start != -100:
                    start += pad_amount
                    end += pad_amount
            else:
                padded_mask = mask + [0] * pad_amount

            padded_masks.append(padded_mask)
            adjusted_starts.append(start)
            adjusted_ends.append(end)

        batch["premise_mask"] = self.torch.tensor(
            padded_masks, dtype=self.torch.bool
        )
        batch["labels"] = self.torch.tensor(labels, dtype=self.torch.long)
        batch["label_loss_weights"] = self.torch.tensor(
            label_weights, dtype=self.torch.float
        )
        batch["start_positions"] = self.torch.tensor(
            adjusted_starts, dtype=self.torch.long
        )
        batch["end_positions"] = self.torch.tensor(
            adjusted_ends, dtype=self.torch.long
        )
        batch["span_loss_weights"] = self.torch.tensor(
            span_weights, dtype=self.torch.float
        )
        batch["example_index"] = self.torch.tensor(
            indices, dtype=self.torch.long
        )
        return batch


# ============================================================================
# MODÈLE JOINT
# ============================================================================


def build_joint_model_class(torch: Any):
    """Construit un modèle multi-tâche CamemBERT avec deux têtes standardisées.

    - NLI : tête de classification de type RoBERTa/CamemBERT
      (dropout -> dense -> tanh -> dropout -> projection).
    - Rationale : tête extractive start/end de type question answering.

    La CE des spans est normalisée par l'entropie d'un choix uniforme parmi
    les tokens de prémisse admissibles. Cela met la tâche extractive sur une
    échelle comparable sans choisir un lambda à partir du test.
    """
    nn = torch.nn

    class CamembertClassificationHead(nn.Module):
        def __init__(self, config: Any) -> None:
            super().__init__()
            hidden_size = int(config.hidden_size)
            dropout_p = getattr(config, "classifier_dropout", None)
            if dropout_p is None:
                dropout_p = getattr(config, "hidden_dropout_prob", 0.1)
            self.dropout = nn.Dropout(float(dropout_p))
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.out_proj = nn.Linear(hidden_size, len(LABELS))

            initializer_range = float(getattr(config, "initializer_range", 0.02))
            for layer in (self.dense, self.out_proj):
                nn.init.normal_(layer.weight, mean=0.0, std=initializer_range)
                nn.init.zeros_(layer.bias)

        def forward(self, sequence: Any) -> Any:
            x = sequence[:, 0, :]
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            return self.out_proj(x)

    class JointCamembertModel(nn.Module):
        def __init__(self, encoder: Any, config: Any) -> None:
            super().__init__()
            self.encoder = encoder
            hidden_size = int(config.hidden_size)

            # Même forme de tête que CamemBERT/RoBERTa sequence classification.
            self.label_classifier = CamembertClassificationHead(config)

            # Tête QA standard : deux logits par token, start et end.
            self.qa_outputs = nn.Linear(hidden_size, 2)
            initializer_range = float(getattr(config, "initializer_range", 0.02))
            nn.init.normal_(
                self.qa_outputs.weight,
                mean=0.0,
                std=initializer_range,
            )
            nn.init.zeros_(self.qa_outputs.bias)

        def forward(
            self,
            *,
            input_ids: Any,
            attention_mask: Any,
            premise_mask: Any,
            labels: Any | None = None,
            label_loss_weights: Any | None = None,
            start_positions: Any | None = None,
            end_positions: Any | None = None,
            span_loss_weights: Any | None = None,
            token_type_ids: Any | None = None,
        ) -> dict[str, Any]:
            encoder_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "return_dict": True,
            }
            if token_type_ids is not None:
                encoder_kwargs["token_type_ids"] = token_type_ids

            outputs = self.encoder(**encoder_kwargs)
            sequence = outputs.last_hidden_state

            label_logits = self.label_classifier(sequence)

            qa_logits = self.qa_outputs(sequence)
            start_logits = qa_logits[..., 0]
            end_logits = qa_logits[..., 1]

            # La tête span ne peut sélectionner que des tokens de la prémisse.
            invalid = ~premise_mask.bool()
            start_logits = start_logits.masked_fill(invalid, -1e4)
            end_logits = end_logits.masked_fill(invalid, -1e4)

            result = {
                "label_logits": label_logits,
                "start_logits": start_logits,
                "end_logits": end_logits,
            }

            if labels is None:
                return result

            # -----------------------------------------------------------------
            # 1) NLI : CE standard, même échelle que le baseline classification.
            # -----------------------------------------------------------------
            per_window_label_loss = nn.CrossEntropyLoss(reduction="none")(
                label_logits,
                labels,
            )
            if label_loss_weights is None:
                label_loss_weights = per_window_label_loss.new_ones(
                    per_window_label_loss.shape
                )
            label_denominator = label_loss_weights.sum().clamp_min(1e-12)
            label_loss = (
                per_window_label_loss * label_loss_weights
            ).sum() / label_denominator

            # -----------------------------------------------------------------
            # 2) Span : CE start/end uniquement sur les fenêtres supervisées.
            #    La CE brute dépend fortement du nombre de positions possibles.
            #    On la normalise donc, exemple par exemple, par log(K), où K est
            #    le nombre de tokens de prémisse autorisés dans la fenêtre.
            # -----------------------------------------------------------------
            span_loss_raw = label_loss.new_zeros(())
            span_loss_normalized = label_loss.new_zeros(())

            if start_positions is not None and end_positions is not None:
                valid_span = (
                    (start_positions != -100)
                    & (end_positions != -100)
                )
                if span_loss_weights is not None:
                    valid_span = valid_span & (span_loss_weights > 0)

                if bool(valid_span.any().item()):
                    valid_start_logits = start_logits[valid_span]
                    valid_end_logits = end_logits[valid_span]
                    valid_start_positions = start_positions[valid_span]
                    valid_end_positions = end_positions[valid_span]

                    per_start_loss = nn.CrossEntropyLoss(reduction="none")(
                        valid_start_logits,
                        valid_start_positions,
                    )
                    per_end_loss = nn.CrossEntropyLoss(reduction="none")(
                        valid_end_logits,
                        valid_end_positions,
                    )
                    per_span_loss_raw = 0.5 * (
                        per_start_loss + per_end_loss
                    )

                    # Nombre de positions réellement admissibles par exemple.
                    valid_token_counts = (
                        premise_mask[valid_span]
                        .sum(dim=1)
                        .to(dtype=per_span_loss_raw.dtype)
                        .clamp_min(2.0)
                    )
                    random_choice_entropy = torch.log(valid_token_counts)
                    per_span_loss_normalized = (
                        per_span_loss_raw / random_choice_entropy
                    )

                    if span_loss_weights is None:
                        valid_weights = per_span_loss_raw.new_ones(
                            per_span_loss_raw.shape
                        )
                    else:
                        valid_weights = span_loss_weights[valid_span]

                    span_denominator = valid_weights.sum().clamp_min(1e-12)
                    span_loss_raw = (
                        per_span_loss_raw * valid_weights
                    ).sum() / span_denominator
                    span_loss_normalized = (
                        per_span_loss_normalized * valid_weights
                    ).sum() / span_denominator

            # CE(label) garde exactement son poids standard. La composante span
            # est ramenée à une échelle ~1 au hasard au lieu de ~log(K).
            loss = label_loss + span_loss_normalized

            result.update(
                {
                    "loss": loss,
                    "label_loss": label_loss.detach(),
                    "span_loss_raw": span_loss_raw.detach(),
                    "span_loss_normalized": span_loss_normalized.detach(),
                }
            )
            return result

    return JointCamembertModel


# ============================================================================
# MÉTRIQUES
# ============================================================================


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def inclusive_token_set(start: int | None, end: int | None) -> set[int]:
    if start is None or end is None:
        return set()
    return set(range(int(start), int(end) + 1))


def span_scores(gold: set[int], pred: set[int]) -> tuple[float, float, bool]:
    if not gold and not pred:
        return 1.0, 1.0, True
    if not gold or not pred:
        return 0.0, 0.0, False

    intersection = len(gold & pred)
    precision = intersection / len(pred)
    recall = intersection / len(gold)
    token_f1 = safe_div(2.0 * precision * recall, precision + recall)
    union = len(gold | pred)
    iou = intersection / union if union else 0.0
    return token_f1, iou, gold == pred


def decode_best_span(
    start_logits: Sequence[float],
    end_logits: Sequence[float],
    premise_mask: Sequence[bool],
) -> tuple[int, int]:
    # O(L) : pour chaque fin, conserve le meilleur début vu jusque-là.
    best_start_position: int | None = None
    best_start_score = -float("inf")
    best_pair: tuple[int, int] | None = None
    best_pair_score = -float("inf")

    for position, is_premise in enumerate(premise_mask):
        if not is_premise:
            continue

        start_score = float(start_logits[position])
        if start_score > best_start_score:
            best_start_score = start_score
            best_start_position = position

        if best_start_position is None:
            continue

        pair_score = best_start_score + float(end_logits[position])
        if pair_score > best_pair_score:
            best_pair_score = pair_score
            best_pair = (best_start_position, position)

    if best_pair is None:
        raise RuntimeError("Impossible de décoder un span dans la prémisse.")
    return best_pair


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> dict[str, Any]:
    matrix = {
        gold: {pred: 0 for pred in LABELS}
        for gold in LABELS
    }
    for gold, pred in zip(y_true, y_pred):
        matrix[gold][pred] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[g][label] for g in LABELS if g != label)
        fn = sum(matrix[label][p] for p in LABELS if p != label)
        support = sum(matrix[label].values())
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    n = len(y_true)
    accuracy = safe_div(sum(a == b for a, b in zip(y_true, y_pred)), n)
    macro_precision = statistics.mean(
        float(per_class[label]["precision"]) for label in LABELS
    )
    macro_recall = statistics.mean(
        float(per_class[label]["recall"]) for label in LABELS
    )
    macro_f1 = statistics.mean(
        float(per_class[label]["f1"]) for label in LABELS
    )

    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    observed = accuracy
    expected = sum(
        (true_counts[label] / n) * (pred_counts[label] / n)
        for label in LABELS
    )
    kappa = (
        (observed - expected) / (1.0 - expected)
        if n and not math.isclose(1.0 - expected, 0.0)
        else float("nan")
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "cohen_kappa": kappa,
        "per_class": per_class,
        "confusion_matrix_gold_rows_pred_columns": matrix,
    }


def score_predictions(
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    y_true = [row["gold_label"] for row in prediction_rows]
    y_pred = [row["predicted_label"] for row in prediction_rows]
    classification = classification_metrics(y_true, y_pred)

    span_rows: list[dict[str, Any]] = []
    both_neutral = 0
    neutral_vs_non_neutral = 0
    joint_iou_success = 0
    joint_exact_success = 0

    for row in prediction_rows:
        gold_label = row["gold_label"]
        pred_label = row["predicted_label"]
        gold_tokens = inclusive_token_set(
            row["gold_rationale_start_token"],
            row["gold_rationale_end_token"],
        )
        pred_tokens = inclusive_token_set(
            row["predicted_rationale_start_token"],
            row["predicted_rationale_end_token"],
        )
        token_f1, iou, exact = span_scores(gold_tokens, pred_tokens)
        row["rationale_token_f1"] = token_f1
        row["rationale_iou"] = iou
        row["rationale_exact_match"] = exact
        row["label_correct"] = gold_label == pred_label

        if gold_label == "neutral" and pred_label == "neutral":
            both_neutral += 1
            joint_iou_success += 1
            joint_exact_success += 1
        elif (gold_label == "neutral") != (pred_label == "neutral"):
            neutral_vs_non_neutral += 1
        else:
            span_rows.append(row)
            if gold_label == pred_label and iou >= 0.50:
                joint_iou_success += 1
            if gold_label == pred_label and exact:
                joint_exact_success += 1

    n = len(prediction_rows)
    rationale = {
        "s_span_definition": (
            "instances où gold et modèle donnent tous deux un label non-neutral"
        ),
        "s_span": len(span_rows),
        "both_neutral": both_neutral,
        "neutral_vs_non_neutral_disagreements": neutral_vs_non_neutral,
        "macro_token_f1": (
            statistics.mean(float(row["rationale_token_f1"]) for row in span_rows)
            if span_rows
            else None
        ),
        "macro_iou": (
            statistics.mean(float(row["rationale_iou"]) for row in span_rows)
            if span_rows
            else None
        ),
        "exact_match": (
            statistics.mean(
                1.0 if row["rationale_exact_match"] else 0.0
                for row in span_rows
            )
            if span_rows
            else None
        ),
        "joint_iou_at_0_50": safe_div(joint_iou_success, n),
        "joint_exact_match": safe_div(joint_exact_success, n),
    }

    return {
        "classification": classification,
        "rationale": rationale,
    }


# ============================================================================
# IMPORTS ML ET ENVIRONNEMENT
# ============================================================================


def import_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        import transformers
        from torch.utils.data import DataLoader
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:
        raise RuntimeError(
            "Dépendance manquante. Vérifie torch, transformers et numpy dans "
            "le même Python que celui qui exécute ce script."
        ) from exc

    return {
        "np": np,
        "torch": torch,
        "transformers": transformers,
        "DataLoader": DataLoader,
        "AutoConfig": AutoConfig,
        "AutoModel": AutoModel,
        "AutoTokenizer": AutoTokenizer,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }


def environment_manifest(deps: dict[str, Any]) -> dict[str, Any]:
    np = deps["np"]
    torch = deps["torch"]
    transformers = deps["transformers"]
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        ),
    }


# ============================================================================
# ENTRAÎNEMENT ET ÉVALUATION
# ============================================================================


def move_batch_to_device(
    batch: dict[str, Any],
    device: Any,
) -> tuple[dict[str, Any], Any]:
    example_index = batch["example_index"]
    model_batch = {
        key: value.to(device)
        for key, value in batch.items()
        if key != "example_index"
    }
    return model_batch, example_index


def evaluate_model(
    *,
    model: Any,
    dataloader: Any,
    metadata: Sequence[dict[str, Any]],
    device: Any,
    torch: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    """Évalue au niveau INSTANCE après agrégation des fenêtres.

    Les probabilités de label sont moyennées entre toutes les fenêtres d'une
    même instance. Si le label agrégé est non-neutral, le span est décodé dans
    la fenêtre qui attribue la plus forte probabilité à ce label. Aucun offset
    gold n'intervient dans cette sélection.
    """
    model.eval()
    total_loss = 0.0
    batches = 0
    windows_by_instance: dict[int, list[dict[str, Any]]] = {}

    with torch.no_grad():
        for batch in dataloader:
            model_batch, example_indices = move_batch_to_device(batch, device)
            outputs = model(**model_batch)
            total_loss += float(outputs["loss"].detach().cpu().item())
            batches += 1

            label_logits = outputs["label_logits"].detach().cpu()
            label_probs = torch.softmax(label_logits, dim=-1)
            start_logits = outputs["start_logits"].detach().cpu()
            end_logits = outputs["end_logits"].detach().cpu()
            premise_mask = model_batch["premise_mask"].detach().cpu()

            for local_index, example_index_tensor in enumerate(example_indices):
                window_index = int(example_index_tensor.item())
                meta = metadata[window_index]
                instance_index = int(meta["instance_index"])
                windows_by_instance.setdefault(instance_index, []).append(
                    {
                        "meta": meta,
                        "label_probs": label_probs[local_index].tolist(),
                        "start_logits": start_logits[local_index].tolist(),
                        "end_logits": end_logits[local_index].tolist(),
                        "premise_mask": premise_mask[local_index].tolist(),
                    }
                )

    rows: list[dict[str, Any]] = []
    for instance_index in sorted(windows_by_instance):
        windows = windows_by_instance[instance_index]
        if not windows:
            raise RuntimeError(f"Instance {instance_index}: aucune fenêtre.")

        first_meta = windows[0]["meta"]
        n_windows = len(windows)
        mean_probs = [
            sum(float(window["label_probs"][label_id]) for window in windows)
            / n_windows
            for label_id in range(len(LABELS))
        ]
        predicted_label_id = max(
            range(len(LABELS)), key=lambda idx: mean_probs[idx]
        )
        predicted_label = ID_TO_LABEL[predicted_label_id]

        predicted_ws_start: int | None = None
        predicted_ws_end: int | None = None
        predicted_char_start: int | None = None
        predicted_char_end: int | None = None
        predicted_model_start: int | None = None
        predicted_model_end: int | None = None
        selected_local_window: int | None = None

        if predicted_label in NON_NEUTRAL_LABELS:
            selected = max(
                windows,
                key=lambda window: float(
                    window["label_probs"][predicted_label_id]
                ),
            )
            selected_meta = selected["meta"]
            pred_start, pred_end = decode_best_span(
                selected["start_logits"],
                selected["end_logits"],
                selected["premise_mask"],
            )
            predicted_model_start = pred_start
            predicted_model_end = pred_end
            selected_local_window = int(selected_meta["local_window_index"])
            offset_mapping = selected_meta["offset_mapping"]
            predicted_char_start = int(offset_mapping[pred_start][0])
            predicted_char_end = int(offset_mapping[pred_end][1])
            predicted_ws_start, predicted_ws_end = char_span_to_whitespace_span(
                premise=selected_meta["premise"],
                start_char=predicted_char_start,
                end_char=predicted_char_end,
                record_id=selected_meta["id"],
            )

        rows.append(
            {
                "id": first_meta["id"],
                "gold_label": first_meta["label"],
                "predicted_label": predicted_label,
                "gold_rationale_start_token": first_meta["gold_ws_start"],
                "gold_rationale_end_token": first_meta["gold_ws_end"],
                "predicted_rationale_start_token": predicted_ws_start,
                "predicted_rationale_end_token": predicted_ws_end,
                "gold_rationale_start_char": first_meta["rationale_start"],
                "gold_rationale_end_char": first_meta["rationale_end"],
                "predicted_rationale_start_char": predicted_char_start,
                "predicted_rationale_end_char": predicted_char_end,
                "predicted_model_start_position": predicted_model_start,
                "predicted_model_end_position": predicted_model_end,
                "window_count": n_windows,
                "selected_window": selected_local_window,
                "aggregated_label_probabilities": {
                    LABELS[i]: float(mean_probs[i]) for i in range(len(LABELS))
                },
            }
        )

    metrics = score_predictions(rows)
    mean_loss = total_loss / batches if batches else float("nan")
    return metrics, rows, mean_loss


def optimizer_for_model(model: Any, torch: Any) -> Any:
    no_decay_terms = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(term in name for term in no_decay_terms):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
    )


def run_signature(
    *,
    split_dir: Path,
    effective_max_length: int,
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "script_version": TRAINING_SCRIPT_VERSION,
        "split_protocol": SPLIT_PROTOCOL_VERSION,
        "model_id": MODEL_ID,
        "train_sha256": sha256_file(split_dir / "train.jsonl"),
        "validation_sha256": sha256_file(split_dir / "validation.jsonl"),
        "test_sha256": sha256_file(split_dir / "test.jsonl"),
        "max_length": effective_max_length,
        "doc_stride": DOC_STRIDE,
        "windowing": "overflow on premise only; hypothesis repeated in each window",
        "classification_window_aggregation": "mean softmax probabilities",
        "span_window_selection": "highest probability for aggregated predicted non-neutral label",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": NUM_TRAIN_EPOCHS,
        "warmup_ratio": WARMUP_RATIO,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "span_loss_normalization": SPAN_LOSS_NORMALIZATION,
        "loss": "CE(label) + entropy-normalized span CE",
        "selection_metric": "validation macro_f1; tie-break joint_iou_at_0_50",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "evaluation_rationale_tokenization": r"\S+",
        "environment": environment,
    }


def save_confusion_csv(path: Path, metrics: dict[str, Any]) -> None:
    matrix = metrics["classification"][
        "confusion_matrix_gold_rows_pred_columns"
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold\\pred", *LABELS])
        for gold in LABELS:
            writer.writerow([gold, *[matrix[gold][pred] for pred in LABELS]])


def run_one_seed(
    *,
    seed: int,
    output_root: Path,
    tokenizer: Any,
    tokenized_by_split: dict[str, list[dict[str, Any]]],
    metadata_by_split: dict[str, list[dict[str, Any]]],
    effective_max_length: int,
    split_dir: Path,
    dependencies: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    np = dependencies["np"]
    torch = dependencies["torch"]
    DataLoader = dependencies["DataLoader"]
    AutoConfig = dependencies["AutoConfig"]
    AutoModel = dependencies["AutoModel"]
    scheduler_factory = dependencies["get_linear_schedule_with_warmup"]

    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = seed_dir / "metrics.json"
    checkpoint_file = seed_dir / "best_model_state.pt"

    signature = run_signature(
        split_dir=split_dir,
        effective_max_length=effective_max_length,
        environment=environment,
    )
    signature["seed"] = seed

    if SKIP_COMPLETED_RUNS and metrics_file.is_file():
        existing = json.loads(metrics_file.read_text(encoding="utf-8"))
        if existing.get("run_signature") == signature:
            print(f"Run déjà terminé et compatible, ignoré : seed {seed}")
            return existing

    check_free_disk(output_root)
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    set_global_seed(seed, torch, np)

    config = AutoConfig.from_pretrained(MODEL_ID)
    encoder = AutoModel.from_pretrained(MODEL_ID, config=config)
    JointModel = build_joint_model_class(torch)
    model = JointModel(encoder, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    collator = JointCollator(tokenizer, torch)
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        ListDataset(tokenized_by_split["train"]),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=bool(torch.cuda.is_available()),
        generator=generator,
    )
    validation_loader = DataLoader(
        ListDataset(tokenized_by_split["validation"]),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=bool(torch.cuda.is_available()),
    )
    test_loader = DataLoader(
        ListDataset(tokenized_by_split["test"]),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=bool(torch.cuda.is_available()),
    )

    optimizer = optimizer_for_model(model, torch)
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / GRADIENT_ACCUMULATION_STEPS
    )
    total_optimizer_steps = optimizer_steps_per_epoch * NUM_TRAIN_EPOCHS
    warmup_steps = int(round(total_optimizer_steps * WARMUP_RATIO))
    scheduler = scheduler_factory(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    print("=" * 72)
    print(f"{MODEL_DISPLAY_NAME} — seed {seed}")
    print(f"Device                   : {device}")
    print(f"Batch train/eval         : {TRAIN_BATCH_SIZE}/{EVAL_BATCH_SIZE}")
    print(f"Steps optimiseur         : {total_optimizer_steps}")
    print(f"Warmup steps             : {warmup_steps}")
    print("Loss multi-tâche         : CE(label) + CE(span)/log(K)")
    print("Checkpoint selection     : val Macro-F1; tie-break Joint@.50")
    print(f"Window stride            : {DOC_STRIDE}")
    print("=" * 72)

    history: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0

    for epoch in range(1, NUM_TRAIN_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_total_loss = 0.0
        epoch_label_loss = 0.0
        epoch_span_loss_raw = 0.0
        epoch_span_loss_normalized = 0.0
        batch_count = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            model_batch, _ = move_batch_to_device(batch, device)
            outputs = model(**model_batch)
            loss = outputs["loss"] / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            epoch_total_loss += float(outputs["loss"].detach().cpu().item())
            epoch_label_loss += float(
                outputs["label_loss"].detach().cpu().item()
            )
            epoch_span_loss_raw += float(
                outputs["span_loss_raw"].detach().cpu().item()
            )
            epoch_span_loss_normalized += float(
                outputs["span_loss_normalized"].detach().cpu().item()
            )
            batch_count += 1

            should_step = (
                batch_index % GRADIENT_ACCUMULATION_STEPS == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        validation_metrics, validation_rows, validation_loss = evaluate_model(
            model=model,
            dataloader=validation_loader,
            metadata=metadata_by_split["validation"],
            device=device,
            torch=torch,
        )
        val_joint = float(
            validation_metrics["rationale"]["joint_iou_at_0_50"]
        )
        val_macro_f1 = float(
            validation_metrics["classification"]["macro_f1"]
        )
        selection_key = (val_macro_f1, val_joint)

        record = {
            "epoch": epoch,
            "train_loss": epoch_total_loss / max(batch_count, 1),
            "train_label_loss": epoch_label_loss / max(batch_count, 1),
            "train_span_loss_raw": (
                epoch_span_loss_raw / max(batch_count, 1)
            ),
            "train_span_loss_normalized": (
                epoch_span_loss_normalized / max(batch_count, 1)
            ),
            "validation_loss": validation_loss,
            "validation": validation_metrics,
            "learning_rate_after_epoch": float(scheduler.get_last_lr()[0]),
        }
        history.append(record)
        write_json(seed_dir / "training_history.json", history)

        print(
            f"Epoch {epoch:02d}/{NUM_TRAIN_EPOCHS} | "
            f"loss={record['train_loss']:.4f} "
            f"(label={record['train_label_loss']:.4f}, "
            f"span_raw={record['train_span_loss_raw']:.4f}, "
            f"span_norm={record['train_span_loss_normalized']:.4f}) | "
            f"val Macro-F1={val_macro_f1:.4f} | "
            f"val rationale-F1="
            f"{validation_metrics['rationale']['macro_token_f1']:.4f} | "
            f"val Joint@.50={val_joint:.4f}"
        )

        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_file)
            write_jsonl(
                seed_dir / "best_validation_predictions.jsonl",
                validation_rows,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping après epoch {epoch} "
                    f"(meilleur epoch={best_epoch})."
                )
                break

    if not checkpoint_file.is_file() or best_epoch is None:
        raise RuntimeError("Aucun checkpoint valide n'a été sauvegardé.")

    try:
        best_state = torch.load(
            checkpoint_file,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        best_state = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(best_state)

    validation_metrics, validation_rows, validation_loss = evaluate_model(
        model=model,
        dataloader=validation_loader,
        metadata=metadata_by_split["validation"],
        device=device,
        torch=torch,
    )
    test_metrics, test_rows, test_loss = evaluate_model(
        model=model,
        dataloader=test_loader,
        metadata=metadata_by_split["test"],
        device=device,
        torch=torch,
    )

    write_jsonl(seed_dir / "validation_predictions.jsonl", validation_rows)
    write_jsonl(seed_dir / "test_predictions.jsonl", test_rows)
    save_confusion_csv(seed_dir / "test_confusion_matrix.csv", test_metrics)

    result = {
        "experiment": MODEL_DISPLAY_NAME,
        "created_at_utc": utc_timestamp(),
        "seed": seed,
        "best_epoch": best_epoch,
        "run_signature": signature,
        "validation_loss": validation_loss,
        "test_loss": test_loss,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    write_json(metrics_file, result)

    if not KEEP_BEST_CHECKPOINT and checkpoint_file.exists():
        checkpoint_file.unlink()

    del model
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"Seed {seed} terminé | test Macro-F1="
        f"{test_metrics['classification']['macro_f1']:.4f} | "
        f"rationale-F1={test_metrics['rationale']['macro_token_f1']:.4f} | "
        f"Joint@.50={test_metrics['rationale']['joint_iou_at_0_50']:.4f}"
    )
    return result


# ============================================================================
# AGRÉGATION MULTI-SEEDS
# ============================================================================


def aggregate_values(values: Sequence[float]) -> dict[str, Any]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "values": list(values),
    }


def aggregate_runs(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metric_paths = {
        "accuracy": ("classification", "accuracy"),
        "macro_f1": ("classification", "macro_f1"),
        "cohen_kappa": ("classification", "cohen_kappa"),
        "rationale_macro_token_f1": ("rationale", "macro_token_f1"),
        "rationale_macro_iou": ("rationale", "macro_iou"),
        "rationale_exact_match": ("rationale", "exact_match"),
        "joint_iou_at_0_50": ("rationale", "joint_iou_at_0_50"),
        "joint_exact_match": ("rationale", "joint_exact_match"),
    }

    metrics: dict[str, Any] = {}
    for output_name, (section, metric_name) in metric_paths.items():
        values = [
            float(result["test"][section][metric_name])
            for result in results
        ]
        metrics[output_name] = aggregate_values(values)

    return {
        "experiment": MODEL_DISPLAY_NAME,
        "script_version": TRAINING_SCRIPT_VERSION,
        "model_id": MODEL_ID,
        "split_protocol": SPLIT_PROTOCOL_VERSION,
        "seeds": [result["seed"] for result in results],
        "runs": len(results),
        "metrics": metrics,
        "windowing": {
            "max_length": MAX_LENGTH,
            "doc_stride": DOC_STRIDE,
            "classification_aggregation": "mean softmax probabilities",
            "span_window_selection": "highest predicted-label probability",
        },
        "training_balance": {
            "label_loss": "standard 3-class cross-entropy",
            "span_loss": "start/end cross-entropy divided per example by log(K)",
            "K": "number of admissible premise tokens in the window",
            "checkpoint_selection": "validation macro_f1; tie-break joint_iou_at_0_50",
        },
        "rationale_metric_definition": {
            "tokenization": r"\S+ on premise",
            "macro_span_set": (
                "instances where gold and prediction are both non-neutral"
            ),
            "joint_iou_at_0_50": (
                "label correct AND rationale IoU >= 0.50; correct neutral "
                "predictions count as joint successes"
            ),
        },
    }


def write_aggregate_csv(path: Path, aggregate: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std", "min", "max", "values"])
        for metric, stats in aggregate["metrics"].items():
            writer.writerow(
                [
                    metric,
                    stats["mean"],
                    stats["std"],
                    stats["min"],
                    stats["max"],
                    json.dumps(stats["values"]),
                ]
            )


# ============================================================================
# MAIN
# ============================================================================


def main() -> int:
    project_root, split_dir = find_project_root()
    output_root = project_root / "data" / OUTPUT_DIRECTORY_NAME
    output_root.mkdir(parents=True, exist_ok=True)

    dependencies = import_dependencies()
    torch = dependencies["torch"]
    AutoTokenizer = dependencies["AutoTokenizer"]

    records_by_split, split_summary = load_and_validate_official_splits(split_dir)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "Ce baseline exige le tokenizer rapide CamemBERT pour aligner les "
            "offsets caractères et les sous-tokens."
        )

    model_max_length = getattr(tokenizer, "model_max_length", MAX_LENGTH)
    effective_max_length = min(
        MAX_LENGTH,
        model_max_length
        if isinstance(model_max_length, int) and model_max_length < 1_000_000
        else MAX_LENGTH,
    )

    tokenized_by_split: dict[str, list[dict[str, Any]]] = {}
    metadata_by_split: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("train", "validation", "test"):
        tokenized, metadata = preprocess_split(
            records_by_split[split_name],
            tokenizer=tokenizer,
            split_name=split_name,
            effective_max_length=effective_max_length,
        )
        tokenized_by_split[split_name] = tokenized
        metadata_by_split[split_name] = metadata

    environment = environment_manifest(dependencies)
    distributions = {
        split_name: dict(Counter(row["label"] for row in rows))
        for split_name, rows in records_by_split.items()
    }

    manifest = {
        "experiment": MODEL_DISPLAY_NAME,
        "created_at_utc": utc_timestamp(),
        "script_version": TRAINING_SCRIPT_VERSION,
        "project_root": str(project_root),
        "split_directory": str(split_dir),
        "split_protocol": SPLIT_PROTOCOL_VERSION,
        "split_sizes": EXPECTED_SPLIT_SIZES,
        "label_distributions": distributions,
        "model_id": MODEL_ID,
        "seeds": list(SEEDS),
        "environment": environment,
        "training": {
            "max_length": effective_max_length,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs_max": NUM_TRAIN_EPOCHS,
            "warmup_ratio": WARMUP_RATIO,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "max_grad_norm": MAX_GRAD_NORM,
            "span_loss_normalization": SPAN_LOSS_NORMALIZATION,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "selection": (
                "validation Macro-F1 label; Joint IoU@0.50 as tie-break"
            ),
        },
        "task": {
            "label_head": (
                "CamemBERT/RoBERTa-style classification head: "
                "dropout-dense-tanh-dropout-projection"
            ),
            "span_heads": "single linear QA head with start/end logits over premise tokens only",
            "neutral_span_training": "ignored (-100); predicted neutral => no span",
            "loss": "CE(label) + normalized mean(CE(start), CE(end))",
            "span_loss_normalization": SPAN_LOSS_NORMALIZATION,
        },
        "evaluation": {
            "rationale_tokenization": r"\S+ on the original premise",
            "gold_char_span": "start inclusive, end exclusive",
            "reported_span": "start inclusive, end inclusive",
            "joint_iou_at_0_50": (
                "correct label + rationale IoU>=0.50; correct neutral counts "
                "as a joint success"
            ),
            "deepseek_comparable": True,
        },
        "split_summary_sha256": sha256_file(split_dir / "split_summary.json"),
        "train_sha256": sha256_file(split_dir / "train.jsonl"),
        "validation_sha256": sha256_file(split_dir / "validation.jsonl"),
        "test_sha256": sha256_file(split_dir / "test.jsonl"),
    }
    write_json(output_root / "run_manifest.json", manifest)

    print("Baseline supervisé joint FLEXID")
    print(f"Projet                   : {project_root}")
    print(f"Split                    : {split_dir}")
    print(f"Protocole split          : {SPLIT_PROTOCOL_VERSION}")
    print(
        f"Tailles                   : train={EXPECTED_SPLIT_SIZES['train']}, "
        f"validation={EXPECTED_SPLIT_SIZES['validation']}, "
        f"test={EXPECTED_SPLIT_SIZES['test']}"
    )
    print(f"Modèle                   : {MODEL_ID}")
    print(f"Python                   : {environment['python']}")
    print(f"PyTorch                  : {environment['torch']}")
    print(f"Transformers             : {environment['transformers']}")
    print(f"GPU disponible           : {'oui' if environment['cuda_available'] else 'non'}")
    if not environment["cuda_available"]:
        print("AVERTISSEMENT            : entraînement sur CPU, donc lent.")
    print("")

    results: list[dict[str, Any]] = []
    for seed in SEEDS:
        result = run_one_seed(
            seed=seed,
            output_root=output_root,
            tokenizer=tokenizer,
            tokenized_by_split=tokenized_by_split,
            metadata_by_split=metadata_by_split,
            effective_max_length=effective_max_length,
            split_dir=split_dir,
            dependencies=dependencies,
            environment=environment,
        )
        results.append(result)

    aggregate = aggregate_runs(results)
    write_json(output_root / "aggregate_metrics.json", aggregate)
    write_aggregate_csv(output_root / "aggregate_metrics.csv", aggregate)

    metrics = aggregate["metrics"]
    print("")
    print("=" * 72)
    print("RÉSULTATS AGRÉGÉS — TEST OFFICIEL 150")
    print("=" * 72)
    print(
        f"Accuracy              : {metrics['accuracy']['mean']:.4f} "
        f"± {metrics['accuracy']['std']:.4f}"
    )
    print(
        f"Macro-F1              : {metrics['macro_f1']['mean']:.4f} "
        f"± {metrics['macro_f1']['std']:.4f}"
    )
    print(
        f"Cohen kappa           : {metrics['cohen_kappa']['mean']:.4f} "
        f"± {metrics['cohen_kappa']['std']:.4f}"
    )
    print(
        f"Rationale macro-F1    : "
        f"{metrics['rationale_macro_token_f1']['mean']:.4f} "
        f"± {metrics['rationale_macro_token_f1']['std']:.4f}"
    )
    print(
        f"Rationale macro-IoU   : "
        f"{metrics['rationale_macro_iou']['mean']:.4f} "
        f"± {metrics['rationale_macro_iou']['std']:.4f}"
    )
    print(
        f"Joint IoU@0.50        : "
        f"{metrics['joint_iou_at_0_50']['mean']:.4f} "
        f"± {metrics['joint_iou_at_0_50']['std']:.4f}"
    )
    print(f"Sorties               : {output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterruption demandée.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        raise SystemExit(1)
