#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================================
# CONFIGURATION DE L'EXPÉRIENCE
# Version du scoring : FLEXID-DEEPSEEK-MDEBERTA-ALIGNED-v1.
# Exécution par défaut : recalcul local, sans appel API (équivalent à --score-only).
# --annotate autorise explicitement la génération des prédictions manquantes.
# Les anciens fichiers de métriques et les prédictions sont préservés en recalcul.
# Rationales : gold == prediction != neutral ; tokens inclusifs [s,e].
# Character EM : égalité des offsets [début,fin) ; Token EM conservé séparément.
# ============================================================================

# Même split officiel que CamemBERT/JuriBERT.
SPLIT_RELATIVE_DIRECTORY = Path("data") / "flexid_exact_group_split"
INPUT_FILENAME = "test.jsonl"
SPLIT_SUMMARY_FILENAME = "split_summary.json"
SPLIT_MEMBERSHIP_FILENAME = "split_membership.jsonl"
EXPECTED_SPLIT_PROTOCOL = "FLEXID-EXACT-GROUP-SPLIT-v3"
EXPECTED_INSTANCE_COUNT = 150
EXPECTED_LABEL_COUNTS = {
    "entailment": 50,
    "contradiction": 50,
    "neutral": 50,
}

# Sorties volontairement séparées de l'ancienne expérience sur 180 instances.
OUTPUT_DIRECTORY_NAME = "results_deepseek_official_test_v3"
PREDICTIONS_FILENAME = "deepseek_official_test_150_predictions.jsonl"
SCORED_PREDICTIONS_FILENAME = "deepseek_official_test_150_scored_aligned.jsonl"
METRICS_FILENAME = "deepseek_official_test_150_metrics_aligned.json"
CONFUSION_FILENAME = "deepseek_official_test_150_confusion_aligned.csv"
API_LOG_FILENAME = "deepseek_official_test_150_api_log.jsonl"
FAILURES_FILENAME = "deepseek_official_test_150_failures.jsonl"
MANIFEST_FILENAME = "deepseek_official_test_150_run_manifest.json"
PROMPT_FILENAME = "deepseek_official_test_150_prompt.txt"

SCORING_VERSION = "FLEXID-DEEPSEEK-MDEBERTA-ALIGNED-v1"

MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"

THINKING_TYPE = "enabled"
REASONING_EFFORT = "high"
MAX_OUTPUT_TOKENS = 8192

MAX_ATTEMPTS_PER_INSTANCE = 4
REQUEST_TIMEOUT_SECONDS = 180.0
MAX_CONSECUTIVE_FAILURES = 5
DELAY_BETWEEN_REQUESTS_SECONDS = 0.15

ALLOWED_LABELS = {"entailment", "contradiction", "neutral"}
LABEL_ORDER = ("entailment", "contradiction", "neutral")
NON_NEUTRAL_LABELS = {"entailment", "contradiction"}

# Tokenisation déterministe utilisée uniquement pour demander/évaluer les spans.
# Chaque séquence non blanche de la prémisse est un token.
TOKEN_PATTERN = re.compile(r"\S+")

SYSTEM_PROMPT = """Tu es chargé d'annoter des instances d'inférence juridique

en français. Tu dois décider du label uniquement à partir de la prémisse

juridique et des faits explicitement décrits. N'utilise aucune information

juridique ou factuelle extérieure pour compléter le cas.



DÉFINITIONS DES LABELS



Entailment : l'hypothèse découle nécessairement de la prémisse et des faits

explicitement décrits. Si la prémisse et les faits sont vrais, la conclusion

juridique ne peut pas être fausse.



Contradiction : l'hypothèse affirme une conséquence que la prémisse exclut

nécessairement dans la situation décrite. Si la prémisse et les faits sont

vrais, la conclusion juridique ne peut pas être vraie.



Neutral : la prémisse ne suffit ni à confirmer ni à réfuter l'hypothèse.

Une information juridique ou factuelle extérieure serait nécessaire pour

trancher.



RÈGLES POUR LE RATIONALE



1. Le rationale doit provenir uniquement de la prémisse tokenisée.

2. Pour entailment ou contradiction, sélectionne le plus petit passage

   CONTINU qui suffit à justifier le label dans la situation décrite.

3. Les numéros de tokens sont affichés entre crochets, par exemple [17].

4. rationale_start_token et rationale_end_token sont des bornes INCLUSIVES.

5. N'ajoute pas de tokens inutiles avant ou après le fondement pertinent.

6. Pour neutral, aucun passage ne suffit à trancher : les deux bornes doivent

   obligatoirement être null.

7. Ne retourne jamais un rationale provenant de l'hypothèse ou des faits.

8. Retourne exactement un objet JSON, sans commentaire, sans Markdown et sans

   explication.



SCHÉMA JSON OBLIGATOIRE



Pour un label non neutre :

{

  "id": "FLEXID-EXEMPLE",

  "label": "entailment",

  "rationale_start_token": 4,

  "rationale_end_token": 11

}



Pour neutral :

{

  "id": "FLEXID-EXEMPLE",

  "label": "neutral",

  "rationale_start_token": null,

  "rationale_end_token": null

}



Les seules valeurs autorisées pour label sont :

"entailment", "contradiction" et "neutral".

"""


# ============================================================================
# UTILITAIRES
# ============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL invalide dans {path}, ligne {line_number} : {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}, ligne {line_number}: un objet JSON est attendu."
                )
            records.append(record)

    if not records:
        raise ValueError(f"Aucune instance trouvée dans {path}")
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def find_project_root() -> tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    current_dir = Path.cwd().resolve()

    candidates: list[Path] = []
    for starting_point in (script_dir, current_dir):
        candidates.append(starting_point)
        candidates.extend(starting_point.parents)

    seen: set[Path] = set()
    checked: list[Path] = []

    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        split_dir = candidate / SPLIT_RELATIVE_DIRECTORY
        checked.append(split_dir)
        required = (
            split_dir / INPUT_FILENAME,
            split_dir / SPLIT_SUMMARY_FILENAME,
            split_dir / SPLIT_MEMBERSHIP_FILENAME,
        )
        if all(path.is_file() for path in required):
            return candidate, split_dir

    raise FileNotFoundError(
        "Impossible de trouver le split officiel FLEXID. Chemins testés :\n  - "
        + "\n  - ".join(str(path) for path in checked[:12])
    )


# ============================================================================
# PRÉPARATION DU TEST OFFICIEL ET TOKENISATION DES RATIONALES
# ============================================================================

def tokenize_premise(premise: str) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for token_id, match in enumerate(TOKEN_PATTERN.finditer(premise), start=1):
        tokens.append(
            {
                "id": token_id,
                "text": match.group(0),
                "start_char": match.start(),
                "end_char": match.end(),
            }
        )
    if not tokens:
        raise ValueError("Prémisse vide après tokenisation.")
    return tokens


def render_tokenized_premise(tokens: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{token['id']}] {token['text']}" for token in tokens)


def char_span_to_inclusive_token_span(
    *,
    premise: str,
    tokens: list[dict[str, Any]],
    start_char: int,
    end_char: int,
    record_id: str,
) -> tuple[int, int]:
    if not (0 <= start_char < end_char <= len(premise)):
        raise ValueError(
            f"{record_id}: span caractère invalide [{start_char}, {end_char})."
        )

    selected = [
        token
        for token in tokens
        if token["end_char"] > start_char and token["start_char"] < end_char
    ]
    if not selected:
        raise ValueError(
            f"{record_id}: le rationale gold ne recouvre aucun token."
        )
    return selected[0]["id"], selected[-1]["id"]


def prepare_official_test(
    records: list[dict[str, Any]],
    *,
    split_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if len(records) != EXPECTED_INSTANCE_COUNT:
        raise ValueError(
            f"Le test officiel doit contenir exactement {EXPECTED_INSTANCE_COUNT} "
            f"instances, mais {len(records)} ont été trouvées."
        )

    summary = json.loads(
        (split_dir / SPLIT_SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    protocol_version = summary.get("protocol_version")
    if protocol_version != EXPECTED_SPLIT_PROTOCOL:
        raise ValueError(
            f"Protocole de split inattendu : {protocol_version!r}; "
            f"attendu : {EXPECTED_SPLIT_PROTOCOL!r}."
        )

    membership = load_jsonl(split_dir / SPLIT_MEMBERSHIP_FILENAME)
    membership_by_id = {
        str(row.get("id", "")): str(row.get("split", ""))
        for row in membership
    }

    blind_instances: list[dict[str, Any]] = []
    gold_by_id: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    label_counts: Counter[str] = Counter()

    for position, record in enumerate(records, start=1):
        required = ("id", "premise", "hypothesis_facts", "label")
        missing = [field for field in required if field not in record]
        if missing:
            raise ValueError(
                f"Test #{position}: champs absents : {', '.join(missing)}"
            )

        record_id = str(record["id"]).strip()
        premise = record["premise"]
        hypothesis_facts = record["hypothesis_facts"]
        label = str(record["label"]).strip().casefold()

        if not record_id:
            raise ValueError(f"Test #{position}: id vide.")
        if record_id in seen_ids:
            raise ValueError(f"Identifiant dupliqué : {record_id}")
        seen_ids.add(record_id)

        if membership_by_id.get(record_id) != "test":
            raise ValueError(
                f"{record_id}: absent du split test dans split_membership.jsonl."
            )
        if not isinstance(premise, str) or not premise:
            raise ValueError(f"{record_id}: premise invalide.")
        if not isinstance(hypothesis_facts, str) or not hypothesis_facts:
            raise ValueError(f"{record_id}: hypothesis_facts invalide.")
        if label not in ALLOWED_LABELS:
            raise ValueError(f"{record_id}: label gold invalide {label!r}.")

        tokens = tokenize_premise(premise)
        tokenized_premise = render_tokenized_premise(tokens)

        gold_start_token: int | None
        gold_end_token: int | None

        if label == "neutral":
            gold_start_token = None
            gold_end_token = None
            start_char = end_char = 0
        else:
            if "rationale_start" not in record or "rationale_end" not in record:
                raise ValueError(
                    f"{record_id}: rationale_start/rationale_end absents."
                )
            start_char = int(record["rationale_start"])
            end_char = int(record["rationale_end"])
            gold_start_token, gold_end_token = char_span_to_inclusive_token_span(
                premise=premise,
                tokens=tokens,
                start_char=start_char,
                end_char=end_char,
                record_id=record_id,
            )

            rationale_text = record.get("rationale_text")
            if isinstance(rationale_text, str) and rationale_text:
                if premise[start_char:end_char] != rationale_text:
                    raise ValueError(
                        f"{record_id}: rationale_text ne correspond pas exactement "
                        "aux offsets caractères."
                    )

        # IMPORTANT : seul cet objet aveugle est envoyé au modèle.
        blind_instances.append(
            {
                "id": record_id,
                "tokenized_premise": tokenized_premise,
                "hypothesis_facts": hypothesis_facts,
                "max_token_id": tokens[-1]["id"],
            }
        )

        gold_by_id[record_id] = {
            "id": record_id,
            "label": label,
            "rationale_start_token": gold_start_token,
            "rationale_end_token": gold_end_token,
            # Métadonnées réservées au scoring : jamais envoyées à l'API.
            "rationale_start_char": start_char,
            "rationale_end_char": end_char,
            "premise_tokens": tokens,
        }
        label_counts[label] += 1

    if dict(label_counts) != EXPECTED_LABEL_COUNTS:
        observed = {label: label_counts[label] for label in LABEL_ORDER}
        raise ValueError(
            f"Distribution du test inattendue : {observed}; "
            f"attendue : {EXPECTED_LABEL_COUNTS}."
        )

    return blind_instances, gold_by_id


# ============================================================================
# PROMPT PAR INSTANCE
# ============================================================================

def build_user_prompt(

    instance: dict[str, Any],

    validation_error: str | None = None,

) -> str:

    correction = ""

    if validation_error:

        correction = (

            "\n\nCORRECTION TECHNIQUE\n"

            "La réponse précédente ne respectait pas le schéma demandé : "

            f"{validation_error}\n"

            "Refais l'annotation de la même instance et retourne uniquement "

            "l'objet JSON valide."

        )



    return f"""Annote l'instance suivante.



IDENTIFIANT

{instance['id']}



PRÉMISSE TOKENISÉE

{instance['tokenized_premise']}



HYPOTHÈSE ET FAITS EXPLICITEMENT DÉCRITS

{instance['hypothesis_facts']}



Le dernier numéro de token valide dans la prémisse est

{instance['max_token_id']}.



Retourne uniquement l'objet JSON demandé.{correction}"""


# ============================================================================
# VALIDATION DES PRÉDICTIONS
# ============================================================================

def parse_strict_integer(
    value: Any,
    *,
    field_name: str,
    record_id: str,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{record_id}: {field_name} ne peut pas être booléen.")
    if isinstance(value, int):
        return value
    raise ValueError(
        f"{record_id}: {field_name} doit être un entier JSON ou null."
    )


def validate_prediction(
    payload: Any,
    instance: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("la réponse JSON doit être un objet")

    expected_keys = {
        "id",
        "label",
        "rationale_start_token",
        "rationale_end_token",
    }
    actual_keys = set(payload)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing:
        raise ValueError("champs absents : " + ", ".join(missing))
    if extra:
        raise ValueError("champs supplémentaires : " + ", ".join(extra))

    record_id = payload["id"]
    if record_id != instance["id"]:
        raise ValueError(
            f"id incorrect : attendu {instance['id']!r}, reçu {record_id!r}"
        )

    label = payload["label"]
    if not isinstance(label, str):
        raise ValueError("label doit être une chaîne")
    label = label.strip().casefold()
    if label not in ALLOWED_LABELS:
        raise ValueError(
            f"label invalide {label!r}; valeurs autorisées : "
            "entailment, contradiction, neutral"
        )

    start = parse_strict_integer(
        payload["rationale_start_token"],
        field_name="rationale_start_token",
        record_id=instance["id"],
    )
    end = parse_strict_integer(
        payload["rationale_end_token"],
        field_name="rationale_end_token",
        record_id=instance["id"],
    )

    if label == "neutral":
        if start is not None or end is not None:
            raise ValueError("pour neutral, les deux bornes doivent être null")
    else:
        if start is None or end is None:
            raise ValueError(
                f"pour {label}, les deux bornes doivent être des entiers"
            )
        if start < 1 or end < 1:
            raise ValueError("les bornes doivent être supérieures ou égales à 1")
        if start > end:
            raise ValueError("rationale_start_token dépasse rationale_end_token")
        if end > instance["max_token_id"]:
            raise ValueError(
                f"rationale_end_token={end} dépasse le dernier token valide "
                f"{instance['max_token_id']}"
            )

    return {
        "id": instance["id"],
        "label": label,
        "rationale_start_token": start,
        "rationale_end_token": end,
    }


def parse_and_validate_response(
    content: str | None,
    instance: dict[str, Any],
) -> dict[str, Any]:
    if content is None or not content.strip():
        raise ValueError("contenu final vide")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON invalide : {exc}") from exc
    return validate_prediction(payload, instance)


# ============================================================================
# REPRISE ET MANIFESTE
# ============================================================================

def load_existing_predictions(
    path: Path,
    instances_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    predictions: dict[str, dict[str, Any]] = {}
    for line_number, payload in enumerate(load_jsonl(path), start=1):
        record_id = str(payload.get("id", "")).strip()
        if record_id not in instances_by_id:
            raise ValueError(
                f"{path}, ligne {line_number}: id inconnu {record_id!r}."
            )
        if record_id in predictions:
            raise ValueError(f"{path}: prédiction dupliquée pour {record_id}.")
        predictions[record_id] = validate_prediction(
            payload, instances_by_id[record_id]
        )
    return predictions


def build_manifest(input_file: Path) -> dict[str, Any]:
    return {
        "experiment": "FLEXID DeepSeek V4 Flash official test 150",
        "created_at_utc": utc_now_iso(),
        "split_protocol": EXPECTED_SPLIT_PROTOCOL,
        "input_file": str(input_file),
        "input_sha256": sha256_file(input_file),
        "expected_instances": EXPECTED_INSTANCE_COUNT,
        "expected_label_counts": EXPECTED_LABEL_COUNTS,
        "model": MODEL,
        "base_url": BASE_URL,
        "thinking": {"type": THINKING_TYPE},
        "reasoning_effort": REASONING_EFFORT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_attempts_per_instance": MAX_ATTEMPTS_PER_INSTANCE,
        "response_format": {"type": "json_object"},
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "premise_tokenization": r"regex \S+, token ids start at 1",
        "prediction_span_convention": "start inclusive, end inclusive",
        "gold_char_span_convention": "start inclusive, end exclusive",
        "gold_not_sent_to_model": True,
        "prediction_schema": {
            "id": "string",
            "label": list(LABEL_ORDER),
            "rationale_start_token": "integer|null",
            "rationale_end_token": "integer|null",
        },
        "one_request_per_instance": True,
        "reasoning_content_saved": False,
    }


def verify_or_create_manifest(
    path: Path,
    expected_manifest: dict[str, Any],
) -> None:
    if not path.exists():
        write_json_atomic(path, expected_manifest)
        return

    existing = json.loads(path.read_text(encoding="utf-8"))
    stable_keys = (
        "split_protocol",
        "input_sha256",
        "expected_instances",
        "expected_label_counts",
        "model",
        "base_url",
        "thinking",
        "reasoning_effort",
        "max_output_tokens",
        "response_format",
        "system_prompt_sha256",
        "premise_tokenization",
        "prediction_span_convention",
        "gold_char_span_convention",
        "gold_not_sent_to_model",
        "prediction_schema",
        "one_request_per_instance",
        "reasoning_content_saved",
    )
    differences = [
        key
        for key in stable_keys
        if existing.get(key) != expected_manifest.get(key)
    ]
    if differences:
        raise RuntimeError(
            "Le manifeste existant ne correspond pas à la configuration "
            "actuelle. Archive ou supprime le dossier de sortie avant une "
            "nouvelle expérience. Champs différents : "
            + ", ".join(differences)
        )


# ============================================================================
# APPEL À L'API
# ============================================================================

def import_openai() -> tuple[Any, Any, Any, Any]:
    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            OpenAI,
            RateLimitError,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Le paquet openai n'est pas installé. Exécute :\n"
            "  python -m pip install -U openai"
        ) from exc
    return OpenAI, APIConnectionError, APIStatusError, RateLimitError


def usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    result: dict[str, Any] = {}
    for field in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    ):
        value = getattr(usage, field, None)
        if value is not None:
            result[field] = value

    completion_details = getattr(usage, "completion_tokens_details", None)
    if completion_details is not None:
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
        if reasoning_tokens is not None:
            result["reasoning_tokens"] = reasoning_tokens
    return result or None


def status_code_from_exception(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def call_model_for_instance(
    *,
    client: Any,
    instance: dict[str, Any],
    api_log_file: Path,
    failure_file: Path,
    api_exception_types: tuple[type[BaseException], ...],
) -> dict[str, Any] | None:
    last_validation_error: str | None = None

    for attempt in range(1, MAX_ATTEMPTS_PER_INSTANCE + 1):
        user_prompt = build_user_prompt(
            instance, validation_error=last_validation_error
        )
        started_at = utc_now_iso()
        start_time = time.monotonic()

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=MAX_OUTPUT_TOKENS,
                reasoning_effort=REASONING_EFFORT,
                extra_body={"thinking": {"type": THINKING_TYPE}},
                stream=False,
            )
            elapsed_seconds = time.monotonic() - start_time

            if not response.choices:
                raise ValueError("la réponse API ne contient aucun choix")

            choice = response.choices[0]
            content = choice.message.content

            try:
                prediction = parse_and_validate_response(content, instance)
            except ValueError as validation_exc:
                last_validation_error = str(validation_exc)
                append_jsonl(
                    api_log_file,
                    {
                        "timestamp_utc": started_at,
                        "id": instance["id"],
                        "attempt": attempt,
                        "status": "invalid_model_output",
                        "validation_error": last_validation_error,
                        "raw_final_content": (
                            content[:2000] if isinstance(content, str) else None
                        ),
                        "model_requested": MODEL,
                        "model_returned": getattr(response, "model", None),
                        "response_id": getattr(response, "id", None),
                        "system_fingerprint": getattr(
                            response, "system_fingerprint", None
                        ),
                        "finish_reason": getattr(choice, "finish_reason", None),
                        "usage": usage_to_dict(getattr(response, "usage", None)),
                        "elapsed_seconds": round(elapsed_seconds, 3),
                        "reasoning_content_saved": False,
                        "user_prompt_sha256": sha256_text(user_prompt),
                    },
                )
                if attempt < MAX_ATTEMPTS_PER_INSTANCE:
                    time.sleep(min(8.0, 1.5 * attempt))
                    continue
                break

            append_jsonl(
                api_log_file,
                {
                    "timestamp_utc": started_at,
                    "id": instance["id"],
                    "attempt": attempt,
                    "status": "success",
                    "model_requested": MODEL,
                    "model_returned": getattr(response, "model", None),
                    "response_id": getattr(response, "id", None),
                    "system_fingerprint": getattr(
                        response, "system_fingerprint", None
                    ),
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "usage": usage_to_dict(getattr(response, "usage", None)),
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "reasoning_content_saved": False,
                    "user_prompt_sha256": sha256_text(user_prompt),
                },
            )
            return prediction

        except api_exception_types as exc:
            elapsed_seconds = time.monotonic() - start_time
            status_code = status_code_from_exception(exc)
            append_jsonl(
                api_log_file,
                {
                    "timestamp_utc": started_at,
                    "id": instance["id"],
                    "attempt": attempt,
                    "status": "api_error",
                    "exception_type": type(exc).__name__,
                    "status_code": status_code,
                    "error": str(exc)[:2000],
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "user_prompt_sha256": sha256_text(user_prompt),
                },
            )

            if status_code in {401, 403}:
                raise RuntimeError(
                    "Authentification DeepSeek refusée. Vérifie la clé API."
                ) from exc

            if attempt < MAX_ATTEMPTS_PER_INSTANCE:
                delay = min(60.0, 2.0 ** (attempt - 1))
                delay += random.uniform(0.0, 0.5)
                time.sleep(delay)
                continue

            last_validation_error = (
                f"échec API après {MAX_ATTEMPTS_PER_INSTANCE} tentatives : "
                f"{type(exc).__name__}: {str(exc)[:500]}"
            )
            break

        except Exception as exc:
            elapsed_seconds = time.monotonic() - start_time
            append_jsonl(
                api_log_file,
                {
                    "timestamp_utc": started_at,
                    "id": instance["id"],
                    "attempt": attempt,
                    "status": "unexpected_error",
                    "exception_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "user_prompt_sha256": sha256_text(user_prompt),
                },
            )
            last_validation_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            break

    append_jsonl(
        failure_file,
        {
            "timestamp_utc": utc_now_iso(),
            "id": instance["id"],
            "attempts": MAX_ATTEMPTS_PER_INSTANCE,
            "last_error": last_validation_error,
        },
    )
    return None


# ============================================================================
# MÉTRIQUES OFFICIELLES
# ============================================================================

def inclusive_token_set(start: int | None, end: int | None) -> set[int]:
    if start is None or end is None:
        return set()
    return set(range(start, end + 1))


def span_scores(gold: set[int], predicted: set[int]) -> tuple[float, float, bool]:
    if not gold and not predicted:
        return 1.0, 1.0, True
    if not gold or not predicted:
        return 0.0, 0.0, False
    intersection = len(gold & predicted)
    precision = intersection / len(predicted)
    recall = intersection / len(gold)
    f1 = (
        0.0
        if precision + recall == 0
        else 2.0 * precision * recall / (precision + recall)
    )
    iou = intersection / len(gold | predicted)
    return f1, iou, gold == predicted


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def cohen_kappa(y_true: list[str], y_pred: list[str]) -> float:
    n = len(y_true)
    observed = sum(a == b for a, b in zip(y_true, y_pred)) / n
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    expected = sum(
        (true_counts[label] / n) * (pred_counts[label] / n)
        for label in LABEL_ORDER
    )
    if math.isclose(1.0 - expected, 0.0):
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def prediction_character_bounds(
    gold: dict[str, Any], prediction: dict[str, Any],
) -> tuple[int, int]:
    """Reconstitue [début,fin) aux frontières des tokens prédits, sans ajuster au gold."""
    if prediction["label"] == "neutral":
        if (prediction["rationale_start_token"] is not None
                or prediction["rationale_end_token"] is not None):
            raise ValueError(f"{prediction['id']}: un label neutral doit avoir un span vide.")
        return 0, 0
    start = prediction["rationale_start_token"]
    end = prediction["rationale_end_token"]
    tokens = gold["premise_tokens"]
    if (type(start) is not int or type(end) is not int
            or not 1 <= start <= end <= len(tokens)):
        raise ValueError(f"{prediction['id']}: bornes de tokens invalides.")
    return tokens[start - 1]["start_char"], tokens[end - 1]["end_char"]


def compute_metrics(
    *,
    ordered_instances: list[dict[str, Any]],
    gold_by_id: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[int]]]:
    scored_rows: list[dict[str, Any]] = []
    y_true: list[str] = []
    y_pred: list[str] = []

    matrix = {
        gold_label: {pred_label: 0 for pred_label in LABEL_ORDER}
        for gold_label in LABEL_ORDER
    }

    span_rows: list[dict[str, Any]] = []
    both_neutral = 0
    neutral_vs_nonneutral = 0
    joint_iou_success = 0
    joint_exact_success = 0
    joint_token_exact_success = 0
    nonneutral_label_disagreements = 0

    for instance in ordered_instances:
        record_id = instance["id"]
        gold = gold_by_id[record_id]
        pred = predictions[record_id]

        gold_label = gold["label"]
        pred_label = pred["label"]
        y_true.append(gold_label)
        y_pred.append(pred_label)
        matrix[gold_label][pred_label] += 1

        gold_tokens = inclusive_token_set(
            gold["rationale_start_token"], gold["rationale_end_token"]
        )
        pred_tokens = inclusive_token_set(
            pred["rationale_start_token"], pred["rationale_end_token"]
        )

        token_f1, iou, token_exact = span_scores(gold_tokens, pred_tokens)
        pred_start_char, pred_end_char = prediction_character_bounds(gold, pred)
        gold_start_char = gold["rationale_start_char"]
        gold_end_char = gold["rationale_end_char"]
        character_exact = (gold_start_char, gold_end_char) == (pred_start_char, pred_end_char)
        span_eligible = gold_label == pred_label and gold_label in NON_NEUTRAL_LABELS

        if gold_label == "neutral" and pred_label == "neutral":
            both_neutral += 1
            joint_iou_success += 1
            joint_exact_success += 1
            joint_token_exact_success += 1
        elif (gold_label == "neutral") != (pred_label == "neutral"):
            neutral_vs_nonneutral += 1
        elif span_eligible:
            span_rows.append(
                {
                    "id": record_id,
                    "gold_label": gold_label,
                    "predicted_label": pred_label,
                    "token_f1": token_f1,
                    "iou": iou,
                    "character_exact_match": character_exact,
                    "token_exact_match": token_exact,
                }
            )
            if gold_label == pred_label and iou >= 0.50:
                joint_iou_success += 1
            if character_exact:
                joint_exact_success += 1
            if token_exact:
                joint_token_exact_success += 1
        else:
            # Deux labels non neutres différents : échec joint, exclu des moyennes conditionnelles.
            nonneutral_label_disagreements += 1

        scored_rows.append(
            {
                "id": record_id,
                "gold_label": gold_label,
                "predicted_label": pred_label,
                "label_correct": gold_label == pred_label,
                "gold_rationale_start_token": gold["rationale_start_token"],
                "gold_rationale_end_token": gold["rationale_end_token"],
                "predicted_rationale_start_token": pred[
                    "rationale_start_token"
                ],
                "predicted_rationale_end_token": pred[
                    "rationale_end_token"
                ],
                "rationale_token_f1": token_f1,
                "rationale_iou": iou,
                "span_eligible": span_eligible,
                "gold_rationale_start_char": gold_start_char,
                "gold_rationale_end_char": gold_end_char,
                "predicted_rationale_start_char": pred_start_char,
                "predicted_rationale_end_char": pred_end_char,
                "rationale_exact_match": character_exact,
                "rationale_character_exact_match": character_exact,
                "rationale_token_exact_match": token_exact,
            }
        )

    n = len(y_true)
    accuracy = sum(a == b for a, b in zip(y_true, y_pred)) / n

    per_class: dict[str, dict[str, float | int]] = {}
    for label in LABEL_ORDER:
        tp = matrix[label][label]
        fp = sum(matrix[g][label] for g in LABEL_ORDER if g != label)
        fn = sum(matrix[label][p] for p in LABEL_ORDER if p != label)
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

    macro_precision = sum(
        float(per_class[label]["precision"]) for label in LABEL_ORDER
    ) / len(LABEL_ORDER)
    macro_recall = sum(
        float(per_class[label]["recall"]) for label in LABEL_ORDER
    ) / len(LABEL_ORDER)
    macro_f1 = sum(
        float(per_class[label]["f1"]) for label in LABEL_ORDER
    ) / len(LABEL_ORDER)

    rationale_metrics: dict[str, Any] = {
        "s_span_definition": (
            "instances où gold et DeepSeek donnent le même label non-neutral "
            "(prédiction de label correcte)"
        ),
        "s_span": len(span_rows),
        "eligible_ids": [row["id"] for row in span_rows],
        "non_neutral_label_disagreements_excluded": nonneutral_label_disagreements,
        "exact_match_definition": "égalité des bornes de caractères [start,end)",
        "token_exact_match_definition": "égalité des ensembles de tokens inclusifs",
        "both_neutral": both_neutral,
        "neutral_vs_non_neutral_disagreements": neutral_vs_nonneutral,
        "macro_token_f1": (
            sum(row["token_f1"] for row in span_rows) / len(span_rows)
            if span_rows
            else None
        ),
        "macro_iou": (
            sum(row["iou"] for row in span_rows) / len(span_rows)
            if span_rows
            else None
        ),
        "exact_match": (
            sum(bool(row["character_exact_match"]) for row in span_rows)
            / len(span_rows)
            if span_rows
            else None
        ),
        "character_exact_match": (
            sum(row["character_exact_match"] for row in span_rows) / len(span_rows)
            if span_rows else None
        ),
        "token_exact_match": (
            sum(row["token_exact_match"] for row in span_rows) / len(span_rows)
            if span_rows else None
        ),
        "joint_iou_at_0_50": joint_iou_success / n,
        "joint_exact_match": joint_exact_success / n,
        "joint_character_exact_match": joint_exact_success / n,
        "joint_token_exact_match": joint_token_exact_success / n,
    }

    metrics = {
        "experiment": "DeepSeek official FLEXID test",
        "scoring_version": SCORING_VERSION,
        "character_span_reconstruction": (
            "DeepSeek prédit des tokens entiers : début du premier token, "
            "fin exclusive du dernier, sans ajustement aux frontières du gold"
        ),
        "instances": n,
        "split_protocol": EXPECTED_SPLIT_PROTOCOL,
        "classification": {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "cohen_kappa": cohen_kappa(y_true, y_pred),
            "per_class": per_class,
            "confusion_matrix_gold_rows_pred_columns": matrix,
        },
        "rationale": rationale_metrics,
        "tokenization": {
            "rule": r"\S+",
            "prediction_span_convention": "start inclusive, end inclusive",
            "gold_char_span_convention": "start inclusive, end exclusive",
        },
    }

    matrix_rows = [
        [matrix[gold_label][pred_label] for pred_label in LABEL_ORDER]
        for gold_label in LABEL_ORDER
    ]
    return metrics, scored_rows, matrix_rows


def write_confusion_csv(path: Path, matrix_rows: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold \\ predicted", *LABEL_ORDER])
        for label, row in zip(LABEL_ORDER, matrix_rows):
            writer.writerow([label, *row])


# ============================================================================
# EXÉCUTION
# ============================================================================

def annotate_and_score() -> int:
    project_root, split_dir = find_project_root()
    input_file = split_dir / INPUT_FILENAME
    output_dir = project_root / "data" / OUTPUT_DIRECTORY_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = output_dir / PREDICTIONS_FILENAME
    scored_file = output_dir / SCORED_PREDICTIONS_FILENAME
    metrics_file = output_dir / METRICS_FILENAME
    confusion_file = output_dir / CONFUSION_FILENAME
    api_log_file = output_dir / API_LOG_FILENAME
    failures_file = output_dir / FAILURES_FILENAME
    manifest_file = output_dir / MANIFEST_FILENAME
    prompt_file = output_dir / PROMPT_FILENAME

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Définir DEEPSEEK_API_KEY pour utiliser --annotate.")

    raw_records = load_jsonl(input_file)
    instances, gold_by_id = prepare_official_test(
        raw_records,
        split_dir=split_dir,
    )
    instances_by_id = {instance["id"]: instance for instance in instances}

    manifest = build_manifest(input_file)
    verify_or_create_manifest(manifest_file, manifest)

    if prompt_file.exists():
        existing_prompt = prompt_file.read_text(encoding="utf-8")
        if existing_prompt != SYSTEM_PROMPT:
            raise RuntimeError(
                "Le prompt sauvegardé diffère du prompt actuel. Archive ou "
                "supprime le dossier de sortie avant de recommencer."
            )
    else:
        prompt_file.write_text(SYSTEM_PROMPT, encoding="utf-8")

    predictions = load_existing_predictions(
        predictions_file,
        instances_by_id,
    )

    OpenAI, APIConnectionError, APIStatusError, RateLimitError = import_openai()
    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    pending = [
        instance for instance in instances if instance["id"] not in predictions
    ]

    print("Évaluation FLEXID — DeepSeek-V4-Flash — test officiel 150")
    print(f"Projet                   : {project_root}")
    print(f"Split                    : {split_dir}")
    print(f"Entrée                   : {input_file}")
    print(f"Protocole split          : {EXPECTED_SPLIT_PROTOCOL}")
    print(f"Modèle                   : {MODEL}")
    print(f"Thinking                 : {THINKING_TYPE}")
    print(f"Reasoning effort         : {REASONING_EFFORT}")
    print(f"Instances test           : {len(instances)}")
    print(f"Déjà validées            : {len(predictions)}")
    print(f"À traiter                : {len(pending)}")
    print(f"Sorties                  : {output_dir}")
    print("")

    consecutive_failures = 0
    run_failures = 0
    api_exception_types = (
        APIConnectionError,
        APIStatusError,
        RateLimitError,
    )

    position_by_id = {
        instance["id"]: index
        for index, instance in enumerate(instances, start=1)
    }

    for pending_index, instance in enumerate(pending, start=1):
        global_position = position_by_id[instance["id"]]
        print(
            f"[{global_position:03d}/{len(instances)}] {instance['id']}...",
            end=" ",
            flush=True,
        )

        prediction = call_model_for_instance(
            client=client,
            instance=instance,
            api_log_file=api_log_file,
            failure_file=failures_file,
            api_exception_types=api_exception_types,
        )

        if prediction is None:
            run_failures += 1
            consecutive_failures += 1
            print("ÉCHEC")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"{MAX_CONSECUTIVE_FAILURES} échecs consécutifs. Arrêt "
                    "préventif pour éviter de multiplier les appels."
                )
            continue

        consecutive_failures = 0
        predictions[instance["id"]] = prediction
        append_jsonl(predictions_file, prediction)
        print(
            f"OK — {prediction['label']} "
            f"[{prediction['rationale_start_token']}, "
            f"{prediction['rationale_end_token']}]"
        )

        if (
            DELAY_BETWEEN_REQUESTS_SECONDS > 0
            and pending_index < len(pending)
        ):
            time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)

    ordered_predictions = [
        predictions[instance["id"]]
        for instance in instances
        if instance["id"] in predictions
    ]
    write_jsonl_atomic(predictions_file, ordered_predictions)

    completed = len(ordered_predictions)
    missing_ids = [
        instance["id"]
        for instance in instances
        if instance["id"] not in predictions
    ]

    print("")
    print("Exécution API terminée.")
    print(f"Prédictions validées     : {completed}/{len(instances)}")
    print(f"Échecs pendant ce run    : {run_failures}")
    print(f"Prédictions              : {predictions_file}")
    print(f"Journal API              : {api_log_file}")
    print(f"Échecs                   : {failures_file}")
    print(f"Manifeste                : {manifest_file}")
    print(f"Prompt                   : {prompt_file}")

    if missing_ids:
        print(
            "Instances encore manquantes : " + ", ".join(missing_ids[:20])
        )
        print(
            "Relance le même script : les prédictions déjà validées seront "
            "ignorées."
        )
        return 2

    metrics, scored_rows, matrix_rows = compute_metrics(
        ordered_instances=instances,
        gold_by_id=gold_by_id,
        predictions=predictions,
    )
    save_scoring_results(
        metrics, scored_rows, matrix_rows,
        output_dir=output_dir, input_file=input_file,
        predictions_file=predictions_file, manifest_file=manifest_file,
    )
    return 0



def save_scoring_results(
    metrics: dict[str, Any], scored_rows: list[dict[str, Any]], matrix_rows: list[list[int]],
    *, output_dir: Path, input_file: Path, predictions_file: Path, manifest_file: Path,
) -> None:
    metrics["scoring_provenance"] = {
        "created_at_utc": utc_now_iso(),
        "input_file": str(input_file.resolve()),
        "input_sha256": sha256_file(input_file),
        "predictions_file": str(predictions_file.resolve()),
        "predictions_sha256": sha256_file(predictions_file),
        "scoring_script_sha256": sha256_file(Path(__file__)),
        "inference_manifest_sha256": sha256_file(manifest_file) if manifest_file.exists() else None,
        "inference_manifest": json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else None,
    }
    write_json_atomic(output_dir / METRICS_FILENAME, metrics)
    write_jsonl_atomic(output_dir / SCORED_PREDICTIONS_FILENAME, scored_rows)
    write_confusion_csv(output_dir / CONFUSION_FILENAME, matrix_rows)
    def fmt(value: float | None) -> str:
        return "N/A (aucun cas admissible)" if value is None else f"{value:.4f}"
    classification, rationale = metrics["classification"], metrics["rationale"]
    print("Résultats DeepSeek — scoring aligné sur mDeBERTa")
    print(f"Instances               : {metrics['instances']}")
    print(f"Accuracy                : {fmt(classification['accuracy'])}")
    print(f"Macro-F1 labels         : {fmt(classification['macro_f1'])}")
    print(f"Cohen kappa             : {fmt(classification['cohen_kappa'])}")
    print(f"S_span (label correct)  : {rationale['s_span']}")
    print(f"Macro token-F1          : {fmt(rationale['macro_token_f1'])}")
    print(f"Macro IoU               : {fmt(rationale['macro_iou'])}")
    print(f"Character EM            : {fmt(rationale['character_exact_match'])}")
    print(f"Token EM (complément)   : {fmt(rationale['token_exact_match'])}")
    print(f"Joint IoU@0.50          : {fmt(rationale['joint_iou_at_0_50'])}")
    print(f"Joint Character EM      : {fmt(rationale['joint_character_exact_match'])}")
    print(f"Métriques               : {output_dir / METRICS_FILENAME}")
    print(f"Prédictions scorées     : {output_dir / SCORED_PREDICTIONS_FILENAME}")
    print("Les prédictions sources et les anciens résultats ne sont pas modifiés par le recalcul.")


def score_existing_predictions(args: argparse.Namespace) -> int:
    if args.project_root is None:
        project_root, split_dir = find_project_root()
    else:
        project_root = args.project_root.resolve()
        split_dir = project_root / SPLIT_RELATIVE_DIRECTORY
    input_file = split_dir / INPUT_FILENAME
    inference_dir = project_root / "data" / OUTPUT_DIRECTORY_NAME
    predictions_file = args.predictions or inference_dir / PREDICTIONS_FILENAME
    output_dir = args.output_dir or inference_dir
    manifest_file = predictions_file.parent / MANIFEST_FILENAME
    instances, gold_by_id = prepare_official_test(load_jsonl(input_file), split_dir=split_dir)
    predictions = load_existing_predictions(predictions_file, {row["id"]: row for row in instances})
    missing = [row["id"] for row in instances if row["id"] not in predictions]
    if missing:
        raise ValueError(
            f"Recalcul impossible : {len(missing)} prédiction(s) manquante(s) "
            f"dans {predictions_file}. Aucun appel API effectué. "
            + ", ".join(missing[:10])
        )
    if manifest_file.exists():
        # Vérifie la provenance sans créer ni réécrire le manifeste d'inférence.
        recorded_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if recorded_manifest.get("input_sha256") != sha256_file(input_file):
            raise ValueError("Le test actuel ne correspond pas au hash du manifeste d'inférence.")
        if recorded_manifest.get("split_protocol") != EXPECTED_SPLIT_PROTOCOL:
            raise ValueError("Le manifeste d'inférence indique un autre protocole de split.")
    else:
        print("Manifeste d'inférence absent : les IDs sont vérifiés, mais le lien avec la version du test n'est pas attesté.", file=sys.stderr)
    # Prevent custom paths from overwriting any input, even with an unusual filename.
    protected = {predictions_file.resolve(), input_file.resolve(), manifest_file.resolve(),
                 (split_dir / SPLIT_SUMMARY_FILENAME).resolve(),
                 (split_dir / SPLIT_MEMBERSHIP_FILENAME).resolve()}
    for filename in (METRICS_FILENAME, SCORED_PREDICTIONS_FILENAME, CONFUSION_FILENAME):
        if (output_dir / filename).resolve() in protected:
            raise ValueError("Le chemin de sortie écraserait un fichier source.")
    metrics, scored_rows, matrix_rows = compute_metrics(
        ordered_instances=instances, gold_by_id=gold_by_id, predictions=predictions,
    )
    save_scoring_results(metrics, scored_rows, matrix_rows, output_dir=output_dir,
                         input_file=input_file, predictions_file=predictions_file,
                         manifest_file=manifest_file)
    print("Recalcul terminé sans appel API.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recalcul local des 150 prédictions DeepSeek avec le protocole mDeBERTa.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--score-only", action="store_true", help="Recalcul sans API (mode par défaut).")
    mode.add_argument("--annotate", action="store_true", help="Autoriser les appels API pour les prédictions manquantes.")
    parser.add_argument("--project-root", type=Path, help="Dossier flexid contenant data/ (recalcul uniquement).")
    parser.add_argument("--predictions", type=Path, help="JSONL des prédictions existantes (recalcul uniquement).")
    parser.add_argument("--output-dir", type=Path, help="Dossier des nouvelles métriques (recalcul uniquement).")
    args = parser.parse_args()
    if args.annotate:
        if args.project_root or args.predictions or args.output_dir:
            parser.error("Avec --annotate, utiliser les chemins du projet d'origine sans options de recalcul.")
        return annotate_and_score()
    return score_existing_predictions(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nInterruption demandée. Les prédictions déjà validées ont été "
            "conservées.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        raise SystemExit(1)
