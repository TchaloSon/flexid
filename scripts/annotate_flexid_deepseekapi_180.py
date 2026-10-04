#!/usr/bin/env python3


from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================================
# CONFIGURATION DE L'EXPÉRIENCE
# ============================================================================

INPUT_FILENAME = (
    "flexid_kappa_rationale_180_unique_law_refs_60_each.jsonl"
)

PREDICTIONS_FILENAME = "flexid_deepseek_v4_flash_predictions.jsonl"
API_LOG_FILENAME = "flexid_deepseek_v4_flash_api_log.jsonl"
FAILURES_FILENAME = "flexid_deepseek_v4_flash_failures.jsonl"
MANIFEST_FILENAME = "flexid_deepseek_v4_flash_run_manifest.json"
PROMPT_FILENAME = "flexid_deepseek_v4_flash_prompt.txt"

EXPECTED_INSTANCE_COUNT = 180

MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"

THINKING_TYPE = "enabled"
REASONING_EFFORT = "high"

# Cette limite comprend le raisonnement et la réponse finale.
MAX_OUTPUT_TOKENS = 8192

MAX_ATTEMPTS_PER_INSTANCE = 4
REQUEST_TIMEOUT_SECONDS = 180.0
MAX_CONSECUTIVE_FAILURES = 5

# Petite pause entre les requêtes pour éviter une rafale inutile.
DELAY_BETWEEN_REQUESTS_SECONDS = 0.15

ALLOWED_LABELS = {"entailment", "contradiction", "neutral"}
NON_NEUTRAL_LABELS = {"entailment", "contradiction"}

# Ces champs ne doivent jamais apparaître dans le fichier envoyé au modèle.
FORBIDDEN_INPUT_FIELDS = {
    "label",
    "rationale_start",
    "rationale_end",
    "rationale_text",
    "rationale_start_char",
    "rationale_end_char",
    "rationale_start_token",
    "rationale_end_token",
    "rationale_token_text",
    "rationale_token_ids",
}

TOKEN_LINE_PATTERN = re.compile(r"(?m)^\[(\d+)\]\s+\S")


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
4. rationale_start_token et rationale_end_token sont des bornes inclusives.
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
# CHEMINS ET UTILITAIRES
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


def find_project_root() -> Path:
    """
    Détecte la racine du projet sans chemin absolu.

    Le script peut être placé dans scripts/ ou directement à la racine.
    """
    script_dir = Path(__file__).resolve().parent
    current_dir = Path.cwd().resolve()

    candidates: list[Path] = []
    for starting_point in (script_dir, current_dir):
        candidates.append(starting_point)
        candidates.extend(starting_point.parents)

    seen: set[Path] = set()

    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        expected_input = candidate / "data" / INPUT_FILENAME
        if expected_input.is_file():
            return candidate

    raise FileNotFoundError(
        "Impossible de trouver le fichier d'entrée attendu :\n"
        f"  data/{INPUT_FILENAME}\n"
        "Place le script dans scripts/ ou à la racine du projet."
    )


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Ajoute une ligne JSON et force son écriture sur disque."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
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


def write_jsonl_atomic(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
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


# ============================================================================
# LECTURE ET VALIDATION DU FICHIER D'ENTRÉE
# ============================================================================

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
                    f"{path}, ligne {line_number} : un objet JSON est attendu."
                )

            records.append(record)

    if not records:
        raise ValueError(f"Aucune instance trouvée dans {path}")

    return records


def render_json_value(value: Any) -> str:
    """
    Rend proprement une chaîne, une liste ou un objet JSON dans le prompt.
    """
    if isinstance(value, str):
        return value.strip()

    return json.dumps(value, ensure_ascii=False, indent=2)


def extract_token_ids(tokenized_premise: str, record_id: str) -> list[int]:
    token_ids = [
        int(match.group(1))
        for match in TOKEN_LINE_PATTERN.finditer(tokenized_premise)
    ]

    if not token_ids:
        raise ValueError(
            f"{record_id}: aucun token numéroté n'a été détecté dans "
            "tokenized_premise."
        )

    expected = list(range(1, len(token_ids) + 1))

    if token_ids != expected:
        raise ValueError(
            f"{record_id}: les tokens doivent être consécutifs de 1 à "
            f"{len(token_ids)}. Début détecté : {token_ids[:10]}"
        )

    return token_ids


def prepare_input_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(records) != EXPECTED_INSTANCE_COUNT:
        raise ValueError(
            f"Le fichier doit contenir exactement {EXPECTED_INSTANCE_COUNT} "
            f"instances, mais {len(records)} ont été trouvées."
        )

    prepared: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for position, record in enumerate(records, start=1):
        leaked_fields = sorted(FORBIDDEN_INPUT_FIELDS.intersection(record))
        if leaked_fields:
            raise ValueError(
                f"Instance #{position}: le fichier destiné au modèle contient "
                "des champs gold interdits : "
                + ", ".join(leaked_fields)
            )

        record_id = str(record.get("id", "")).strip()
        if not record_id:
            raise ValueError(f"Instance #{position}: champ id absent.")

        if record_id in seen_ids:
            raise ValueError(f"Identifiant dupliqué : {record_id}")
        seen_ids.add(record_id)

        tokenized_premise = record.get("tokenized_premise")
        if not isinstance(tokenized_premise, str) or not tokenized_premise.strip():
            raise ValueError(
                f"{record_id}: tokenized_premise absent ou vide."
            )

        if "hypothesis_facts" not in record:
            raise ValueError(f"{record_id}: hypothesis_facts absent.")

        hypothesis_facts = render_json_value(record["hypothesis_facts"])
        if not hypothesis_facts:
            raise ValueError(f"{record_id}: hypothesis_facts vide.")

        token_ids = extract_token_ids(tokenized_premise, record_id)

        prepared.append(
            {
                "id": record_id,
                "tokenized_premise": tokenized_premise.strip(),
                "hypothesis_facts": hypothesis_facts,
                "max_token_id": token_ids[-1],
            }
        )

    return prepared


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
{instance["id"]}

PRÉMISSE TOKENISÉE
{instance["tokenized_premise"]}

HYPOTHÈSE ET FAITS EXPLICITEMENT DÉCRITS
{instance["hypothesis_facts"]}

Le dernier numéro de token valide dans la prémisse est
{instance["max_token_id"]}.

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
        raise ValueError(
            f"{record_id}: {field_name} ne peut pas être booléen."
        )

    if isinstance(value, int):
        return value

    # Le modèle doit produire un entier JSON, pas 8.0 ni "8".
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
            raise ValueError(
                "pour neutral, les deux bornes doivent être null"
            )

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
                f"rationale_end_token={end} dépasse le dernier token "
                f"valide {instance['max_token_id']}"
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
            raise ValueError(
                f"{path}: prédiction dupliquée pour {record_id}."
            )

        predictions[record_id] = validate_prediction(
            payload,
            instances_by_id[record_id],
        )

    return predictions


def build_manifest(input_file: Path) -> dict[str, Any]:
    return {
        "experiment": "FLEXID DeepSeek V4 Flash annotation",
        "created_at_utc": utc_now_iso(),
        "input_file": str(input_file),
        "input_sha256": sha256_file(input_file),
        "expected_instances": EXPECTED_INSTANCE_COUNT,
        "model": MODEL,
        "base_url": BASE_URL,
        "thinking": {"type": THINKING_TYPE},
        "reasoning_effort": REASONING_EFFORT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_attempts_per_instance": MAX_ATTEMPTS_PER_INSTANCE,
        "response_format": {"type": "json_object"},
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "prediction_schema": {
            "id": "string",
            "label": sorted(ALLOWED_LABELS),
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
    """
    Empêche de mélanger des prédictions produites avec un autre prompt,
    un autre fichier d'entrée ou un autre modèle.
    """
    if not path.exists():
        write_json_atomic(path, expected_manifest)
        return

    existing = json.loads(path.read_text(encoding="utf-8"))

    stable_keys = (
        "input_sha256",
        "expected_instances",
        "model",
        "base_url",
        "thinking",
        "reasoning_effort",
        "max_output_tokens",
        "response_format",
        "system_prompt_sha256",
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
            "actuelle. Pour éviter de mélanger deux expériences, archive ou "
            "supprime les anciennes sorties. Champs différents : "
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

    completion_details = getattr(
        usage,
        "completion_tokens_details",
        None,
    )
    if completion_details is not None:
        reasoning_tokens = getattr(
            completion_details,
            "reasoning_tokens",
            None,
        )
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
            instance,
            validation_error=last_validation_error,
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
                extra_body={
                    "thinking": {
                        "type": THINKING_TYPE,
                    }
                },
                stream=False,
            )

            elapsed_seconds = time.monotonic() - start_time

            if not response.choices:
                raise ValueError("la réponse API ne contient aucun choix")

            choice = response.choices[0]
            message = choice.message
            content = message.content

            try:
                prediction = parse_and_validate_response(
                    content,
                    instance,
                )
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
                            content[:2000]
                            if isinstance(content, str)
                            else None
                        ),
                        "model_requested": MODEL,
                        "model_returned": getattr(
                            response,
                            "model",
                            None,
                        ),
                        "response_id": getattr(response, "id", None),
                        "system_fingerprint": getattr(
                            response,
                            "system_fingerprint",
                            None,
                        ),
                        "finish_reason": getattr(
                            choice,
                            "finish_reason",
                            None,
                        ),
                        "usage": usage_to_dict(
                            getattr(response, "usage", None)
                        ),
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
                        response,
                        "system_fingerprint",
                        None,
                    ),
                    "finish_reason": getattr(
                        choice,
                        "finish_reason",
                        None,
                    ),
                    "usage": usage_to_dict(
                        getattr(response, "usage", None)
                    ),
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

            # Les erreurs d'authentification ou de permission ne seront pas
            # corrigées par une nouvelle tentative.
            if status_code in {401, 403}:
                raise RuntimeError(
                    "Authentification DeepSeek refusée. Vérifie la variable "
                    "DEEPSEEK_API_KEY."
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

            last_validation_error = (
                f"{type(exc).__name__}: {str(exc)[:500]}"
            )
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
# EXÉCUTION
# ============================================================================

def main() -> int:
    project_root = find_project_root()
    data_dir = project_root / "data"

    input_file = data_dir / INPUT_FILENAME
    predictions_file = data_dir / PREDICTIONS_FILENAME
    api_log_file = data_dir / API_LOG_FILENAME
    failures_file = data_dir / FAILURES_FILENAME
    manifest_file = data_dir / MANIFEST_FILENAME
    prompt_file = data_dir / PROMPT_FILENAME

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "La variable d'environnement DEEPSEEK_API_KEY est absente.\n"
            "Sous PowerShell :\n"
            '  $env:DEEPSEEK_API_KEY="votre_clé"'
        )

    raw_records = load_jsonl(input_file)
    instances = prepare_input_records(raw_records)
    instances_by_id = {
        instance["id"]: instance
        for instance in instances
    }

    manifest = build_manifest(input_file)
    verify_or_create_manifest(manifest_file, manifest)

    # Conserver le prompt exact utilisé dans l'expérience.
    if prompt_file.exists():
        existing_prompt = prompt_file.read_text(encoding="utf-8")
        if existing_prompt != SYSTEM_PROMPT:
            raise RuntimeError(
                "Le fichier de prompt existant diffère du prompt actuel. "
                "Archive ou supprime les anciennes sorties avant une nouvelle "
                "expérience."
            )
    else:
        prompt_file.write_text(SYSTEM_PROMPT, encoding="utf-8")

    predictions = load_existing_predictions(
        predictions_file,
        instances_by_id,
    )

    OpenAI, APIConnectionError, APIStatusError, RateLimitError = (
        import_openai()
    )

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    pending = [
        instance
        for instance in instances
        if instance["id"] not in predictions
    ]

    print("Annotation FLEXID avec DeepSeek-V4-Flash")
    print(f"Projet                   : {project_root}")
    print(f"Entrée                   : {input_file}")
    print(f"Modèle                   : {MODEL}")
    print(f"Thinking                 : {THINKING_TYPE}")
    print(f"Reasoning effort         : {REASONING_EFFORT}")
    print(f"Instances totales       : {len(instances)}")
    print(f"Déjà validées           : {len(predictions)}")
    print(f"À traiter               : {len(pending)}")
    print("")

    consecutive_failures = 0
    run_failures = 0

    api_exception_types = (
        APIConnectionError,
        APIStatusError,
        RateLimitError,
    )

    for pending_index, instance in enumerate(pending, start=1):
        global_position = next(
            index
            for index, candidate in enumerate(instances, start=1)
            if candidate["id"] == instance["id"]
        )

        print(
            f"[{global_position:03d}/{len(instances)}] "
            f"{instance['id']}...",
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
                    f"{MAX_CONSECUTIVE_FAILURES} échecs consécutifs. "
                    "Arrêt préventif pour éviter de multiplier les appels. "
                    "Relance le script après avoir examiné le journal."
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

    # Réécriture canonique dans l'ordre exact du fichier d'entrée.
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
    print("Exécution terminée.")
    print(f"Prédictions validées     : {completed}/{len(instances)}")
    print(f"Échecs pendant ce run    : {run_failures}")
    print(f"Prédictions              : {predictions_file}")
    print(f"Journal API              : {api_log_file}")
    print(f"Échecs                   : {failures_file}")
    print(f"Manifeste                : {manifest_file}")
    print(f"Prompt                   : {prompt_file}")

    if missing_ids:
        print(
            "Instances encore manquantes : "
            + ", ".join(missing_ids[:20])
        )
        print(
            "Relance le même script : les prédictions déjà validées "
            "seront ignorées."
        )
        return 2

    print(
        "Les 180 prédictions sont complètes et ordonnées comme le "
        "fichier d'entrée."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nInterruption demandée. Les prédictions déjà validées ont "
            "été conservées.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        raise SystemExit(1)
