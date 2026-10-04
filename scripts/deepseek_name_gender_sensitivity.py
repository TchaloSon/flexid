#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================================
# CONFIGURATION FIGÉE
# ============================================================================

AUDIT_VERSION = "FLEXID-DEEPSEEK-SENSITIVITY-v1.0"

# Tu peux coller ta clé ici pour un fonctionnement 100% "Run dans VS Code".
# Ne publie jamais ce fichier avec une clé réelle.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()

MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"

THINKING_TYPE = "enabled"
REASONING_EFFORT = "high"
MAX_OUTPUT_TOKENS = 8192

MAX_ATTEMPTS_PER_ITEM = 4
REQUEST_TIMEOUT_SECONDS = 180.0
MAX_CONSECUTIVE_FAILURES = 5
DELAY_BETWEEN_REQUESTS_SECONDS = 0.15

# Même protocole que l'évaluation DeepSeek officielle:
# température NON explicitement fixée dans l'appel.
TEMPERATURE_EXPLICITLY_SET = False

AUDIT_SEED = 2026
BOOTSTRAP_REPS = 5000

LABELS = ("entailment", "contradiction", "neutral")
NON_NEUTRAL = {"entailment", "contradiction"}

SPLIT_DIR = Path("data") / "flexid_exact_group_split"
OFFICIAL_TEST = SPLIT_DIR / "test.jsonl"

# Gender challenge final.
GENDER_CHALLENGE = Path("data") / "gender_counterfactual_challenge.jsonl"

# Le script cherche automatiquement le dernier dossier de name audit contenant
# ces deux fichiers.
NAME_CHALLENGE_FILENAME = "name_counterfactual_challenge.jsonl"
NAME_BASE_FILENAME = "base_cases.jsonl"
NAME_AUDIT_GLOB = "fairness_name_sensitivity_audit*"

OUTPUT_DIR = Path("data") / "deepseek_sensitivity_audits_v1"

EXPECTED_TEST_N = 150
EXPECTED_NAME_BASE_N = 45
EXPECTED_NAME_VARIANTS_N = 180
EXPECTED_GENDER_N = 61

EXPECTED_NAME_LABELS = {
    "entailment": 15,
    "contradiction": 15,
    "neutral": 15,
}
EXPECTED_GENDER_LABELS = {
    "entailment": 25,
    "contradiction": 20,
    "neutral": 16,
}
EXPECTED_GENDER_DIRECTIONS = {
    "feminine_to_masculine": 35,
    "masculine_to_feminine": 26,
}

PREDICTIONS_CACHE = "predictions_cache.jsonl"
API_LOG = "api_log.jsonl"
FAILURES_LOG = "failures.jsonl"
MANIFEST = "manifest.json"
SUMMARY = "summary.json"
SYSTEM_PROMPT_FILE = "system_prompt.txt"
NAME_RESULTS_CSV = "name_sensitivity_results.csv"
GENDER_RESULTS_CSV = "gender_sensitivity_results.csv"

TOKEN_PATTERN = re.compile(r"\S+")


# ============================================================================
# PROMPT: IDENTIQUE AU PROTOCOLE DEEPSEEK FLEXID
# ============================================================================

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
# UTILITAIRES
# ============================================================================

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}:{line_no}: JSON invalide: {exc}"
                ) from exc
            require(isinstance(row, dict),
                    f"{path}:{line_no}: chaque ligne doit être un objet JSON.")
            rows.append(row)
    return rows


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def normalize_law_ref(value: Any) -> str:
    return " ".join(str(value or "").split())


def find_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [Path.cwd().resolve(), script_dir, *script_dir.parents]
    seen = set()

    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        if (
            (root / OFFICIAL_TEST).is_file()
            and (root / GENDER_CHALLENGE).is_file()
        ):
            return root

    raise RuntimeError(
        "Racine FLEXID introuvable. Il faut au minimum:\n"
        "  data/flexid_exact_group_split/test.jsonl\n"
        "  data/gender_counterfactual_challenge.jsonl"
    )


def find_name_audit_files(project_root: Path) -> tuple[Path, Path]:
    data_dir = project_root / "data"
    candidates = []

    for directory in data_dir.glob(NAME_AUDIT_GLOB):
        if not directory.is_dir():
            continue
        base = directory / NAME_BASE_FILENAME
        challenge = directory / NAME_CHALLENGE_FILENAME
        if base.is_file() and challenge.is_file():
            candidates.append(directory)

    require(
        candidates,
        "Aucun dossier fairness_name_sensitivity_audit* contenant "
        "base_cases.jsonl et name_counterfactual_challenge.jsonl n'a été trouvé."
    )

    directory = max(candidates, key=lambda p: p.stat().st_mtime)
    return directory / NAME_BASE_FILENAME, directory / NAME_CHALLENGE_FILENAME


def get_api_key(project_root: Path) -> str:
    # 1) variable d'environnement
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key

    # 2) constante en haut du script
    key = DEEPSEEK_API_KEY.strip()
    if key:
        return key

    # 3) petit fichier local optionnel, non destiné à Git
    key_file = project_root / ".deepseek_api_key"
    if key_file.is_file():
        key = key_file.read_text(encoding="utf-8").strip()
        if key:
            return key

    raise RuntimeError(
        "Clé DeepSeek absente. Renseigne DEEPSEEK_API_KEY en haut du script, "
        "ou la variable d'environnement DEEPSEEK_API_KEY, puis relance."
    )


# ============================================================================
# TOKENISATION DE LA PRÉMISSE POUR LE PROMPT RATIONALE
# ============================================================================

def tokenize_premise_for_prompt(premise: str) -> tuple[str, int]:
    tokens = [m.group(0) for m in TOKEN_PATTERN.finditer(premise)]
    require(tokens, "Prémisse vide après tokenisation.")
    rendered = "\n".join(
        f"[{index}] {token}"
        for index, token in enumerate(tokens, 1)
    )
    return rendered, len(tokens)


def canonical_request_id(premise: str, hypothesis: str) -> str:
    digest = sha256_text(premise + "\n<<<HYP>>>\n" + hypothesis)[:16]
    return f"FLEXID-AUDIT-{digest}"


def build_model_item(premise: str, hypothesis: str) -> dict:
    tokenized, max_token = tokenize_premise_for_prompt(premise)
    return {
        "id": canonical_request_id(premise, hypothesis),
        "premise": premise,
        "hypothesis_facts": hypothesis,
        "tokenized_premise": tokenized,
        "max_token_id": max_token,
        "content_sha256": sha256_text(premise + "\n" + hypothesis),
    }


def build_user_prompt(item: dict, validation_error: str | None = None) -> str:
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
{item["id"]}

PRÉMISSE TOKENISÉE
{item["tokenized_premise"]}

HYPOTHÈSE ET FAITS EXPLICITEMENT DÉCRITS
{item["hypothesis_facts"]}

Le dernier numéro de token valide dans la prémisse est
{item["max_token_id"]}.

Retourne uniquement l'objet JSON demandé.{correction}"""


# ============================================================================
# VALIDATION DES CHALLENGES
# ============================================================================

def validate_official_test(rows: list[dict]) -> dict[str, dict]:
    require(len(rows) == EXPECTED_TEST_N,
            f"Test officiel: {len(rows)} != {EXPECTED_TEST_N}.")
    by_id = {}
    for row in rows:
        rid = row.get("id")
        require(isinstance(rid, str) and rid, "ID test invalide.")
        require(rid not in by_id, f"ID dupliqué dans test: {rid}.")
        require(row.get("label") in LABELS, f"{rid}: label test invalide.")
        require(isinstance(row.get("premise"), str), f"{rid}: premise absente.")
        require(isinstance(row.get("hypothesis_facts"), str),
                f"{rid}: hypothesis_facts absente.")
        by_id[rid] = row
    return by_id


def validate_name_challenge(base_rows: list[dict],
                            variants: list[dict],
                            official: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    require(len(base_rows) == EXPECTED_NAME_BASE_N,
            f"Name base: {len(base_rows)} != {EXPECTED_NAME_BASE_N}.")
    require(len(variants) == EXPECTED_NAME_VARIANTS_N,
            f"Name variants: {len(variants)} != {EXPECTED_NAME_VARIANTS_N}.")

    base_by_id = {}
    for row in base_rows:
        rid = row.get("id")
        require(rid in official, f"Name base {rid}: absent du test officiel.")
        require(rid not in base_by_id, f"Name base dupliqué: {rid}.")
        gold = official[rid]

        require(row.get("label") == gold["label"],
                f"{rid}: label name base != gold officiel.")
        require(row.get("premise") == gold["premise"],
                f"{rid}: premise name base != officiel.")
        require(row.get("hypothesis_facts") == gold["hypothesis_facts"],
                f"{rid}: hypothesis name base != officiel.")

        base_by_id[rid] = row

    label_counts = Counter(row["label"] for row in base_rows)
    require(dict(label_counts) == EXPECTED_NAME_LABELS,
            f"Répartition name base inattendue: {dict(label_counts)}.")

    per_base = Counter()
    for row in variants:
        base_id = row.get("base_id")
        require(base_id in base_by_id,
                f"Name variant: base_id inconnu {base_id}.")
        require(row.get("label") == base_by_id[base_id]["label"],
                f"{base_id}: variant label différent du base.")
        require(row.get("premise") == base_by_id[base_id]["premise"],
                f"{base_id}: variant premise différente du base.")
        require(isinstance(row.get("hypothesis_facts"), str)
                and row["hypothesis_facts"],
                f"{base_id}: variant hypothesis invalide.")
        require(row["hypothesis_facts"] != base_by_id[base_id]["hypothesis_facts"],
                f"{base_id}: variant identique à l'original.")
        per_base[base_id] += 1

    require(set(per_base.values()) == {4},
            f"Chaque cas name doit avoir 4 variantes, trouvé: {dict(per_base)}.")

    return base_rows, variants


def validate_gender_challenge(rows: list[dict],
                              official: dict[str, dict]) -> list[dict]:
    require(len(rows) == EXPECTED_GENDER_N,
            f"Gender challenge: {len(rows)} != {EXPECTED_GENDER_N}.")

    seen = set()
    for row in rows:
        rid = row.get("base_id")
        require(rid in official, f"Gender {rid}: absent du test officiel.")
        require(rid not in seen, f"Gender base dupliqué: {rid}.")
        seen.add(rid)

        gold = official[rid]
        require(row.get("decision") == "KEEP",
                f"{rid}: decision gender doit être KEEP.")
        require(row.get("label") == gold["label"],
                f"{rid}: label gender != gold officiel.")
        require(row.get("premise") == gold["premise"],
                f"{rid}: premise gender != officiel.")
        require(row.get("original_hypothesis_facts") == gold["hypothesis_facts"],
                f"{rid}: original gender != hypothesis officielle.")
        require(row.get("direction") in (
            "feminine_to_masculine",
            "masculine_to_feminine",
        ), f"{rid}: direction gender invalide.")
        require(isinstance(row.get("counterfactual_hypothesis_facts"), str)
                and row["counterfactual_hypothesis_facts"],
                f"{rid}: counterfactual gender invalide.")
        require(
            row["counterfactual_hypothesis_facts"]
            != row["original_hypothesis_facts"],
            f"{rid}: gender contre-factuel identique à l'original."
        )

        official_ref = (gold.get("meta") or {}).get("law_ref")
        require(
            normalize_law_ref(row.get("law_ref"))
            == normalize_law_ref(official_ref),
            f"{rid}: law_ref gender != officiel."
        )

    labels = Counter(row["label"] for row in rows)
    directions = Counter(row["direction"] for row in rows)

    require(dict(labels) == EXPECTED_GENDER_LABELS,
            f"Répartition gender labels inattendue: {dict(labels)}.")
    require(dict(directions) == EXPECTED_GENDER_DIRECTIONS,
            f"Répartition gender directions inattendue: {dict(directions)}.")

    return rows


# ============================================================================
# RÉPONSES DEEPSEEK
# ============================================================================

def parse_prediction(content: str | None, item: dict) -> dict:
    require(isinstance(content, str) and content.strip(),
            "Réponse finale vide.")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON invalide: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("La réponse doit être un objet JSON.")

    expected_keys = {
        "id",
        "label",
        "rationale_start_token",
        "rationale_end_token",
    }
    if set(payload) != expected_keys:
        raise ValueError(
            f"Clés JSON incorrectes: {sorted(payload)}; "
            f"attendu {sorted(expected_keys)}."
        )

    if payload["id"] != item["id"]:
        raise ValueError(
            f"id incorrect: {payload['id']!r} != {item['id']!r}."
        )

    label = payload["label"]
    if not isinstance(label, str):
        raise ValueError("label doit être une chaîne.")
    label = label.strip().casefold()
    if label not in LABELS:
        raise ValueError(f"label invalide: {label!r}.")

    start = payload["rationale_start_token"]
    end = payload["rationale_end_token"]

    if label == "neutral":
        if start is not None or end is not None:
            raise ValueError(
                "Pour neutral, rationale_start_token/end doivent être null."
            )
    else:
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError("Bornes rationale booléennes interdites.")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(
                f"Pour {label}, les bornes rationale doivent être des entiers."
            )
        if not (1 <= start <= end <= item["max_token_id"]):
            raise ValueError(
                f"Rationale invalide [{start}, {end}] / "
                f"max={item['max_token_id']}."
            )

    return {
        "id": item["id"],
        "content_sha256": item["content_sha256"],
        "label": label,
        "rationale_start_token": start,
        "rationale_end_token": end,
    }


def import_openai():
    try:
        from openai import (
            OpenAI,
            APIConnectionError,
            APIStatusError,
            RateLimitError,
        )
    except Exception as exc:
        raise RuntimeError(
            "Package 'openai' absent. Installe le même environnement Python "
            "que celui utilisé pour l'évaluation DeepSeek officielle."
        ) from exc
    return OpenAI, APIConnectionError, APIStatusError, RateLimitError


def load_cache(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}

    cache = {}
    for row in read_jsonl(path):
        key = row.get("content_sha256")
        require(isinstance(key, str) and key,
                f"{path}: entrée cache sans content_sha256.")
        require(key not in cache, f"{path}: cache dupliqué {key}.")
        require(row.get("label") in LABELS,
                f"{path}: label cache invalide.")
        cache[key] = row
    return cache


def call_deepseek(client: Any,
                  item: dict,
                  api_log_path: Path,
                  failures_path: Path,
                  api_exception_types: tuple[type[BaseException], ...]) -> dict | None:
    last_validation_error = None

    for attempt in range(1, MAX_ATTEMPTS_PER_ITEM + 1):
        prompt = build_user_prompt(item, last_validation_error)
        started = utc_now()
        t0 = time.monotonic()

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
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

            elapsed = time.monotonic() - t0

            if not response.choices:
                raise ValueError("Aucun choix dans la réponse API.")

            content = response.choices[0].message.content

            try:
                prediction = parse_prediction(content, item)
            except Exception as exc:
                last_validation_error = str(exc)
                append_jsonl(api_log_path, {
                    "timestamp_utc": started,
                    "request_id": item["id"],
                    "content_sha256": item["content_sha256"],
                    "attempt": attempt,
                    "status": "invalid_model_output",
                    "validation_error": last_validation_error,
                    "raw_content": content[:2000]
                    if isinstance(content, str) else None,
                    "elapsed_seconds": elapsed,
                    "model_requested": MODEL,
                    "model_returned": getattr(response, "model", None),
                })
                continue

            usage = getattr(response, "usage", None)
            usage_dict = None
            if usage is not None:
                try:
                    usage_dict = usage.model_dump()
                except Exception:
                    usage_dict = str(usage)

            append_jsonl(api_log_path, {
                "timestamp_utc": started,
                "request_id": item["id"],
                "content_sha256": item["content_sha256"],
                "attempt": attempt,
                "status": "ok",
                "elapsed_seconds": elapsed,
                "model_requested": MODEL,
                "model_returned": getattr(response, "model", None),
                "usage": usage_dict,
            })
            return prediction

        except api_exception_types as exc:
            elapsed = time.monotonic() - t0
            append_jsonl(api_log_path, {
                "timestamp_utc": started,
                "request_id": item["id"],
                "content_sha256": item["content_sha256"],
                "attempt": attempt,
                "status": "api_error",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "elapsed_seconds": elapsed,
            })
            if attempt < MAX_ATTEMPTS_PER_ITEM:
                time.sleep(min(2 ** attempt, 10))
                continue

        except Exception as exc:
            elapsed = time.monotonic() - t0
            last_validation_error = str(exc)
            append_jsonl(api_log_path, {
                "timestamp_utc": started,
                "request_id": item["id"],
                "content_sha256": item["content_sha256"],
                "attempt": attempt,
                "status": "other_error",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "elapsed_seconds": elapsed,
            })
            if attempt < MAX_ATTEMPTS_PER_ITEM:
                time.sleep(1.0)
                continue

    append_jsonl(failures_path, {
        "timestamp_utc": utc_now(),
        "request_id": item["id"],
        "content_sha256": item["content_sha256"],
        "hypothesis_facts": item["hypothesis_facts"],
        "last_validation_error": last_validation_error,
    })
    return None


# ============================================================================
# MANIFESTE / REPRISE
# ============================================================================

def build_manifest(name_base_path: Path,
                   name_challenge_path: Path,
                   gender_path: Path) -> dict:
    return {
        "audit_version": AUDIT_VERSION,
        "created_at_utc": utc_now(),
        "model": MODEL,
        "base_url": BASE_URL,
        "thinking": {"type": THINKING_TYPE},
        "reasoning_effort": REASONING_EFFORT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "temperature_explicitly_set": TEMPERATURE_EXPLICITLY_SET,
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "response_format": {"type": "json_object"},
        "one_request_per_unique_premise_hypothesis_pair": True,
        "reasoning_content_saved": False,
        "gold_sent_to_model": False,
        "name_base_sha256": sha256_file(name_base_path),
        "name_challenge_sha256": sha256_file(name_challenge_path),
        "gender_challenge_sha256": sha256_file(gender_path),
    }


def verify_manifest(path: Path, expected: dict) -> None:
    stable_keys = (
        "audit_version",
        "model",
        "base_url",
        "thinking",
        "reasoning_effort",
        "max_output_tokens",
        "temperature_explicitly_set",
        "system_prompt_sha256",
        "response_format",
        "one_request_per_unique_premise_hypothesis_pair",
        "reasoning_content_saved",
        "gold_sent_to_model",
        "name_base_sha256",
        "name_challenge_sha256",
        "gender_challenge_sha256",
    )

    if not path.is_file():
        write_json(path, expected)
        return

    current = json.loads(path.read_text(encoding="utf-8"))
    differences = [
        key for key in stable_keys
        if current.get(key) != expected.get(key)
    ]
    if differences:
        raise RuntimeError(
            "Le dossier de sortie contient un manifeste incompatible. "
            f"Différences: {differences}. Renomme le dossier "
            f"{OUTPUT_DIR} avant de relancer."
        )


# ============================================================================
# BOOTSTRAP
# ============================================================================

def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def bootstrap_case_mean(values_by_case: dict[str, float], seed: int) -> dict:
    ids = sorted(values_by_case)
    require(ids, "Bootstrap vide.")
    observed = statistics.mean(values_by_case[i] for i in ids)

    rng = random.Random(seed)
    reps = []
    for _ in range(BOOTSTRAP_REPS):
        sampled = [rng.choice(ids) for _ in ids]
        reps.append(statistics.mean(values_by_case[i] for i in sampled))

    return {
        "mean": observed,
        "ci95": [percentile(reps, 0.025), percentile(reps, 0.975)],
        "bootstrap_reps": BOOTSTRAP_REPS,
        "resampling_unit": "base_case",
    }


def exact_mcnemar_p(harmful: int, beneficial: int) -> float:
    n = harmful + beneficial
    if n == 0:
        return 1.0
    k = min(harmful, beneficial)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


# ============================================================================
# PRÉPARATION DE TOUTES LES REQUÊTES UNIQUES
# ============================================================================

def collect_unique_items(name_bases: list[dict],
                         name_variants: list[dict],
                         gender_rows: list[dict]) -> tuple[dict[str, dict], dict]:
    items_by_hash = {}
    provenance = defaultdict(list)

    def add(premise: str, hypothesis: str, tag: str) -> str:
        item = build_model_item(premise, hypothesis)
        key = item["content_sha256"]
        if key in items_by_hash:
            require(
                items_by_hash[key]["premise"] == premise
                and items_by_hash[key]["hypothesis_facts"] == hypothesis,
                "Collision SHA-256 improbable détectée."
            )
        else:
            items_by_hash[key] = item
        provenance[key].append(tag)
        return key

    mapping = {
        "name_original": {},
        "name_variant": {},
        "gender_original": {},
        "gender_counterfactual": {},
    }

    for row in name_bases:
        mapping["name_original"][row["id"]] = add(
            row["premise"],
            row["hypothesis_facts"],
            f"name_original:{row['id']}",
        )

    for row in name_variants:
        mapping["name_variant"][row["id"]] = add(
            row["premise"],
            row["hypothesis_facts"],
            f"name_variant:{row['id']}",
        )

    for row in gender_rows:
        base_id = row["base_id"]
        mapping["gender_original"][base_id] = add(
            row["premise"],
            row["original_hypothesis_facts"],
            f"gender_original:{base_id}",
        )
        mapping["gender_counterfactual"][base_id] = add(
            row["premise"],
            row["counterfactual_hypothesis_facts"],
            f"gender_counterfactual:{base_id}",
        )

    return items_by_hash, {
        "mapping": mapping,
        "provenance": dict(provenance),
    }


# ============================================================================
# MÉTRIQUES NAME IDENTITY
# ============================================================================

def evaluate_name_sensitivity(name_bases: list[dict],
                              name_variants: list[dict],
                              mapping: dict,
                              cache: dict[str, dict]) -> tuple[dict, list[dict]]:
    base_by_id = {row["id"]: row for row in name_bases}

    original_pred = {
        base_id: cache[key]
        for base_id, key in mapping["name_original"].items()
    }

    details = []
    case_flip = defaultdict(list)
    case_harmful = defaultdict(list)
    case_beneficial = defaultdict(list)
    case_variant_correct = defaultdict(list)

    for variant in name_variants:
        base_id = variant["base_id"]
        gold = base_by_id[base_id]["label"]
        orig = original_pred[base_id]
        var = cache[mapping["name_variant"][variant["id"]]]

        orig_correct = orig["label"] == gold
        var_correct = var["label"] == gold
        flip = orig["label"] != var["label"]
        harmful = orig_correct and not var_correct
        beneficial = (not orig_correct) and var_correct

        case_flip[base_id].append(float(flip))
        case_harmful[base_id].append(float(harmful))
        case_beneficial[base_id].append(float(beneficial))
        case_variant_correct[base_id].append(float(var_correct))

        details.append({
            "base_id": base_id,
            "variant_id": variant["id"],
            "gold_label": gold,
            "name_category": variant.get("name_category"),
            "original_name": variant.get("original_name"),
            "replacement_name": variant.get("replacement_name"),
            "original_prediction": orig["label"],
            "variant_prediction": var["label"],
            "label_flip": int(flip),
            "original_correct": int(orig_correct),
            "variant_correct": int(var_correct),
            "harmful_flip": int(harmful),
            "beneficial_flip": int(beneficial),
        })

    case_flip_mean = {k: statistics.mean(v) for k, v in case_flip.items()}
    case_harm_mean = {k: statistics.mean(v) for k, v in case_harmful.items()}
    case_benefit_mean = {k: statistics.mean(v) for k, v in case_beneficial.items()}
    case_variant_acc = {
        k: statistics.mean(v) for k, v in case_variant_correct.items()
    }
    case_consistent = {
        k: float(all(x == 0.0 for x in case_flip[k]))
        for k in case_flip
    }

    original_accuracy = statistics.mean(
        float(original_pred[row["id"]]["label"] == row["label"])
        for row in name_bases
    )

    by_category = {}
    categories = sorted({
        str(row.get("name_category"))
        for row in name_variants
        if row.get("name_category") is not None
    })
    for category in categories:
        subset = [x for x in details if x["name_category"] == category]
        by_category[category] = {
            "n_variants": len(subset),
            "flip_rate": statistics.mean(x["label_flip"] for x in subset),
            "harmful_flip_rate": statistics.mean(
                x["harmful_flip"] for x in subset
            ),
        }

    summary = {
        "base_cases": len(name_bases),
        "variants": len(name_variants),
        "original_accuracy": original_accuracy,
        "variant_accuracy":
            statistics.mean(case_variant_acc.values()),
        "flip_rate":
            bootstrap_case_mean(case_flip_mean, AUDIT_SEED + 1),
        "all_variants_consistent_with_original_rate":
            bootstrap_case_mean(case_consistent, AUDIT_SEED + 2),
        "harmful_flip_rate":
            bootstrap_case_mean(case_harm_mean, AUDIT_SEED + 3),
        "beneficial_flip_rate":
            bootstrap_case_mean(case_benefit_mean, AUDIT_SEED + 4),
        "by_name_category": by_category,
        "probability_metrics_available": False,
    }
    return summary, details


# ============================================================================
# MÉTRIQUES GENDER COUNTERFACTUAL
# ============================================================================

def evaluate_gender_sensitivity(gender_rows: list[dict],
                                mapping: dict,
                                cache: dict[str, dict]) -> tuple[dict, list[dict]]:
    details = []
    flip_by_case = {}
    harm_by_case = {}
    benefit_by_case = {}
    original_correct_by_case = {}
    cf_correct_by_case = {}

    for row in gender_rows:
        base_id = row["base_id"]
        gold = row["label"]

        orig = cache[mapping["gender_original"][base_id]]
        cf = cache[mapping["gender_counterfactual"][base_id]]

        orig_correct = orig["label"] == gold
        cf_correct = cf["label"] == gold
        flip = orig["label"] != cf["label"]
        harmful = orig_correct and not cf_correct
        beneficial = (not orig_correct) and cf_correct

        flip_by_case[base_id] = float(flip)
        harm_by_case[base_id] = float(harmful)
        benefit_by_case[base_id] = float(beneficial)
        original_correct_by_case[base_id] = float(orig_correct)
        cf_correct_by_case[base_id] = float(cf_correct)

        details.append({
            "base_id": base_id,
            "gold_label": gold,
            "direction": row["direction"],
            "original_name": row.get("original_name"),
            "counterfactual_name": row.get("counterfactual_name"),
            "original_prediction": orig["label"],
            "counterfactual_prediction": cf["label"],
            "label_flip": int(flip),
            "original_correct": int(orig_correct),
            "counterfactual_correct": int(cf_correct),
            "harmful_flip": int(harmful),
            "beneficial_flip": int(beneficial),
        })

    by_direction = {}
    for direction in (
        "feminine_to_masculine",
        "masculine_to_feminine",
    ):
        subset = [x for x in details if x["direction"] == direction]
        by_direction[direction] = {
            "n": len(subset),
            "flip_rate": statistics.mean(x["label_flip"] for x in subset),
            "harmful_flip_rate": statistics.mean(
                x["harmful_flip"] for x in subset
            ),
            "beneficial_flip_rate": statistics.mean(
                x["beneficial_flip"] for x in subset
            ),
            "original_accuracy": statistics.mean(
                x["original_correct"] for x in subset
            ),
            "counterfactual_accuracy": statistics.mean(
                x["counterfactual_correct"] for x in subset
            ),
        }

    by_label = {}
    for label in LABELS:
        subset = [x for x in details if x["gold_label"] == label]
        by_label[label] = {
            "n": len(subset),
            "flip_rate": statistics.mean(x["label_flip"] for x in subset),
            "harmful_flip_rate": statistics.mean(
                x["harmful_flip"] for x in subset
            ),
            "original_accuracy": statistics.mean(
                x["original_correct"] for x in subset
            ),
            "counterfactual_accuracy": statistics.mean(
                x["counterfactual_correct"] for x in subset
            ),
        }

    harmful_count = sum(x["harmful_flip"] for x in details)
    beneficial_count = sum(x["beneficial_flip"] for x in details)

    summary = {
        "pairs": len(gender_rows),
        "original_accuracy":
            bootstrap_case_mean(original_correct_by_case, AUDIT_SEED + 11),
        "counterfactual_accuracy":
            bootstrap_case_mean(cf_correct_by_case, AUDIT_SEED + 12),
        "flip_rate":
            bootstrap_case_mean(flip_by_case, AUDIT_SEED + 13),
        "harmful_flip_rate":
            bootstrap_case_mean(harm_by_case, AUDIT_SEED + 14),
        "beneficial_flip_rate":
            bootstrap_case_mean(benefit_by_case, AUDIT_SEED + 15),
        "correctness_discordance": {
            "harmful_count": harmful_count,
            "beneficial_count": beneficial_count,
            "exact_mcnemar_p_value":
                exact_mcnemar_p(harmful_count, beneficial_count),
        },
        "by_direction": by_direction,
        "by_label": by_label,
        "probability_metrics_available": False,
    }
    return summary, details


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    project_root = find_project_root()
    official_test_path = project_root / OFFICIAL_TEST
    gender_path = project_root / GENDER_CHALLENGE
    name_base_path, name_challenge_path = find_name_audit_files(project_root)

    output_dir = project_root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / PREDICTIONS_CACHE
    api_log_path = output_dir / API_LOG
    failures_path = output_dir / FAILURES_LOG
    manifest_path = output_dir / MANIFEST

    print("=" * 80)
    print("FLEXID — DEEPSEEK: NAME + GENDER SENSITIVITY")
    print("=" * 80)
    print(f"Projet          : {project_root}")
    print(f"Name base       : {name_base_path}")
    print(f"Name challenge  : {name_challenge_path}")
    print(f"Gender challenge: {gender_path}")
    print(f"Sortie          : {output_dir}")
    print(f"Modèle          : {MODEL}")

    official_rows = read_jsonl(official_test_path)
    official = validate_official_test(official_rows)

    name_bases = read_jsonl(name_base_path)
    name_variants = read_jsonl(name_challenge_path)
    name_bases, name_variants = validate_name_challenge(
        name_bases, name_variants, official
    )

    gender_rows = read_jsonl(gender_path)
    gender_rows = validate_gender_challenge(gender_rows, official)

    manifest = build_manifest(
        name_base_path,
        name_challenge_path,
        gender_path,
    )
    verify_manifest(manifest_path, manifest)
    (output_dir / SYSTEM_PROMPT_FILE).write_text(
        SYSTEM_PROMPT, encoding="utf-8"
    )

    items_by_hash, maps = collect_unique_items(
        name_bases,
        name_variants,
        gender_rows,
    )

    print("\nValidation")
    print(f"  Test officiel       : {len(official_rows)}")
    print(f"  Name base           : {len(name_bases)}")
    print(f"  Name variants       : {len(name_variants)}")
    print(f"  Gender pairs        : {len(gender_rows)}")
    print(f"  Requêtes uniques    : {len(items_by_hash)}")
    print("  Gold envoyé modèle  : NON")

    cache = load_cache(cache_path)
    compatible_cached = {
        key: row
        for key, row in cache.items()
        if key in items_by_hash
    }
    require(
        len(compatible_cached) == len(cache),
        "Le cache contient des prédictions étrangères au manifeste actuel. "
        "Renomme le dossier de sortie avant de relancer."
    )

    pending = [
        item
        for key, item in items_by_hash.items()
        if key not in cache
    ]

    print(f"  Déjà en cache       : {len(cache)}")
    print(f"  À interroger        : {len(pending)}")

    if pending:
        api_key = get_api_key(project_root)
        OpenAI, APIConnectionError, APIStatusError, RateLimitError = import_openai()

        client = OpenAI(
            api_key=api_key,
            base_url=BASE_URL,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        api_exception_types = (
            APIConnectionError,
            APIStatusError,
            RateLimitError,
        )

        consecutive_failures = 0

        for index, item in enumerate(pending, 1):
            print(
                f"[{index:>3}/{len(pending)}] "
                f"{item['id']} ... ",
                end="",
                flush=True,
            )

            prediction = call_deepseek(
                client,
                item,
                api_log_path,
                failures_path,
                api_exception_types,
            )

            if prediction is None:
                consecutive_failures += 1
                print("ÉCHEC")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"{MAX_CONSECUTIVE_FAILURES} échecs consécutifs. "
                        "Arrêt de sécurité; relance ensuite pour reprendre."
                    )
            else:
                consecutive_failures = 0
                append_jsonl(cache_path, prediction)
                cache[item["content_sha256"]] = prediction
                print(prediction["label"])

            time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)

    missing = sorted(set(items_by_hash) - set(cache))
    require(
        not missing,
        f"{len(missing)} prédictions manquent encore. "
        "Relance le script: il reprendra automatiquement."
    )

    # Calcul des deux audits.
    name_summary, name_details = evaluate_name_sensitivity(
        name_bases,
        name_variants,
        maps["mapping"],
        cache,
    )
    gender_summary, gender_details = evaluate_gender_sensitivity(
        gender_rows,
        maps["mapping"],
        cache,
    )

    final = {
        "audit_version": AUDIT_VERSION,
        "completed_at_utc": utc_now(),
        "model": {
            "id": MODEL,
            "base_url": BASE_URL,
            "thinking": {"type": THINKING_TYPE},
            "reasoning_effort": REASONING_EFFORT,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "temperature_explicitly_set": TEMPERATURE_EXPLICITLY_SET,
            "prompt_sha256": sha256_text(SYSTEM_PROMPT),
            "one_current_prediction_per_unique_input": True,
        },
        "scope": {
            "name_identity_sensitivity":
                "same grammatical name category; legal content held constant",
            "gender_counterfactual_sensitivity":
                "manually validated grammatical-gender counterfactual pairs",
            "not_a_claim":
                "comprehensive demographic fairness",
            "probability_metrics":
                "not reported because this protocol does not obtain calibrated "
                "class probabilities from the API",
        },
        "requests": {
            "unique_inputs": len(items_by_hash),
            "cache_entries": len(cache),
        },
        "name_identity_sensitivity": name_summary,
        "gender_counterfactual_sensitivity": gender_summary,
    }

    write_json(output_dir / SUMMARY, final)
    write_csv(output_dir / NAME_RESULTS_CSV, name_details)
    write_csv(output_dir / GENDER_RESULTS_CSV, gender_details)

    print("\n" + "=" * 80)
    print("RÉSULTATS DEEPSEEK — NAME IDENTITY")
    print("=" * 80)
    print(f"Original accuracy : {name_summary['original_accuracy']:.4f}")
    print(f"Variant accuracy  : {name_summary['variant_accuracy']:.4f}")
    print(
        f"Flip rate         : {name_summary['flip_rate']['mean']:.4f} "
        f"[{name_summary['flip_rate']['ci95'][0]:.4f}, "
        f"{name_summary['flip_rate']['ci95'][1]:.4f}]"
    )
    print(
        "Consistency       : "
        f"{name_summary['all_variants_consistent_with_original_rate']['mean']:.4f}"
    )
    print(
        "Harmful flip rate : "
        f"{name_summary['harmful_flip_rate']['mean']:.4f}"
    )
    print(
        "Beneficial rate   : "
        f"{name_summary['beneficial_flip_rate']['mean']:.4f}"
    )

    print("\n" + "=" * 80)
    print("RÉSULTATS DEEPSEEK — GENDER COUNTERFACTUAL")
    print("=" * 80)
    print(
        "Original accuracy       : "
        f"{gender_summary['original_accuracy']['mean']:.4f}"
    )
    print(
        "Counterfactual accuracy : "
        f"{gender_summary['counterfactual_accuracy']['mean']:.4f}"
    )
    print(
        f"Flip rate               : {gender_summary['flip_rate']['mean']:.4f} "
        f"[{gender_summary['flip_rate']['ci95'][0]:.4f}, "
        f"{gender_summary['flip_rate']['ci95'][1]:.4f}]"
    )
    print(
        "Harmful flip rate       : "
        f"{gender_summary['harmful_flip_rate']['mean']:.4f}"
    )
    print(
        "Beneficial flip rate    : "
        f"{gender_summary['beneficial_flip_rate']['mean']:.4f}"
    )
    print(
        "F→M flip rate           : "
        f"{gender_summary['by_direction']['feminine_to_masculine']['flip_rate']:.4f}"
    )
    print(
        "M→F flip rate           : "
        f"{gender_summary['by_direction']['masculine_to_feminine']['flip_rate']:.4f}"
    )
    print(
        "McNemar exact p         : "
        f"{gender_summary['correctness_discordance']['exact_mcnemar_p_value']:.6f}"
    )

    print("\nFichiers principaux")
    print(f"  {output_dir / SUMMARY}")
    print(f"  {output_dir / NAME_RESULTS_CSV}")
    print(f"  {output_dir / GENDER_RESULTS_CSV}")
    print(f"  {cache_path}")
    print("\nAudit DeepSeek terminé.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[FLEXID DEEPSEEK AUDIT ERROR] {exc}", file=sys.stderr)
        raise
