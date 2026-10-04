#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENT_VERSION = "FLEXID-DEEPSEEK-FEWSHOT-THINKING-180-v1.0"

INPUT_FILENAME = "flexid_kappa_rationale_180_unique_law_refs_60_each.jsonl"
EXPECTED_INSTANCE_COUNT = 180

SPLIT_RELATIVE_DIR = Path("data") / "flexid_exact_group_split"
EXPECTED_SPLIT_SIZES = {
    "train": 701,
    "validation": 151,
    "test": 150,
}

MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"

THINKING_TYPE = "enabled"
REASONING_EFFORT = "high"
MAX_OUTPUT_TOKENS = 8192

# Few-shot ablation
FEW_SHOT_PER_LABEL = 2
FEW_SHOT_TOTAL = 6
FEW_SHOT_SELECTION_SEED = 2026
MAX_DEMO_PREMISE_TOKENS = 160

MAX_ATTEMPTS_PER_INSTANCE = 4
REQUEST_TIMEOUT_SECONDS = 180.0
MAX_CONSECUTIVE_FAILURES = 5
DELAY_BETWEEN_REQUESTS_SECONDS = 0.15

ALLOWED_LABELS = ("entailment", "contradiction", "neutral")
NON_NEUTRAL_LABELS = {"entailment", "contradiction"}

# Facultatif: mets temporairement la clé ici si tu ne veux pas utiliser
# la variable d'environnement. Ne committe jamais une vraie clé.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()

# Sorties FEW-SHOT distinctes des sorties ZERO-SHOT.
PREDICTIONS_FILENAME = "flexid_deepseek_v4_flash_fewshot_thinking_predictions.jsonl"
API_LOG_FILENAME = "flexid_deepseek_v4_flash_fewshot_thinking_api_log.jsonl"
FAILURES_FILENAME = "flexid_deepseek_v4_flash_fewshot_thinking_failures.jsonl"
MANIFEST_FILENAME = "flexid_deepseek_v4_flash_fewshot_thinking_manifest.json"
PROMPT_FILENAME = "flexid_deepseek_v4_flash_fewshot_thinking_prompt.txt"
DEMOS_FILENAME = "flexid_deepseek_v4_flash_fewshot_demos.json"
EVALUATION_FILENAME = "flexid_deepseek_v4_flash_fewshot_thinking_evaluation.json"
COMPARISON_FILENAME = "flexid_deepseek_v4_flash_fewshot_vs_zeroshot.json"

# Sorties zero-shot déjà existantes, si présentes.
ZERO_SHOT_PREDICTIONS_FILENAME = "flexid_deepseek_v4_flash_predictions.jsonl"
ZERO_SHOT_API_LOG_FILENAME = "flexid_deepseek_v4_flash_api_log.jsonl"

BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 2027

TOKEN_LINE_PATTERN = re.compile(r"(?m)^\[(\d+)\]\s+(\S+)")
WHITESPACE_TOKEN_PATTERN = re.compile(r"\S+")

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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_law_ref(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def get_law_ref(row: dict[str, Any]) -> str:
    root = row.get("law_ref")
    meta = row.get("meta")
    nested = meta.get("law_ref") if isinstance(meta, dict) else None

    if root is not None and nested is not None:
        require(
            normalize_law_ref(root) == normalize_law_ref(nested),
            f"{row.get('id')}: law_ref racine et meta.law_ref divergent."
        )
    value = root if root is not None else nested
    require(value is not None, f"{row.get('id')}: law_ref absente.")
    return str(value)


def find_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    current_dir = Path.cwd().resolve()
    candidates = []

    for start in (script_dir, current_dir):
        candidates.append(start)
        candidates.extend(start.parents)

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        input_file = candidate / "data" / INPUT_FILENAME
        split_dir = candidate / SPLIT_RELATIVE_DIR

        if (
            input_file.is_file()
            and all((split_dir / f"{name}.jsonl").is_file()
                    for name in EXPECTED_SPLIT_SIZES)
        ):
            return candidate

    raise FileNotFoundError(
        "Racine FLEXID introuvable. Il faut:\n"
        f"  data/{INPUT_FILENAME}\n"
        "  data/flexid_exact_group_split/train.jsonl\n"
        "  data/flexid_exact_group_split/validation.jsonl\n"
        "  data/flexid_exact_group_split/test.jsonl"
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"Fichier introuvable: {path}")
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}, ligne {line_no}: JSON invalide: {exc}"
                ) from exc
            require(isinstance(row, dict),
                    f"{path}, ligne {line_no}: objet JSON attendu.")
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    tmp.replace(path)


def get_api_key(project_root: Path) -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key

    key = DEEPSEEK_API_KEY.strip()
    if key:
        return key

    optional_file = project_root / ".deepseek_api_key"
    if optional_file.is_file():
        key = optional_file.read_text(encoding="utf-8").strip()
        if key:
            return key

    raise RuntimeError(
        "Clé DeepSeek absente. Définis DEEPSEEK_API_KEY dans l'environnement "
        "ou renseigne temporairement la constante DEEPSEEK_API_KEY en haut "
        "du script."
    )


# ============================================================================
# TOKENISATION / GOLD RATIONALE
# ============================================================================

def whitespace_tokens_with_offsets(text: str):
    return list(WHITESPACE_TOKEN_PATTERN.finditer(text))


def tokenize_premise(premise: str) -> tuple[str, int]:
    matches = whitespace_tokens_with_offsets(premise)
    require(matches, "Prémisse vide après tokenisation.")
    rendered = "\n".join(
        f"[{index}] {match.group(0)}"
        for index, match in enumerate(matches, 1)
    )
    return rendered, len(matches)


def map_char_span_to_token_span(
    premise: str,
    start_char: int,
    end_char: int,
) -> tuple[int, int]:
    require(0 <= start_char < end_char <= len(premise),
            f"Span caractères invalide [{start_char}, {end_char}) "
            f"pour longueur {len(premise)}.")

    matches = whitespace_tokens_with_offsets(premise)
    covered = [
        index
        for index, match in enumerate(matches, 1)
        if match.start() < end_char and match.end() > start_char
    ]
    require(covered, "Le rationale caractère ne couvre aucun token.")
    return covered[0], covered[-1]


def gold_token_span(row: dict[str, Any]) -> tuple[int | None, int | None]:
    label = row["label"]
    if label == "neutral":
        return None, None

    premise = row["premise"]

    # Schéma canonique FLEXID: rationale_start/end = offsets caractères,
    # rationale_text = substring correspondante.
    start = row.get("rationale_start")
    end = row.get("rationale_end")
    rationale_text = row.get("rationale_text")

    if isinstance(start, int) and isinstance(end, int) and end > start:
        if isinstance(rationale_text, str) and rationale_text:
            extracted = premise[start:end]
            require(
                extracted == rationale_text,
                f"{row['id']}: rationale_text ne correspond pas exactement "
                "à premise[rationale_start:rationale_end]."
            )
        return map_char_span_to_token_span(premise, start, end)

    # Fallback éventuel si un split contient déjà des bornes token.
    token_start = row.get("rationale_start_token")
    token_end = row.get("rationale_end_token")
    if isinstance(token_start, int) and isinstance(token_end, int):
        require(1 <= token_start <= token_end,
                f"{row['id']}: bornes token gold invalides.")
        return token_start, token_end

    # Dernier fallback: retrouver rationale_text exactement.
    if isinstance(rationale_text, str) and rationale_text:
        start = premise.find(rationale_text)
        require(start >= 0, f"{row['id']}: rationale_text introuvable.")
        end = start + len(rationale_text)
        return map_char_span_to_token_span(premise, start, end)

    raise RuntimeError(
        f"{row['id']}: impossible de construire le span rationale gold."
    )


def render_hypothesis(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2)


# ============================================================================
# VALIDATION 180 + SPLITS
# ============================================================================

def prepare_target_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    require(
        len(rows) == EXPECTED_INSTANCE_COUNT,
        f"Le fichier 180 doit contenir {EXPECTED_INSTANCE_COUNT} instances; "
        f"trouvé {len(rows)}."
    )

    prepared = []
    seen = set()

    for position, row in enumerate(rows, 1):
        leaked = sorted(FORBIDDEN_INPUT_FIELDS.intersection(row))
        require(
            not leaked,
            f"Instance #{position}: champs gold interdits dans l'entrée modèle: "
            + ", ".join(leaked)
        )

        rid = str(row.get("id", "")).strip()
        require(rid, f"Instance #{position}: id absent.")
        require(rid not in seen, f"ID cible dupliqué: {rid}.")
        seen.add(rid)

        tokenized = row.get("tokenized_premise")
        require(
            isinstance(tokenized, str) and tokenized.strip(),
            f"{rid}: tokenized_premise absente."
        )

        token_matches = list(TOKEN_LINE_PATTERN.finditer(tokenized))
        require(token_matches, f"{rid}: aucun token numéroté.")
        ids = [int(m.group(1)) for m in token_matches]
        require(
            ids == list(range(1, len(ids) + 1)),
            f"{rid}: numérotation tokens non consécutive."
        )

        hypothesis = render_hypothesis(row.get("hypothesis_facts"))
        require(hypothesis, f"{rid}: hypothesis_facts vide.")

        prepared.append({
            "id": rid,
            "tokenized_premise": tokenized.strip(),
            "hypothesis_facts": hypothesis,
            "max_token_id": ids[-1],
        })

    return prepared


def load_splits(project_root: Path) -> dict[str, list[dict[str, Any]]]:
    split_dir = project_root / SPLIT_RELATIVE_DIR
    splits = {}

    all_ids = set()
    for name, expected_n in EXPECTED_SPLIT_SIZES.items():
        rows = load_jsonl(split_dir / f"{name}.jsonl")
        require(
            len(rows) == expected_n,
            f"{name}: {len(rows)} != {expected_n}."
        )
        for row in rows:
            require(row.get("label") in ALLOWED_LABELS,
                    f"{row.get('id')}: label split invalide.")
            rid = row.get("id")
            require(isinstance(rid, str) and rid, f"{name}: id invalide.")
            require(rid not in all_ids, f"ID présent dans plusieurs splits: {rid}.")
            all_ids.add(rid)
        splits[name] = rows

    require(len(all_ids) == sum(EXPECTED_SPLIT_SIZES.values()),
            "Nombre total d'IDs split inattendu.")
    return splits


def make_gold_index(splits: dict[str, list[dict[str, Any]]]) -> dict[str, dict]:
    result = {}
    for split_name, rows in splits.items():
        for row in rows:
            copy = dict(row)
            copy["_split"] = split_name
            result[row["id"]] = copy
    return result


def validate_targets_against_gold(
    targets: list[dict[str, Any]],
    gold_by_id: dict[str, dict],
) -> None:
    missing = [row["id"] for row in targets if row["id"] not in gold_by_id]
    require(
        not missing,
        "Certaines cibles 180 ne sont pas dans le split officiel v3: "
        + ", ".join(missing[:20])
    )

    for target in targets:
        gold = gold_by_id[target["id"]]
        tokenized_gold, max_gold = tokenize_premise(gold["premise"])
        require(
            target["max_token_id"] == max_gold,
            f"{target['id']}: nombre de tokens target != gold."
        )
        require(
            target["tokenized_premise"] == tokenized_gold,
            f"{target['id']}: tokenized_premise cible != tokenisation du gold."
        )
        require(
            target["hypothesis_facts"] == render_hypothesis(
                gold["hypothesis_facts"]
            ),
            f"{target['id']}: hypothesis target != gold."
        )


# ============================================================================
# SÉLECTION DES 6 DÉMONSTRATIONS
# ============================================================================

def demo_sort_key(row: dict[str, Any]) -> str:
    return sha256_text(f"{FEW_SHOT_SELECTION_SEED}|{row['id']}")


def select_few_shot_demos(
    train_rows: list[dict[str, Any]],
    target_ids: set[str],
    target_law_refs: set[str],
) -> list[dict[str, Any]]:
    candidates_by_label = {label: [] for label in ALLOWED_LABELS}

    for row in train_rows:
        rid = row["id"]
        if rid in target_ids:
            continue

        law_ref_norm = normalize_law_ref(get_law_ref(row))
        if law_ref_norm in target_law_refs:
            # Evite une démonstration issue de la même référence juridique
            # qu'une des 180 cibles.
            continue

        premise = row.get("premise")
        hypothesis = row.get("hypothesis_facts")
        if not isinstance(premise, str) or not premise.strip():
            continue
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            continue

        tokenized, max_token = tokenize_premise(premise)
        if max_token > MAX_DEMO_PREMISE_TOKENS:
            continue

        try:
            start, end = gold_token_span(row)
        except Exception:
            continue

        candidates_by_label[row["label"]].append({
            "id": rid,
            "label": row["label"],
            "premise": premise,
            "tokenized_premise": tokenized,
            "hypothesis_facts": hypothesis,
            "max_token_id": max_token,
            "rationale_start_token": start,
            "rationale_end_token": end,
            "law_ref": get_law_ref(row),
        })

    selected = []
    used_refs = set()

    # Alternance des labels pour éviter de regrouper les classes dans le prompt.
    for round_index in range(FEW_SHOT_PER_LABEL):
        for label in ALLOWED_LABELS:
            candidates = sorted(
                candidates_by_label[label],
                key=demo_sort_key,
            )

            choice = None
            for candidate in candidates:
                ref = normalize_law_ref(candidate["law_ref"])
                if ref in used_refs:
                    continue
                if candidate["id"] in {x["id"] for x in selected}:
                    continue
                choice = candidate
                break

            require(
                choice is not None,
                f"Pas assez de démonstrations valides pour {label}."
            )
            selected.append(choice)
            used_refs.add(normalize_law_ref(choice["law_ref"]))

    require(len(selected) == FEW_SHOT_TOTAL,
            f"Few-shot: {len(selected)} != {FEW_SHOT_TOTAL}.")
    counts = Counter(row["label"] for row in selected)
    require(
        all(counts[label] == FEW_SHOT_PER_LABEL for label in ALLOWED_LABELS),
        f"Répartition few-shot incorrecte: {dict(counts)}."
    )
    require(
        len({normalize_law_ref(row["law_ref"]) for row in selected})
        == FEW_SHOT_TOTAL,
        "Les démonstrations n'ont pas six law_ref distinctes."
    )

    return selected


def build_instance_prompt(
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


def demo_answer(demo: dict[str, Any]) -> str:
    return json.dumps(
        {
            "id": demo["id"],
            "label": demo["label"],
            "rationale_start_token": demo["rationale_start_token"],
            "rationale_end_token": demo["rationale_end_token"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_messages(
    target: dict[str, Any],
    demos: list[dict[str, Any]],
    validation_error: str | None = None,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for demo in demos:
        messages.append({
            "role": "user",
            "content": build_instance_prompt(demo),
        })
        messages.append({
            "role": "assistant",
            "content": demo_answer(demo),
        })

    messages.append({
        "role": "user",
        "content": build_instance_prompt(target, validation_error),
    })
    return messages


def save_prompt_artifact(path: Path, demos: list[dict[str, Any]]) -> None:
    parts = [
        "=== SYSTEM PROMPT ===",
        SYSTEM_PROMPT,
        "",
        "=== SIX FEW-SHOT DEMONSTRATIONS ===",
        "",
    ]
    for index, demo in enumerate(demos, 1):
        parts.extend([
            f"--- DEMO {index} / {demo['label']} / {demo['id']} ---",
            "USER:",
            build_instance_prompt(demo),
            "ASSISTANT:",
            demo_answer(demo),
            "",
        ])
    parts.append(
        "Pour chaque cible, le même bloc de 6 démonstrations précède "
        "le prompt de l'instance cible."
    )
    path.write_text("\n".join(parts), encoding="utf-8")


# ============================================================================
# VALIDATION DE SORTIE
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
        raise ValueError(f"{record_id}: {field_name} booléen interdit.")
    if isinstance(value, int):
        return value
    raise ValueError(f"{record_id}: {field_name} doit être int ou null.")


def validate_prediction(
    payload: Any,
    instance: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("La réponse JSON doit être un objet.")

    expected_keys = {
        "id",
        "label",
        "rationale_start_token",
        "rationale_end_token",
    }
    actual = set(payload)
    missing = expected_keys - actual
    extra = actual - expected_keys
    if missing:
        raise ValueError("Champs absents: " + ", ".join(sorted(missing)))
    if extra:
        raise ValueError("Champs supplémentaires: " + ", ".join(sorted(extra)))

    if payload["id"] != instance["id"]:
        raise ValueError(
            f"id incorrect: attendu {instance['id']!r}, "
            f"reçu {payload['id']!r}."
        )

    label = payload["label"]
    if not isinstance(label, str):
        raise ValueError("label doit être une chaîne.")
    label = label.strip().casefold()
    if label not in ALLOWED_LABELS:
        raise ValueError(f"label invalide: {label!r}.")

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
                "Pour neutral, les deux bornes rationale doivent être null."
            )
    else:
        if start is None or end is None:
            raise ValueError(
                f"Pour {label}, les deux bornes doivent être entières."
            )
        if not (1 <= start <= end <= instance["max_token_id"]):
            raise ValueError(
                f"Span invalide [{start}, {end}], max="
                f"{instance['max_token_id']}."
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
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Contenu final vide.")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON invalide: {exc}") from exc
    return validate_prediction(payload, instance)


def load_existing_predictions(
    path: Path,
    instances_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    result = {}
    for line_no, payload in enumerate(load_jsonl(path), 1):
        rid = str(payload.get("id", "")).strip()
        require(rid in instances_by_id,
                f"{path}, ligne {line_no}: id inconnu {rid}.")
        require(rid not in result, f"{path}: prédiction dupliquée {rid}.")
        result[rid] = validate_prediction(payload, instances_by_id[rid])
    return result


# ============================================================================
# MANIFESTE
# ============================================================================

def demos_fingerprint(demos: list[dict[str, Any]]) -> str:
    frozen = [
        {
            "id": d["id"],
            "label": d["label"],
            "law_ref": d["law_ref"],
            "rationale_start_token": d["rationale_start_token"],
            "rationale_end_token": d["rationale_end_token"],
        }
        for d in demos
    ]
    return sha256_text(
        json.dumps(frozen, ensure_ascii=False, sort_keys=True)
    )


def build_manifest(
    input_file: Path,
    demos: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_VERSION,
        "created_at_utc": utc_now_iso(),
        "input_file": str(input_file),
        "input_sha256": sha256_file(input_file),
        "expected_instances": EXPECTED_INSTANCE_COUNT,
        "model": MODEL,
        "base_url": BASE_URL,
        "thinking": {"type": THINKING_TYPE},
        "reasoning_effort": REASONING_EFFORT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "temperature_explicitly_set": False,
        "response_format": {"type": "json_object"},
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "few_shot": {
            "total": FEW_SHOT_TOTAL,
            "per_label": FEW_SHOT_PER_LABEL,
            "selection_seed": FEW_SHOT_SELECTION_SEED,
            "source_split": "train",
            "target_ids_excluded": True,
            "target_law_refs_excluded": True,
            "unique_demo_law_refs": True,
            "written_chain_of_thought_in_demos": False,
            "demo_ids": [d["id"] for d in demos],
            "demo_labels": [d["label"] for d in demos],
            "demo_law_refs": [d["law_ref"] for d in demos],
            "demo_fingerprint": demos_fingerprint(demos),
        },
        "one_request_per_target": True,
        "reasoning_content_saved": False,
    }


def verify_or_create_manifest(
    path: Path,
    expected: dict[str, Any],
) -> None:
    if not path.exists():
        write_json_atomic(path, expected)
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
        "temperature_explicitly_set",
        "response_format",
        "system_prompt_sha256",
        "few_shot",
        "one_request_per_target",
        "reasoning_content_saved",
    )

    differences = [
        key for key in stable_keys
        if existing.get(key) != expected.get(key)
    ]
    require(
        not differences,
        "Le manifeste few-shot existant est incompatible. Archive/supprime "
        "les anciennes sorties few-shot avant de relancer. Différences: "
        + ", ".join(differences)
    )


# ============================================================================
# API
# ============================================================================

def import_openai():
    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            OpenAI,
            RateLimitError,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Le paquet openai n'est pas installé. Exécute:\n"
            "  python -m pip install -U openai"
        ) from exc
    return OpenAI, APIConnectionError, APIStatusError, RateLimitError


def usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    try:
        return usage.model_dump()
    except Exception:
        return {"raw": str(usage)}


def status_code_from_exception(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def call_model_for_instance(
    *,
    client: Any,
    instance: dict[str, Any],
    demos: list[dict[str, Any]],
    api_log_file: Path,
    failure_file: Path,
    api_exception_types: tuple[type[BaseException], ...],
) -> dict[str, Any] | None:
    last_validation_error = None

    for attempt in range(1, MAX_ATTEMPTS_PER_INSTANCE + 1):
        messages = build_messages(
            instance,
            demos,
            validation_error=last_validation_error,
        )
        messages_sha = sha256_text(
            json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        )

        started = utc_now_iso()
        start_time = time.monotonic()

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
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

            elapsed = time.monotonic() - start_time
            require(response.choices, "Réponse API sans choice.")

            choice = response.choices[0]
            content = choice.message.content

            try:
                prediction = parse_and_validate_response(content, instance)
            except ValueError as exc:
                last_validation_error = str(exc)
                append_jsonl(api_log_file, {
                    "timestamp_utc": started,
                    "id": instance["id"],
                    "attempt": attempt,
                    "status": "invalid_model_output",
                    "validation_error": last_validation_error,
                    "raw_final_content":
                        content[:2000] if isinstance(content, str) else None,
                    "model_requested": MODEL,
                    "model_returned": getattr(response, "model", None),
                    "response_id": getattr(response, "id", None),
                    "system_fingerprint":
                        getattr(response, "system_fingerprint", None),
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "usage": usage_to_dict(getattr(response, "usage", None)),
                    "elapsed_seconds": round(elapsed, 3),
                    "messages_sha256": messages_sha,
                    "reasoning_content_saved": False,
                })
                if attempt < MAX_ATTEMPTS_PER_INSTANCE:
                    time.sleep(min(8.0, 1.5 * attempt))
                    continue
                break

            append_jsonl(api_log_file, {
                "timestamp_utc": started,
                "id": instance["id"],
                "attempt": attempt,
                "status": "success",
                "model_requested": MODEL,
                "model_returned": getattr(response, "model", None),
                "response_id": getattr(response, "id", None),
                "system_fingerprint":
                    getattr(response, "system_fingerprint", None),
                "finish_reason": getattr(choice, "finish_reason", None),
                "usage": usage_to_dict(getattr(response, "usage", None)),
                "elapsed_seconds": round(elapsed, 3),
                "messages_sha256": messages_sha,
                "reasoning_content_saved": False,
            })
            return prediction

        except api_exception_types as exc:
            elapsed = time.monotonic() - start_time
            status = status_code_from_exception(exc)

            append_jsonl(api_log_file, {
                "timestamp_utc": started,
                "id": instance["id"],
                "attempt": attempt,
                "status": "api_error",
                "exception_type": type(exc).__name__,
                "status_code": status,
                "error": str(exc)[:2000],
                "elapsed_seconds": round(elapsed, 3),
                "messages_sha256": messages_sha,
            })

            if status in {401, 403}:
                raise RuntimeError(
                    "Authentification DeepSeek refusée."
                ) from exc

            if attempt < MAX_ATTEMPTS_PER_INSTANCE:
                delay = min(60.0, 2.0 ** (attempt - 1))
                delay += random.uniform(0.0, 0.5)
                time.sleep(delay)
                continue

            last_validation_error = (
                f"Échec API après {MAX_ATTEMPTS_PER_INSTANCE} tentatives: "
                f"{type(exc).__name__}: {str(exc)[:500]}"
            )
            break

        except Exception as exc:
            elapsed = time.monotonic() - start_time
            append_jsonl(api_log_file, {
                "timestamp_utc": started,
                "id": instance["id"],
                "attempt": attempt,
                "status": "unexpected_error",
                "exception_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "elapsed_seconds": round(elapsed, 3),
                "messages_sha256": messages_sha,
            })
            last_validation_error = (
                f"{type(exc).__name__}: {str(exc)[:500]}"
            )
            break

    append_jsonl(failure_file, {
        "timestamp_utc": utc_now_iso(),
        "id": instance["id"],
        "attempts": MAX_ATTEMPTS_PER_INSTANCE,
        "last_error": last_validation_error,
    })
    return None


# ============================================================================
# ÉVALUATION
# ============================================================================

def span_set(start: int | None, end: int | None) -> set[int]:
    if start is None or end is None:
        return set()
    return set(range(start, end + 1))


def span_f1(gold: set[int], pred: set[int]) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    inter = len(gold & pred)
    precision = inter / len(pred)
    recall = inter / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def span_iou(gold: set[int], pred: set[int]) -> float:
    union = gold | pred
    if not union:
        return 1.0
    return len(gold & pred) / len(union)


def macro_f1_labels(
    gold_labels: list[str],
    pred_labels: list[str],
) -> tuple[float, dict[str, dict[str, float]]]:
    class_metrics = {}
    f1s = []

    for label in ALLOWED_LABELS:
        tp = sum(g == label and p == label
                 for g, p in zip(gold_labels, pred_labels))
        fp = sum(g != label and p == label
                 for g, p in zip(gold_labels, pred_labels))
        fn = sum(g == label and p != label
                 for g, p in zip(gold_labels, pred_labels))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(g == label for g in gold_labels),
        }
        f1s.append(f1)

    return statistics.mean(f1s), class_metrics


def cohen_kappa(gold_labels: list[str], pred_labels: list[str]) -> float:
    n = len(gold_labels)
    po = sum(g == p for g, p in zip(gold_labels, pred_labels)) / n

    gold_counts = Counter(gold_labels)
    pred_counts = Counter(pred_labels)
    pe = sum(
        (gold_counts[label] / n) * (pred_counts[label] / n)
        for label in ALLOWED_LABELS
    )
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def bootstrap_mean(values: list[float]) -> tuple[float, float]:
    require(values, "Bootstrap demandé sur liste vide.")
    rng = random.Random(BOOTSTRAP_SEED)
    reps = []

    for _ in range(BOOTSTRAP_REPS):
        sample = [rng.choice(values) for _ in values]
        reps.append(statistics.mean(sample))

    reps.sort()

    def pct(q: float) -> float:
        pos = q * (len(reps) - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return reps[lo]
        frac = pos - lo
        return reps[lo] * (1 - frac) + reps[hi] * frac

    return pct(0.025), pct(0.975)


def evaluate_predictions(
    predictions: dict[str, dict[str, Any]],
    target_order: list[str],
    gold_by_id: dict[str, dict],
) -> dict[str, Any]:
    require(
        set(predictions) == set(target_order),
        "Évaluation: prédictions incomplètes ou IDs étrangers."
    )

    gold_labels = []
    pred_labels = []
    confusion = {
        gold: {pred: 0 for pred in ALLOWED_LABELS}
        for gold in ALLOWED_LABELS
    }

    comparable_span_f1 = []
    comparable_iou = []
    comparable_em = []
    joint = []
    neutral_agreements = []

    per_instance = []

    for rid in target_order:
        gold = gold_by_id[rid]
        pred = predictions[rid]

        gold_label = gold["label"]
        pred_label = pred["label"]

        gold_labels.append(gold_label)
        pred_labels.append(pred_label)
        confusion[gold_label][pred_label] += 1

        gold_start, gold_end = gold_token_span(gold)
        pred_start = pred["rationale_start_token"]
        pred_end = pred["rationale_end_token"]

        gset = span_set(gold_start, gold_end)
        pset = span_set(pred_start, pred_end)

        instance_span_f1 = None
        instance_iou = None
        instance_em = None

        if gold_label in NON_NEUTRAL_LABELS and pred_label == gold_label:
            instance_span_f1 = span_f1(gset, pset)
            instance_iou = span_iou(gset, pset)
            instance_em = float(gset == pset)

            comparable_span_f1.append(instance_span_f1)
            comparable_iou.append(instance_iou)
            comparable_em.append(instance_em)

        if gold_label == "neutral":
            neutral_ok = float(
                pred_label == "neutral"
                and pred_start is None
                and pred_end is None
            )
            neutral_agreements.append(neutral_ok)
            joint_ok = neutral_ok
        else:
            if pred_label == gold_label:
                iou = span_iou(gset, pset)
                joint_ok = float(iou >= 0.50)
            else:
                joint_ok = 0.0

        joint.append(joint_ok)

        per_instance.append({
            "id": rid,
            "gold_label": gold_label,
            "pred_label": pred_label,
            "correct_label": int(gold_label == pred_label),
            "gold_rationale_start_token": gold_start,
            "gold_rationale_end_token": gold_end,
            "pred_rationale_start_token": pred_start,
            "pred_rationale_end_token": pred_end,
            "conditional_span_f1": instance_span_f1,
            "conditional_iou": instance_iou,
            "conditional_exact_match": instance_em,
            "joint_iou_at_0_50": joint_ok,
        })

    accuracy = statistics.mean(
        g == p for g, p in zip(gold_labels, pred_labels)
    )
    macro_f1, class_metrics = macro_f1_labels(gold_labels, pred_labels)

    if comparable_span_f1:
        ci_low, ci_high = bootstrap_mean(comparable_span_f1)
        token_f1_mean = statistics.mean(comparable_span_f1)
        iou_mean = statistics.mean(comparable_iou)
        em_mean = statistics.mean(comparable_em)
    else:
        ci_low = ci_high = token_f1_mean = iou_mean = em_mean = 0.0

    return {
        "n": len(target_order),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "cohen_kappa": cohen_kappa(gold_labels, pred_labels),
        "class_metrics": class_metrics,
        "confusion_matrix": confusion,
        "rationale": {
            "conditional_non_neutral_subset_n": len(comparable_span_f1),
            "macro_token_f1": token_f1_mean,
            "macro_token_f1_ci95": [ci_low, ci_high],
            "macro_iou": iou_mean,
            "exact_match": em_mean,
            "neutral_rationale_agreement":
                statistics.mean(neutral_agreements)
                if neutral_agreements else 0.0,
            "joint_iou_at_0_50": statistics.mean(joint),
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "per_instance": per_instance,
    }


# ============================================================================
# COMPARAISON ZERO-SHOT / FEW-SHOT
# ============================================================================

def successful_backend_metadata(log_path: Path) -> dict[str, Any] | None:
    if not log_path.is_file():
        return None

    rows = load_jsonl(log_path)
    successful = [row for row in rows if row.get("status") == "success"]
    if not successful:
        return None

    models = sorted({
        str(row.get("model_returned"))
        for row in successful
        if row.get("model_returned") is not None
    })
    fingerprints = sorted({
        str(row.get("system_fingerprint"))
        for row in successful
        if row.get("system_fingerprint") is not None
    })
    dates = [
        row.get("timestamp_utc")
        for row in successful
        if row.get("timestamp_utc")
    ]

    return {
        "successful_calls": len(successful),
        "model_returned_values": models,
        "system_fingerprints": fingerprints,
        "first_timestamp_utc": min(dates) if dates else None,
        "last_timestamp_utc": max(dates) if dates else None,
    }


def compare_conditions(
    zero_eval: dict[str, Any],
    few_eval: dict[str, Any],
    zero_backend: dict[str, Any] | None,
    few_backend: dict[str, Any] | None,
) -> dict[str, Any]:
    zero_per = {row["id"]: row for row in zero_eval["per_instance"]}
    few_per = {row["id"]: row for row in few_eval["per_instance"]}
    ids = sorted(set(zero_per) & set(few_per))

    label_changed = sum(
        zero_per[rid]["pred_label"] != few_per[rid]["pred_label"]
        for rid in ids
    )

    backend_model_match = None
    fingerprint_match = None

    if zero_backend is not None and few_backend is not None:
        backend_model_match = (
            zero_backend["model_returned_values"]
            == few_backend["model_returned_values"]
        )

        # Si les deux côtés ont un fingerprint observable, on le compare.
        if (
            zero_backend["system_fingerprints"]
            and few_backend["system_fingerprints"]
        ):
            fingerprint_match = (
                zero_backend["system_fingerprints"]
                == few_backend["system_fingerprints"]
            )

    strict_ablation = (
        backend_model_match is True
        and (fingerprint_match is True or fingerprint_match is None)
    )

    return {
        "same_180_target_ids": len(ids) == EXPECTED_INSTANCE_COUNT,
        "n_compared": len(ids),
        "zero_shot": {
            "accuracy": zero_eval["accuracy"],
            "macro_f1": zero_eval["macro_f1"],
            "rationale": zero_eval["rationale"],
        },
        "few_shot_thinking": {
            "accuracy": few_eval["accuracy"],
            "macro_f1": few_eval["macro_f1"],
            "rationale": few_eval["rationale"],
        },
        "delta_few_minus_zero": {
            "accuracy": few_eval["accuracy"] - zero_eval["accuracy"],
            "macro_f1": few_eval["macro_f1"] - zero_eval["macro_f1"],
            "macro_token_f1":
                few_eval["rationale"]["macro_token_f1"]
                - zero_eval["rationale"]["macro_token_f1"],
            "macro_iou":
                few_eval["rationale"]["macro_iou"]
                - zero_eval["rationale"]["macro_iou"],
            "joint_iou_at_0_50":
                few_eval["rationale"]["joint_iou_at_0_50"]
                - zero_eval["rationale"]["joint_iou_at_0_50"],
        },
        "prediction_label_changed_n": label_changed,
        "prediction_label_changed_rate":
            label_changed / len(ids) if ids else None,
        "backend_comparability": {
            "zero_shot": zero_backend,
            "few_shot": few_backend,
            "model_returned_match": backend_model_match,
            "system_fingerprint_match": fingerprint_match,
            "strict_prompt_ablation_supported": strict_ablation,
            "interpretation":
                "Si strict_prompt_ablation_supported=false, la différence "
                "peut aussi refléter un changement de backend/version entre "
                "les deux dates; ne pas l'attribuer uniquement au few-shot."
        },
    }


# ============================================================================
# MAIN
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
    demos_file = data_dir / DEMOS_FILENAME
    evaluation_file = data_dir / EVALUATION_FILENAME
    comparison_file = data_dir / COMPARISON_FILENAME

    raw_targets = load_jsonl(input_file)
    targets = prepare_target_records(raw_targets)
    targets_by_id = {row["id"]: row for row in targets}
    target_ids = set(targets_by_id)

    splits = load_splits(project_root)
    gold_by_id = make_gold_index(splits)
    validate_targets_against_gold(targets, gold_by_id)

    target_law_refs = {
        normalize_law_ref(get_law_ref(gold_by_id[rid]))
        for rid in target_ids
    }

    demos = select_few_shot_demos(
        splits["train"],
        target_ids,
        target_law_refs,
    )

    manifest = build_manifest(input_file, demos)
    verify_or_create_manifest(manifest_file, manifest)

    demo_export = {
        "selection_protocol": manifest["few_shot"],
        "demos": [
            {
                "id": d["id"],
                "label": d["label"],
                "law_ref": d["law_ref"],
                "premise": d["premise"],
                "hypothesis_facts": d["hypothesis_facts"],
                "rationale_start_token": d["rationale_start_token"],
                "rationale_end_token": d["rationale_end_token"],
            }
            for d in demos
        ],
    }

    if demos_file.exists():
        existing = json.loads(demos_file.read_text(encoding="utf-8"))
        require(
            existing == demo_export,
            "Le fichier demos existant diffère de la sélection déterministe."
        )
    else:
        write_json_atomic(demos_file, demo_export)

    if prompt_file.exists():
        # Réécriture autorisée seulement si le contenu est identique.
        old = prompt_file.read_text(encoding="utf-8")
        tmp = prompt_file.with_suffix(".check.tmp")
        save_prompt_artifact(tmp, demos)
        new = tmp.read_text(encoding="utf-8")
        tmp.unlink()
        require(old == new,
                "Le prompt few-shot existant diffère du prompt actuel.")
    else:
        save_prompt_artifact(prompt_file, demos)

    predictions = load_existing_predictions(
        predictions_file,
        targets_by_id,
    )

    api_key = get_api_key(project_root)
    OpenAI, APIConnectionError, APIStatusError, RateLimitError = import_openai()

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    pending = [
        row for row in targets
        if row["id"] not in predictions
    ]

    print("=" * 80)
    print("FLEXID — DEEPSEEK FEW-SHOT + THINKING — 180")
    print("=" * 80)
    print(f"Projet                 : {project_root}")
    print(f"Entrée                 : {input_file}")
    print(f"Modèle demandé         : {MODEL}")
    print(f"Thinking               : {THINKING_TYPE}")
    print(f"Reasoning effort       : {REASONING_EFFORT}")
    print(f"Démonstrations         : {len(demos)} "
          f"({FEW_SHOT_PER_LABEL} par label)")
    print("Source démonstrations  : TRAIN uniquement")
    print("Target IDs exclus      : OUI")
    print("Target law_refs exclues: OUI")
    print(f"Instances cibles       : {len(targets)}")
    print(f"Déjà validées          : {len(predictions)}")
    print(f"À traiter              : {len(pending)}")
    print("\nDémonstrations figées:")
    for d in demos:
        print(f"  {d['id']} | {d['label']:13s} | {d['law_ref']}")
    print("")

    api_exception_types = (
        APIConnectionError,
        APIStatusError,
        RateLimitError,
    )

    consecutive_failures = 0
    run_failures = 0

    for pending_index, instance in enumerate(pending, 1):
        global_position = next(
            i for i, candidate in enumerate(targets, 1)
            if candidate["id"] == instance["id"]
        )

        print(
            f"[{global_position:03d}/{len(targets)}] "
            f"{instance['id']}...",
            end=" ",
            flush=True,
        )

        prediction = call_model_for_instance(
            client=client,
            instance=instance,
            demos=demos,
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
                    "Arrêt de sécurité; relance ensuite pour reprendre."
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
        predictions[row["id"]]
        for row in targets
        if row["id"] in predictions
    ]
    write_jsonl_atomic(predictions_file, ordered_predictions)

    missing = [
        row["id"]
        for row in targets
        if row["id"] not in predictions
    ]

    print("")
    print(f"Prédictions validées : {len(ordered_predictions)}/{len(targets)}")
    print(f"Échecs ce run        : {run_failures}")

    if missing:
        print("Encore manquantes    : " + ", ".join(missing[:20]))
        print("Relance le même script: reprise automatique.")
        return 2

    target_order = [row["id"] for row in targets]
    few_eval = evaluate_predictions(
        predictions,
        target_order,
        gold_by_id,
    )
    write_json_atomic(evaluation_file, few_eval)

    print("\n" + "=" * 80)
    print("RÉSULTATS FEW-SHOT + THINKING")
    print("=" * 80)
    print(f"Accuracy            : {few_eval['accuracy']:.4f}")
    print(f"Macro-F1            : {few_eval['macro_f1']:.4f}")
    print(f"Cohen kappa         : {few_eval['cohen_kappa']:.4f}")
    print(
        "Rationale token F1  : "
        f"{few_eval['rationale']['macro_token_f1']:.4f} "
        f"[{few_eval['rationale']['macro_token_f1_ci95'][0]:.4f}, "
        f"{few_eval['rationale']['macro_token_f1_ci95'][1]:.4f}]"
    )
    print(
        f"Rationale IoU       : "
        f"{few_eval['rationale']['macro_iou']:.4f}"
    )
    print(
        f"Rationale EM        : "
        f"{few_eval['rationale']['exact_match']:.4f}"
    )
    print(
        f"NRA                 : "
        f"{few_eval['rationale']['neutral_rationale_agreement']:.4f}"
    )
    print(
        f"Joint IoU@0.50      : "
        f"{few_eval['rationale']['joint_iou_at_0_50']:.4f}"
    )

    # Comparaison automatique avec le zero-shot déjà exécuté.
    zero_predictions_path = data_dir / ZERO_SHOT_PREDICTIONS_FILENAME
    zero_log_path = data_dir / ZERO_SHOT_API_LOG_FILENAME

    if zero_predictions_path.is_file():
        zero_predictions = load_existing_predictions(
            zero_predictions_path,
            targets_by_id,
        )

        if len(zero_predictions) == EXPECTED_INSTANCE_COUNT:
            zero_eval = evaluate_predictions(
                zero_predictions,
                target_order,
                gold_by_id,
            )

            zero_backend = successful_backend_metadata(zero_log_path)
            few_backend = successful_backend_metadata(api_log_file)

            comparison = compare_conditions(
                zero_eval,
                few_eval,
                zero_backend,
                few_backend,
            )
            write_json_atomic(comparison_file, comparison)

            print("\n" + "=" * 80)
            print("ABLATION: ZERO-SHOT THINKING vs FEW-SHOT THINKING")
            print("=" * 80)
            print(
                f"Zero-shot Macro-F1   : "
                f"{zero_eval['macro_f1']:.4f}"
            )
            print(
                f"Few-shot Macro-F1    : "
                f"{few_eval['macro_f1']:.4f}"
            )
            print(
                f"Delta Macro-F1       : "
                f"{comparison['delta_few_minus_zero']['macro_f1']:+.4f}"
            )
            print(
                f"Zero-shot Joint      : "
                f"{zero_eval['rationale']['joint_iou_at_0_50']:.4f}"
            )
            print(
                f"Few-shot Joint       : "
                f"{few_eval['rationale']['joint_iou_at_0_50']:.4f}"
            )
            print(
                f"Delta Joint          : "
                f"{comparison['delta_few_minus_zero']['joint_iou_at_0_50']:+.4f}"
            )
            print(
                "Backend strictement comparable : "
                f"{comparison['backend_comparability']['strict_prompt_ablation_supported']}"
            )
        else:
            print(
                "\nZero-shot trouvé mais incomplet: comparaison automatique ignorée."
            )
    else:
        print(
            "\nAucun ancien fichier zero-shot trouvé; "
            "l'évaluation few-shot a néanmoins été produite."
        )

    print("\nFichiers:")
    print(f"  Demos       : {demos_file}")
    print(f"  Prompt      : {prompt_file}")
    print(f"  Prédictions : {predictions_file}")
    print(f"  Evaluation  : {evaluation_file}")
    if comparison_file.is_file():
        print(f"  Comparaison : {comparison_file}")
    print(f"  Log API     : {api_log_file}")
    print(f"  Manifeste   : {manifest_file}")
    print("\nExpérience terminée.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nInterruption demandée. Les prédictions déjà validées sont conservées.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERREUR: {exc}", file=sys.stderr)
        raise
