#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FLEXID — Gender Counterfactual Sensitivity Audit

Usage:
  1) Save the final 61-line JSONL challenge as:
       data/gender_counterfactual_challenge.jsonl
     (the script also accepts the same file at the project root).
  2) Open this file in VS Code.
  3) Click "Run Python File".

No command-line arguments are required.

The script validates every pair against the official FLEXID v3 test split,
loads the three completed mDeBERTa joint checkpoints (2026/2027/2028), and
measures prediction sensitivity to the gender counterfactual transformation.

Scope: controlled gender-category / grammatical-gender sensitivity only;
this is not a comprehensive demographic-fairness audit.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import importlib.util
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any


# ============================================================================
# FROZEN CONFIGURATION
# ============================================================================

AUDIT_VERSION = "FLEXID-GENDER-COUNTERFACTUAL-v1.0"
AUDIT_SEED = 2026

SPLIT_RELATIVE_DIR = Path("data") / "flexid_exact_group_split"
CHALLENGE_FILENAME = "gender_counterfactual_challenge.jsonl"
OUTPUT_DIR_NAME = "gender_counterfactual_audit_v1"

EXPECTED_SPLIT_SIZES = {"train": 701, "validation": 151, "test": 150}
EXPECTED_CHALLENGE_SIZE = 61
LABELS = ("entailment", "contradiction", "neutral")
DIRECTIONS = ("feminine_to_masculine", "masculine_to_feminine")
SEEDS = (2026, 2027, 2028)

EXPECTED_LABEL_COUNTS = {
    "entailment": 25,
    "contradiction": 20,
    "neutral": 16,
}
EXPECTED_DIRECTION_COUNTS = {
    "feminine_to_masculine": 35,
    "masculine_to_feminine": 26,
}

BOOTSTRAP_REPS = 5000


# ============================================================================
# GENERAL UTILITIES
# ============================================================================

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            require(isinstance(row, dict), f"{path}:{line_no}: JSON object required.")
            rows.append(row)
    return rows


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
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [Path.cwd().resolve(), script_dir, *script_dir.parents]
    for candidate in candidates:
        split_dir = candidate / SPLIT_RELATIVE_DIR
        if all((split_dir / f"{name}.jsonl").is_file() for name in EXPECTED_SPLIT_SIZES):
            return candidate
    raise RuntimeError(
        "Could not find data/flexid_exact_group_split. Open the FLEXID project "
        "folder in VS Code and run again."
    )


def find_challenge_path(project_root: Path) -> Path:
    candidates = [
        project_root / "data" / CHALLENGE_FILENAME,
        project_root / CHALLENGE_FILENAME,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise RuntimeError(
        f"Challenge file not found. Save the final 61-line JSONL as:\n"
        f"  {project_root / 'data' / CHALLENGE_FILENAME}"
    )


def unique_output_dir(project_root: Path) -> Path:
    base = project_root / "data" / OUTPUT_DIR_NAME
    if not base.exists():
        return base
    for i in range(2, 100):
        candidate = project_root / "data" / f"{OUTPUT_DIR_NAME}_run{i:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Too many previous gender-audit output directories.")


def normalize_law_ref(value: Any) -> str:
    return " ".join(str(value or "").split())


def token_present(text: str, token: str) -> bool:
    import re
    pattern = rf"(?<![\wÀ-ÖØ-öø-ÿ]){re.escape(token)}(?![\wÀ-ÖØ-öø-ÿ])"
    return re.search(pattern, text) is not None


# ============================================================================
# OFFICIAL SPLIT + CHALLENGE VALIDATION
# ============================================================================

REQUIRED_FIELDS = {
    "base_id", "label", "original_name", "counterfactual_name", "direction",
    "original_hypothesis_facts", "counterfactual_hypothesis_facts",
    "premise", "law_ref", "decision",
}


def validate_official_split(project_root: Path) -> dict[str, list[dict]]:
    split_dir = project_root / SPLIT_RELATIVE_DIR
    splits = {
        name: read_jsonl(split_dir / f"{name}.jsonl")
        for name in EXPECTED_SPLIT_SIZES
    }
    for name, expected in EXPECTED_SPLIT_SIZES.items():
        require(len(splits[name]) == expected,
                f"{name}: expected {expected}, got {len(splits[name])}.")
        require(set(row.get("label") for row in splits[name]) <= set(LABELS),
                f"{name}: unexpected label.")
    ids = [row["id"] for row in splits["test"]]
    require(len(ids) == len(set(ids)), "Duplicate IDs in official test split.")
    return splits


def text_change_report(original: str, counterfactual: str) -> dict:
    a = original.split()
    b = counterfactual.split()
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    edits = []
    changed_a = changed_b = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed_a += i2 - i1
        changed_b += j2 - j1
        edits.append({
            "tag": tag,
            "original": " ".join(a[i1:i2]),
            "counterfactual": " ".join(b[j1:j2]),
        })
    return {
        "changed_original_tokens": changed_a,
        "changed_counterfactual_tokens": changed_b,
        "edit_blocks": len(edits),
        "edits": edits,
    }


def validate_challenge(challenge_rows: list[dict], official_test: list[dict]) -> tuple[list[dict], dict]:
    require(len(challenge_rows) == EXPECTED_CHALLENGE_SIZE,
            f"Expected exactly {EXPECTED_CHALLENGE_SIZE} challenge rows, got {len(challenge_rows)}.")

    official_by_id = {row["id"]: row for row in official_test}
    seen = set()
    validated = []
    edit_reports = []

    for line_no, row in enumerate(challenge_rows, 1):
        missing = REQUIRED_FIELDS - set(row)
        require(not missing, f"Challenge line {line_no}: missing {sorted(missing)}.")

        base_id = row["base_id"]
        require(base_id not in seen, f"Duplicate base_id: {base_id}.")
        seen.add(base_id)
        require(base_id in official_by_id, f"{base_id}: not in official test split.")
        official = official_by_id[base_id]

        require(row["decision"] == "KEEP", f"{base_id}: decision must be KEEP.")
        require(row["label"] in LABELS, f"{base_id}: invalid label.")
        require(row["direction"] in DIRECTIONS, f"{base_id}: invalid direction.")
        require(row["label"] == official["label"], f"{base_id}: gold label changed.")
        require(row["premise"] == official["premise"], f"{base_id}: premise changed.")
        require(row["original_hypothesis_facts"] == official["hypothesis_facts"],
                f"{base_id}: original hypothesis differs from official test.")

        official_law_ref = (official.get("meta") or {}).get("law_ref")
        require(normalize_law_ref(row["law_ref"]) == normalize_law_ref(official_law_ref),
                f"{base_id}: law_ref differs from official test.")

        original = row["original_hypothesis_facts"]
        counter = row["counterfactual_hypothesis_facts"]
        original_name = row["original_name"]
        counter_name = row["counterfactual_name"]

        require(isinstance(original, str) and original, f"{base_id}: empty original text.")
        require(isinstance(counter, str) and counter, f"{base_id}: empty counterfactual text.")
        require(original != counter, f"{base_id}: original and counterfactual identical.")
        require(original_name != counter_name, f"{base_id}: names are identical.")
        require(token_present(original, original_name),
                f"{base_id}: original_name not found in original hypothesis.")
        require(token_present(counter, counter_name),
                f"{base_id}: counterfactual_name not found in counterfactual hypothesis.")

        report = text_change_report(original, counter)
        report["base_id"] = base_id
        edit_reports.append(report)

        validated.append({
            "base_id": base_id,
            "label": row["label"],
            "direction": row["direction"],
            "original_name": original_name,
            "counterfactual_name": counter_name,
            "premise": row["premise"],
            "law_ref": official_law_ref,
            "original_hypothesis_facts": original,
            "counterfactual_hypothesis_facts": counter,
        })

    label_counts = Counter(row["label"] for row in validated)
    direction_counts = Counter(row["direction"] for row in validated)

    require(dict(label_counts) == EXPECTED_LABEL_COUNTS,
            f"Unexpected label counts: {dict(label_counts)}; expected {EXPECTED_LABEL_COUNTS}.")
    require(dict(direction_counts) == EXPECTED_DIRECTION_COUNTS,
            f"Unexpected direction counts: {dict(direction_counts)}; expected {EXPECTED_DIRECTION_COUNTS}.")

    report = {
        "challenge_size": len(validated),
        "unique_base_ids": len(seen),
        "label_counts": dict(label_counts),
        "direction_counts": dict(direction_counts),
        "official_test_alignment": True,
        "gold_labels_unchanged": True,
        "premises_unchanged": True,
        "original_hypotheses_match_official_test": True,
        "law_refs_match_after_whitespace_normalization": True,
        "edit_summary": {
            "mean_changed_original_tokens": statistics.mean(x["changed_original_tokens"] for x in edit_reports),
            "mean_changed_counterfactual_tokens": statistics.mean(x["changed_counterfactual_tokens"] for x in edit_reports),
            "max_changed_original_tokens": max(x["changed_original_tokens"] for x in edit_reports),
            "max_changed_counterfactual_tokens": max(x["changed_counterfactual_tokens"] for x in edit_reports),
            "mean_edit_blocks": statistics.mean(x["edit_blocks"] for x in edit_reports),
        },
        "edit_reports": edit_reports,
    }
    return validated, report


# ============================================================================
# BOOTSTRAP + PAIRED TESTS
# ============================================================================

def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def bootstrap_mean(values_by_id: dict[str, float], seed: int) -> dict:
    ids = sorted(values_by_id)
    require(ids, "Cannot bootstrap an empty metric.")
    observed = statistics.mean(values_by_id[i] for i in ids)
    rng = random.Random(seed)
    reps = []
    for _ in range(BOOTSTRAP_REPS):
        sample = [rng.choice(ids) for _ in ids]
        reps.append(statistics.mean(values_by_id[i] for i in sample))
    return {
        "mean": observed,
        "ci95": [percentile(reps, 0.025), percentile(reps, 0.975)],
        "bootstrap_reps": BOOTSTRAP_REPS,
        "resampling_unit": "base_pair",
    }


def exact_mcnemar_p_value(harmful: int, beneficial: int) -> float:
    n = harmful + beneficial
    if n == 0:
        return 1.0
    k = min(harmful, beneficial)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    return 0.5 * sum(abs(float(p[label]) - float(q[label])) for label in LABELS)


# ============================================================================
# LOAD EXISTING mDeBERTa CHECKPOINTS
# ============================================================================

def find_model_script(project_root: Path) -> Path:
    candidates = [
        project_root / "scripts" / "mdeberta_join_label_rationales.py",
        project_root / "mdeberta_join_label_rationales.py",
        project_root / "scripts" / "train_mdeberta_joint_nli_rationale_final.py",
        project_root / "train_mdeberta_joint_nli_rationale_final.py",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise RuntimeError("Could not find mdeberta_join_label_rationales.py.")


def import_model_module(path: Path):
    # Python 3.13 dataclasses require registration before exec_module().
    module_name = "flexid_mdeberta_joint_gender_audit"
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, f"Cannot import {path}.")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def find_latest_complete_result_dir(project_root: Path) -> Path:
    candidates = sorted(
        [p for p in (project_root / "data").glob("results_mdeberta_joint_final*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for directory in candidates:
        ok = True
        for seed in SEEDS:
            best = directory / f"seed_{seed}" / "best"
            if not best.is_dir() or not (
                (best / "model.safetensors").is_file()
                or (best / "pytorch_model.bin").is_file()
            ):
                ok = False
                break
        if ok:
            return directory
    raise RuntimeError("No completed 3-seed mDeBERTa checkpoint directory found.")


def prediction_rows(challenge: list[dict], counterfactual: bool) -> list[dict]:
    rows = []
    for row in challenge:
        rows.append({
            "id": row["base_id"] + ("__cf" if counterfactual else "__orig"),
            "premise": row["premise"],
            "hypothesis_facts": (
                row["counterfactual_hypothesis_facts"]
                if counterfactual
                else row["original_hypothesis_facts"]
            ),
        })
    return rows


# ============================================================================
# ONE SEED
# ============================================================================

def evaluate_one_seed(module: Any, checkpoint: Path, challenge: list[dict], seed: int) -> tuple[dict, list[dict]]:
    deps = module.dependencies()
    args = SimpleNamespace(
        device="auto",
        precision="auto",
        eval_batch_size=4,
        window_batch_budget=8,
    )
    device, precision = module.device_and_precision(args, deps.torch)
    model, tokenizer, settings = module.load_checkpoint(checkpoint, deps)
    model = model.float().to(device)

    orig_items, _ = module.prepare_records(
        prediction_rows(challenge, False), tokenizer, settings, labelled=False
    )
    cf_items, _ = module.prepare_records(
        prediction_rows(challenge, True), tokenizer, settings, labelled=False
    )

    orig_preds = module.predict_items(
        model, orig_items, tokenizer, settings, args, deps, device, precision
    )
    cf_preds = module.predict_items(
        model, cf_items, tokenizer, settings, args, deps, device, precision
    )

    orig = {p["id"].removesuffix("__orig"): p for p in orig_preds}
    cf = {p["id"].removesuffix("__cf"): p for p in cf_preds}
    expected_ids = {row["base_id"] for row in challenge}
    require(set(orig) == expected_ids, f"Seed {seed}: incomplete original predictions.")
    require(set(cf) == expected_ids, f"Seed {seed}: incomplete counterfactual predictions.")

    maps = {key: {} for key in (
        "flip", "harmful", "beneficial", "orig_correct", "cf_correct",
        "tv", "abs_gold_shift", "signed_gold_shift"
    )}
    details = []

    for row in challenge:
        base_id = row["base_id"]
        gold = row["label"]
        po, pc = orig[base_id], cf[base_id]
        lo, lc = po["label"], pc["label"]
        qo, qc = po["label_probabilities"], pc["label_probabilities"]

        orig_correct = lo == gold
        cf_correct = lc == gold
        flip = lo != lc
        harmful = orig_correct and not cf_correct
        beneficial = (not orig_correct) and cf_correct
        tv = total_variation(qo, qc)
        signed = float(qc[gold]) - float(qo[gold])

        maps["flip"][base_id] = float(flip)
        maps["harmful"][base_id] = float(harmful)
        maps["beneficial"][base_id] = float(beneficial)
        maps["orig_correct"][base_id] = float(orig_correct)
        maps["cf_correct"][base_id] = float(cf_correct)
        maps["tv"][base_id] = tv
        maps["abs_gold_shift"][base_id] = abs(signed)
        maps["signed_gold_shift"][base_id] = signed

        details.append({
            "seed": seed,
            "base_id": base_id,
            "label": gold,
            "direction": row["direction"],
            "original_name": row["original_name"],
            "counterfactual_name": row["counterfactual_name"],
            "original_prediction": lo,
            "counterfactual_prediction": lc,
            "label_flip": int(flip),
            "original_correct": int(orig_correct),
            "counterfactual_correct": int(cf_correct),
            "harmful_flip": int(harmful),
            "beneficial_flip": int(beneficial),
            "original_gold_probability": float(qo[gold]),
            "counterfactual_gold_probability": float(qc[gold]),
            "signed_gold_probability_shift": signed,
            "absolute_gold_probability_shift": abs(signed),
            "total_variation_distance": tv,
            "original_prob_entailment": float(qo["entailment"]),
            "original_prob_contradiction": float(qo["contradiction"]),
            "original_prob_neutral": float(qo["neutral"]),
            "counterfactual_prob_entailment": float(qc["entailment"]),
            "counterfactual_prob_contradiction": float(qc["contradiction"]),
            "counterfactual_prob_neutral": float(qc["neutral"]),
        })

    def breakdown(field: str, values: tuple[str, ...]) -> dict:
        out = {}
        for value in values:
            subset = [x for x in details if x[field] == value]
            out[value] = {
                "n": len(subset),
                "flip_rate": statistics.mean(x["label_flip"] for x in subset),
                "harmful_flip_rate": statistics.mean(x["harmful_flip"] for x in subset),
                "beneficial_flip_rate": statistics.mean(x["beneficial_flip"] for x in subset),
                "original_accuracy": statistics.mean(x["original_correct"] for x in subset),
                "counterfactual_accuracy": statistics.mean(x["counterfactual_correct"] for x in subset),
                "mean_total_variation_distance": statistics.mean(x["total_variation_distance"] for x in subset),
                "mean_absolute_gold_probability_shift": statistics.mean(x["absolute_gold_probability_shift"] for x in subset),
                "mean_signed_gold_probability_shift": statistics.mean(x["signed_gold_probability_shift"] for x in subset),
            }
        return out

    by_direction = breakdown("direction", DIRECTIONS)
    by_label = breakdown("label", LABELS)
    harmful_count = sum(x["harmful_flip"] for x in details)
    beneficial_count = sum(x["beneficial_flip"] for x in details)

    summary = {
        "seed": seed,
        "checkpoint": str(checkpoint),
        "device": str(device),
        "precision": precision,
        "n_pairs": len(challenge),
        "original_accuracy": bootstrap_mean(maps["orig_correct"], AUDIT_SEED + seed + 1),
        "counterfactual_accuracy": bootstrap_mean(maps["cf_correct"], AUDIT_SEED + seed + 2),
        "flip_rate": bootstrap_mean(maps["flip"], AUDIT_SEED + seed + 3),
        "harmful_flip_rate": bootstrap_mean(maps["harmful"], AUDIT_SEED + seed + 4),
        "beneficial_flip_rate": bootstrap_mean(maps["beneficial"], AUDIT_SEED + seed + 5),
        "mean_total_variation_distance": bootstrap_mean(maps["tv"], AUDIT_SEED + seed + 6),
        "mean_absolute_gold_probability_shift": bootstrap_mean(maps["abs_gold_shift"], AUDIT_SEED + seed + 7),
        "mean_signed_gold_probability_shift": bootstrap_mean(maps["signed_gold_shift"], AUDIT_SEED + seed + 8),
        "correctness_discordance": {
            "harmful_count": harmful_count,
            "beneficial_count": beneficial_count,
            "exact_mcnemar_p_value": exact_mcnemar_p_value(harmful_count, beneficial_count),
        },
        "direction_balanced_flip_rate": statistics.mean(by_direction[d]["flip_rate"] for d in DIRECTIONS),
        "direction_balanced_harmful_flip_rate": statistics.mean(by_direction[d]["harmful_flip_rate"] for d in DIRECTIONS),
        "by_direction": by_direction,
        "by_label": by_label,
    }

    del model
    if getattr(device, "type", None) == "cuda":
        deps.torch.cuda.empty_cache()
    return summary, details


# ============================================================================
# MULTI-SEED AGGREGATION
# ============================================================================

def scalar(summary: dict, key: str) -> float:
    value = summary[key]
    return float(value["mean"] if isinstance(value, dict) and "mean" in value else value)


def aggregate_seeds(summaries: list[dict]) -> dict:
    keys = (
        "original_accuracy", "counterfactual_accuracy", "flip_rate",
        "harmful_flip_rate", "beneficial_flip_rate",
        "mean_total_variation_distance", "mean_absolute_gold_probability_shift",
        "mean_signed_gold_probability_shift", "direction_balanced_flip_rate",
        "direction_balanced_harmful_flip_rate",
    )
    out = {}
    for key in keys:
        values = [scalar(s, key) for s in summaries]
        out[key] = {
            "values": values,
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    out["by_direction"] = {}
    for direction in DIRECTIONS:
        out["by_direction"][direction] = {}
        for key in (
            "flip_rate", "harmful_flip_rate", "beneficial_flip_rate",
            "original_accuracy", "counterfactual_accuracy",
            "mean_total_variation_distance", "mean_absolute_gold_probability_shift",
            "mean_signed_gold_probability_shift",
        ):
            values = [s["by_direction"][direction][key] for s in summaries]
            out["by_direction"][direction][key] = {
                "values": values,
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }

    out["by_label"] = {}
    for label in LABELS:
        out["by_label"][label] = {}
        for key in (
            "flip_rate", "harmful_flip_rate", "beneficial_flip_rate",
            "original_accuracy", "counterfactual_accuracy",
            "mean_total_variation_distance",
        ):
            values = [s["by_label"][label][key] for s in summaries]
            out["by_label"][label][key] = {
                "values": values,
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    return out


def cross_seed_pair_stability(details: list[dict]) -> dict:
    by_pair = defaultdict(list)
    for row in details:
        by_pair[row["base_id"]].append(row)
    require(all(len(rows) == len(SEEDS) for rows in by_pair.values()),
            "Incomplete cross-seed pair results.")

    any_flip = {bid: float(any(x["label_flip"] for x in rows)) for bid, rows in by_pair.items()}
    any_harmful = {bid: float(any(x["harmful_flip"] for x in rows)) for bid, rows in by_pair.items()}
    return {
        "pairs_with_flip_in_at_least_one_seed": int(sum(any_flip.values())),
        "rate_pairs_with_flip_in_at_least_one_seed": statistics.mean(any_flip.values()),
        "pairs_with_harmful_flip_in_at_least_one_seed": int(sum(any_harmful.values())),
        "rate_pairs_with_harmful_flip_in_at_least_one_seed": statistics.mean(any_harmful.values()),
        "pairs_stable_in_all_three_seeds": len(by_pair) - int(sum(any_flip.values())),
        "rate_pairs_stable_in_all_three_seeds": 1.0 - statistics.mean(any_flip.values()),
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    project_root = find_project_root()
    challenge_path = find_challenge_path(project_root)
    output_dir = unique_output_dir(project_root)
    output_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 80)
    print("FLEXID — GENDER COUNTERFACTUAL SENSITIVITY AUDIT")
    print("=" * 80)
    print(f"Project       : {project_root}")
    print(f"Challenge     : {challenge_path}")
    print(f"Output        : {output_dir}")
    print(f"Audit version : {AUDIT_VERSION}")

    splits = validate_official_split(project_root)
    challenge, validation = validate_challenge(read_jsonl(challenge_path), splits["test"])

    print("\nChallenge validation")
    print(f"  Pairs       : {len(challenge)}")
    print(f"  Labels      : {validation['label_counts']}")
    print(f"  Directions  : {validation['direction_counts']}")
    print("  Official test alignment : OK")

    write_json(output_dir / "challenge_validation.json", {
        "audit_version": AUDIT_VERSION,
        "challenge_sha256": sha256_file(challenge_path),
        **validation,
    })

    model_script = find_model_script(project_root)
    result_dir = find_latest_complete_result_dir(project_root)
    module = import_model_module(model_script)

    print(f"\nModel script  : {model_script}")
    print(f"Checkpoints   : {result_dir}")

    summaries = []
    all_details = []

    for seed in SEEDS:
        checkpoint = result_dir / f"seed_{seed}" / "best"
        print(f"\n--- Seed {seed} ---")
        summary, details = evaluate_one_seed(module, checkpoint, challenge, seed)
        summaries.append(summary)
        all_details.extend(details)

        print(f"Original accuracy       : {summary['original_accuracy']['mean']:.4f}")
        print(f"Counterfactual accuracy : {summary['counterfactual_accuracy']['mean']:.4f}")
        print(f"Flip rate               : {summary['flip_rate']['mean']:.4f} "
              f"[{summary['flip_rate']['ci95'][0]:.4f}, {summary['flip_rate']['ci95'][1]:.4f}]")
        print(f"Harmful flip rate       : {summary['harmful_flip_rate']['mean']:.4f}")
        print(f"Beneficial flip rate    : {summary['beneficial_flip_rate']['mean']:.4f}")
        print(f"Mean TV distance        : {summary['mean_total_variation_distance']['mean']:.4f}")
        print(f"F→M flip rate           : {summary['by_direction']['feminine_to_masculine']['flip_rate']:.4f}")
        print(f"M→F flip rate           : {summary['by_direction']['masculine_to_feminine']['flip_rate']:.4f}")

    aggregate = aggregate_seeds(summaries)
    cross_seed = cross_seed_pair_stability(all_details)

    final = {
        "audit_version": AUDIT_VERSION,
        "audit_seed": AUDIT_SEED,
        "scope": {
            "claim": "controlled gender-category / grammatical-gender sensitivity",
            "not_a_claim": "comprehensive demographic fairness",
            "source_split": "official FLEXID test split",
            "challenge_pairs": len(challenge),
            "counterfactuals_per_pair": 1,
        },
        "challenge": {
            "path": str(challenge_path),
            "sha256": sha256_file(challenge_path),
            "validation": validation,
        },
        "model": {
            "script": str(model_script),
            "result_directory": str(result_dir),
            "seeds": list(SEEDS),
        },
        "per_seed": summaries,
        "aggregate_across_model_seeds": aggregate,
        "cross_seed_pair_stability": cross_seed,
    }

    write_json(output_dir / "summary.json", final)
    write_csv(output_dir / "pair_level_results.csv", all_details)

    print("\n" + "=" * 80)
    print("AGGREGATE ACROSS 3 MODEL SEEDS")
    print("=" * 80)
    for key in (
        "original_accuracy", "counterfactual_accuracy", "flip_rate",
        "harmful_flip_rate", "beneficial_flip_rate",
        "mean_total_variation_distance", "mean_absolute_gold_probability_shift",
        "mean_signed_gold_probability_shift", "direction_balanced_flip_rate",
        "direction_balanced_harmful_flip_rate",
    ):
        item = aggregate[key]
        print(f"{key:48s} {item['mean']:.4f} ± {item['std']:.4f}")

    print("\nBy direction")
    for direction in DIRECTIONS:
        d = aggregate["by_direction"][direction]
        print(f"  {direction:26s} "
              f"flip={d['flip_rate']['mean']:.4f} ± {d['flip_rate']['std']:.4f}  "
              f"harmful={d['harmful_flip_rate']['mean']:.4f} ± {d['harmful_flip_rate']['std']:.4f}")

    print("\nCross-seed pair stability")
    print(f"  Stable in all 3 seeds : {cross_seed['pairs_stable_in_all_three_seeds']}/{len(challenge)} "
          f"({cross_seed['rate_pairs_stable_in_all_three_seeds']:.4f})")
    print(f"  Any harmful flip      : {cross_seed['pairs_with_harmful_flip_in_at_least_one_seed']}/{len(challenge)} "
          f"({cross_seed['rate_pairs_with_harmful_flip_in_at_least_one_seed']:.4f})")

    print("\nMain files")
    print(f"  {output_dir / 'summary.json'}")
    print(f"  {output_dir / 'pair_level_results.csv'}")
    print(f"  {output_dir / 'challenge_validation.json'}")
    print("\nAudit complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[FLEXID GENDER AUDIT ERROR] {exc}", file=sys.stderr)
        raise
