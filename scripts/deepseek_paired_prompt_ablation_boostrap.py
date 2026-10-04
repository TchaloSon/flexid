#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from __future__ import annotations

import csv
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


# ============================================================================
# CONFIGURATION
# ============================================================================

PROTOCOL_VERSION = "FLEXID-DEEPSEEK-PAIRED-ABLATION-v1.0"

ZERO_PREDICTIONS = Path("data") / "flexid_deepseek_v4_flash_predictions.jsonl"
FEW_PREDICTIONS = (
    Path("data") / "flexid_deepseek_v4_flash_fewshot_thinking_predictions.jsonl"
)

SPLIT_DIR = Path("data") / "flexid_exact_group_split"
SPLIT_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl")

EXPECTED_N = 180
LABELS = ("entailment", "contradiction", "neutral")
NON_NEUTRAL = {"entailment", "contradiction"}

BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20260926

OUTPUT_DIR = Path("data") / "deepseek_paired_prompt_ablation"
SUMMARY_FILE = "paired_ablation_summary.json"
DETAIL_FILE = "paired_instance_comparison.csv"

TOKEN_PATTERN = re.compile(r"\S+")


# ============================================================================
# IO / PROJECT ROOT
# ============================================================================

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
                raise RuntimeError(
                    f"{path}, ligne {line_no}: JSON invalide: {exc}"
                ) from exc
            require(isinstance(row, dict),
                    f"{path}, ligne {line_no}: objet JSON attendu.")
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def find_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [Path.cwd().resolve(), script_dir, *script_dir.parents]

    seen = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)

        if (
            (root / ZERO_PREDICTIONS).is_file()
            and (root / FEW_PREDICTIONS).is_file()
            and all((root / SPLIT_DIR / name).is_file() for name in SPLIT_FILES)
        ):
            return root

    raise RuntimeError(
        "Racine FLEXID introuvable. Le script attend les deux fichiers de "
        "prédictions DeepSeek dans data/ et les trois splits officiels dans "
        "data/flexid_exact_group_split/."
    )


# ============================================================================
# GOLD RATIONALE TOKEN SPANS
# ============================================================================

def whitespace_matches(text: str):
    return list(TOKEN_PATTERN.finditer(text))


def map_char_span_to_token_span(
    premise: str,
    start_char: int,
    end_char: int,
) -> tuple[int, int]:
    require(
        0 <= start_char < end_char <= len(premise),
        f"Span caractère invalide [{start_char}, {end_char}) "
        f"pour longueur {len(premise)}."
    )

    matches = whitespace_matches(premise)
    covered = [
        i
        for i, match in enumerate(matches, 1)
        if match.start() < end_char and match.end() > start_char
    ]
    require(covered, "Aucun token couvert par le rationale gold.")
    return covered[0], covered[-1]


def gold_token_span(row: dict[str, Any]) -> tuple[int | None, int | None]:
    label = row["label"]
    if label == "neutral":
        return None, None

    premise = row["premise"]

    start = row.get("rationale_start")
    end = row.get("rationale_end")
    rationale_text = row.get("rationale_text")

    if isinstance(start, int) and isinstance(end, int) and end > start:
        if isinstance(rationale_text, str) and rationale_text:
            require(
                premise[start:end] == rationale_text,
                f"{row['id']}: rationale_text != premise[start:end]."
            )
        return map_char_span_to_token_span(premise, start, end)

    token_start = row.get("rationale_start_token")
    token_end = row.get("rationale_end_token")
    if isinstance(token_start, int) and isinstance(token_end, int):
        require(
            1 <= token_start <= token_end,
            f"{row['id']}: span token gold invalide."
        )
        return token_start, token_end

    if isinstance(rationale_text, str) and rationale_text:
        start = premise.find(rationale_text)
        require(start >= 0, f"{row['id']}: rationale_text introuvable.")
        end = start + len(rationale_text)
        return map_char_span_to_token_span(premise, start, end)

    raise RuntimeError(
        f"{row['id']}: impossible de reconstruire le rationale gold."
    )


def span_iou(
    gold_start: int | None,
    gold_end: int | None,
    pred_start: int | None,
    pred_end: int | None,
) -> float:
    gold = (
        set(range(gold_start, gold_end + 1))
        if gold_start is not None and gold_end is not None
        else set()
    )
    pred = (
        set(range(pred_start, pred_end + 1))
        if pred_start is not None and pred_end is not None
        else set()
    )

    union = gold | pred
    if not union:
        return 1.0
    return len(gold & pred) / len(union)


# ============================================================================
# LABEL METRICS
# ============================================================================

def accuracy(gold: list[str], pred: list[str]) -> float:
    require(len(gold) == len(pred) and gold, "Accuracy: listes invalides.")
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def macro_f1(gold: list[str], pred: list[str]) -> float:
    require(len(gold) == len(pred) and gold, "Macro-F1: listes invalides.")
    f1s = []

    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1s.append(f1)

    return statistics.mean(f1s)


# ============================================================================
# EXACT McNEMAR
# ============================================================================

def exact_mcnemar_p_value(
    zero_correct_few_wrong: int,
    zero_wrong_few_correct: int,
) -> float:
    """
    Exact two-sided McNemar test = exact binomial test on discordant pairs
    under H0: p = 0.5.
    """
    b = zero_correct_few_wrong
    c = zero_wrong_few_correct
    n = b + c

    if n == 0:
        return 1.0

    k = min(b, c)
    lower_tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * lower_tail)


# ============================================================================
# PAIRED BOOTSTRAP
# ============================================================================

def percentile(sorted_values: list[float], q: float) -> float:
    require(sorted_values, "Percentile sur liste vide.")
    if len(sorted_values) == 1:
        return sorted_values[0]

    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return sorted_values[lo]

    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def paired_bootstrap(
    gold: list[str],
    zero_pred: list[str],
    few_pred: list[str],
    zero_joint: list[float],
    few_joint: list[float],
) -> dict[str, Any]:
    """
    Resamples INSTANCE INDICES with replacement.
    The same sampled indices are used for zero-shot and few-shot:
    this is what makes the bootstrap paired.
    """
    n = len(gold)
    require(
        n == len(zero_pred) == len(few_pred) == len(zero_joint) == len(few_joint),
        "Paired bootstrap: longueurs incompatibles."
    )

    observed = {
        "accuracy_delta":
            accuracy(gold, few_pred) - accuracy(gold, zero_pred),
        "macro_f1_delta":
            macro_f1(gold, few_pred) - macro_f1(gold, zero_pred),
        "joint_iou_at_0_50_delta":
            statistics.mean(few_joint) - statistics.mean(zero_joint),
    }

    rng = random.Random(BOOTSTRAP_SEED)

    delta_accuracy = []
    delta_macro_f1 = []
    delta_joint = []

    for _ in range(BOOTSTRAP_REPS):
        indices = [rng.randrange(n) for _ in range(n)]

        g = [gold[i] for i in indices]
        z = [zero_pred[i] for i in indices]
        f = [few_pred[i] for i in indices]

        zj = [zero_joint[i] for i in indices]
        fj = [few_joint[i] for i in indices]

        delta_accuracy.append(accuracy(g, f) - accuracy(g, z))
        delta_macro_f1.append(macro_f1(g, f) - macro_f1(g, z))
        delta_joint.append(statistics.mean(fj) - statistics.mean(zj))

    result = {}

    for key, values in (
        ("accuracy_delta", delta_accuracy),
        ("macro_f1_delta", delta_macro_f1),
        ("joint_iou_at_0_50_delta", delta_joint),
    ):
        values.sort()
        result[key] = {
            "observed": observed[key],
            "ci95_percentile": [
                percentile(values, 0.025),
                percentile(values, 0.975),
            ],
            # Descriptif, PAS une p-value.
            "bootstrap_fraction_delta_gt_0":
                sum(v > 0 for v in values) / len(values),
            "bootstrap_fraction_delta_lt_0":
                sum(v < 0 for v in values) / len(values),
        }

    result["bootstrap_reps"] = BOOTSTRAP_REPS
    result["bootstrap_seed"] = BOOTSTRAP_SEED
    result["resampling_unit"] = "paired_instance"

    return result


# ============================================================================
# LOAD / VALIDATE PREDICTIONS
# ============================================================================

def validate_prediction_row(row: dict[str, Any], source: str) -> None:
    rid = row.get("id")
    require(isinstance(rid, str) and rid, f"{source}: id invalide.")
    require(row.get("label") in LABELS,
            f"{source}/{rid}: label invalide.")

    start = row.get("rationale_start_token")
    end = row.get("rationale_end_token")

    if row["label"] == "neutral":
        require(
            start is None and end is None,
            f"{source}/{rid}: neutral doit avoir rationale null/null."
        )
    else:
        require(
            isinstance(start, int) and not isinstance(start, bool)
            and isinstance(end, int) and not isinstance(end, bool)
            and 1 <= start <= end,
            f"{source}/{rid}: span rationale invalide."
        )


def index_predictions(
    rows: list[dict[str, Any]],
    source: str,
) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        validate_prediction_row(row, source)
        rid = row["id"]
        require(rid not in result, f"{source}: ID dupliqué {rid}.")
        result[rid] = row
    return result


def load_gold(project_root: Path) -> dict[str, dict[str, Any]]:
    gold = {}

    for filename in SPLIT_FILES:
        rows = read_jsonl(project_root / SPLIT_DIR / filename)
        for row in rows:
            rid = row.get("id")
            require(isinstance(rid, str) and rid,
                    f"{filename}: id gold invalide.")
            require(rid not in gold, f"Gold ID dupliqué: {rid}.")
            require(row.get("label") in LABELS,
                    f"{rid}: label gold invalide.")
            gold[rid] = row

    return gold


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    root = find_project_root()

    zero_path = root / ZERO_PREDICTIONS
    few_path = root / FEW_PREDICTIONS
    out_dir = root / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    zero = index_predictions(read_jsonl(zero_path), "zero-shot")
    few = index_predictions(read_jsonl(few_path), "few-shot")
    gold = load_gold(root)

    zero_ids = set(zero)
    few_ids = set(few)

    require(
        zero_ids == few_ids,
        "Les IDs zero-shot et few-shot ne sont pas exactement identiques."
    )
    require(
        len(zero_ids) == EXPECTED_N,
        f"Nombre de paires: {len(zero_ids)} != {EXPECTED_N}."
    )

    missing_gold = sorted(zero_ids - set(gold))
    require(
        not missing_gold,
        "IDs absents du gold officiel: " + ", ".join(missing_gold[:20])
    )

    # Conserve l'ordre du fichier zero-shot pour la traçabilité.
    ordered_ids = [row["id"] for row in read_jsonl(zero_path)]
    require(len(ordered_ids) == EXPECTED_N, "Ordre zero-shot invalide.")

    gold_labels = []
    zero_labels = []
    few_labels = []
    zero_joint = []
    few_joint = []
    details = []

    zero_correct_few_wrong = 0
    zero_wrong_few_correct = 0
    both_correct = 0
    both_wrong = 0

    for rid in ordered_ids:
        g = gold[rid]
        z = zero[rid]
        f = few[rid]

        gold_label = g["label"]
        z_label = z["label"]
        f_label = f["label"]

        z_correct = z_label == gold_label
        f_correct = f_label == gold_label

        if z_correct and f_correct:
            both_correct += 1
        elif z_correct and not f_correct:
            zero_correct_few_wrong += 1
        elif (not z_correct) and f_correct:
            zero_wrong_few_correct += 1
        else:
            both_wrong += 1

        gold_start, gold_end = gold_token_span(g)

        if gold_label == "neutral":
            z_joint = float(
                z_label == "neutral"
                and z["rationale_start_token"] is None
                and z["rationale_end_token"] is None
            )
            f_joint = float(
                f_label == "neutral"
                and f["rationale_start_token"] is None
                and f["rationale_end_token"] is None
            )
            z_iou = None
            f_iou = None
        else:
            if z_label == gold_label:
                z_iou = span_iou(
                    gold_start,
                    gold_end,
                    z["rationale_start_token"],
                    z["rationale_end_token"],
                )
                z_joint = float(z_iou >= 0.50)
            else:
                z_iou = None
                z_joint = 0.0

            if f_label == gold_label:
                f_iou = span_iou(
                    gold_start,
                    gold_end,
                    f["rationale_start_token"],
                    f["rationale_end_token"],
                )
                f_joint = float(f_iou >= 0.50)
            else:
                f_iou = None
                f_joint = 0.0

        gold_labels.append(gold_label)
        zero_labels.append(z_label)
        few_labels.append(f_label)
        zero_joint.append(z_joint)
        few_joint.append(f_joint)

        details.append({
            "id": rid,
            "gold_label": gold_label,
            "zero_label": z_label,
            "few_label": f_label,
            "zero_correct": int(z_correct),
            "few_correct": int(f_correct),
            "correctness_transition": (
                "both_correct"
                if z_correct and f_correct
                else "zero_correct_few_wrong"
                if z_correct and not f_correct
                else "zero_wrong_few_correct"
                if (not z_correct) and f_correct
                else "both_wrong"
            ),
            "prediction_changed": int(z_label != f_label),
            "gold_rationale_start_token": gold_start,
            "gold_rationale_end_token": gold_end,
            "zero_rationale_start_token": z["rationale_start_token"],
            "zero_rationale_end_token": z["rationale_end_token"],
            "few_rationale_start_token": f["rationale_start_token"],
            "few_rationale_end_token": f["rationale_end_token"],
            "zero_conditional_iou": z_iou,
            "few_conditional_iou": f_iou,
            "zero_joint_iou_at_0_50": z_joint,
            "few_joint_iou_at_0_50": f_joint,
            "joint_transition": (
                "0_to_1" if z_joint == 0 and f_joint == 1
                else "1_to_0" if z_joint == 1 and f_joint == 0
                else "same"
            ),
        })

    zero_accuracy = accuracy(gold_labels, zero_labels)
    few_accuracy = accuracy(gold_labels, few_labels)

    zero_macro_f1 = macro_f1(gold_labels, zero_labels)
    few_macro_f1 = macro_f1(gold_labels, few_labels)

    zero_joint_mean = statistics.mean(zero_joint)
    few_joint_mean = statistics.mean(few_joint)

    mcnemar_p = exact_mcnemar_p_value(
        zero_correct_few_wrong,
        zero_wrong_few_correct,
    )

    bootstrap = paired_bootstrap(
        gold_labels,
        zero_labels,
        few_labels,
        zero_joint,
        few_joint,
    )

    joint_0_to_1 = sum(
        z == 0 and f == 1 for z, f in zip(zero_joint, few_joint)
    )
    joint_1_to_0 = sum(
        z == 1 and f == 0 for z, f in zip(zero_joint, few_joint)
    )

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "n_paired_instances": EXPECTED_N,
        "conditions": {
            "zero_shot_thinking": {
                "accuracy": zero_accuracy,
                "macro_f1": zero_macro_f1,
                "joint_iou_at_0_50": zero_joint_mean,
            },
            "few_shot_thinking": {
                "accuracy": few_accuracy,
                "macro_f1": few_macro_f1,
                "joint_iou_at_0_50": few_joint_mean,
            },
        },
        "observed_deltas_few_minus_zero": {
            "accuracy": few_accuracy - zero_accuracy,
            "macro_f1": few_macro_f1 - zero_macro_f1,
            "joint_iou_at_0_50": few_joint_mean - zero_joint_mean,
        },
        "paired_correctness_transitions": {
            "both_correct": both_correct,
            "zero_correct_few_wrong": zero_correct_few_wrong,
            "zero_wrong_few_correct": zero_wrong_few_correct,
            "both_wrong": both_wrong,
        },
        "exact_mcnemar_test_accuracy": {
            "discordant_pairs": (
                zero_correct_few_wrong + zero_wrong_few_correct
            ),
            "zero_correct_few_wrong": zero_correct_few_wrong,
            "zero_wrong_few_correct": zero_wrong_few_correct,
            "two_sided_exact_p_value": mcnemar_p,
            "null_hypothesis":
                "Among discordant pairs, improvements and degradations "
                "are equally likely.",
        },
        "paired_bootstrap": bootstrap,
        "joint_transitions": {
            "zero_fail_to_few_pass": joint_0_to_1,
            "zero_pass_to_few_fail": joint_1_to_0,
        },
        "interpretation_guardrail": (
            "The bootstrap CIs quantify uncertainty of the paired deltas. "
            "The exact McNemar p-value tests paired correctness changes. "
            "A CI including zero does not prove equality; a p-value above "
            "0.05 does not prove absence of an effect."
        ),
    }

    write_json(out_dir / SUMMARY_FILE, summary)
    write_csv(out_dir / DETAIL_FILE, details)

    print("=" * 80)
    print("FLEXID — PAIRED PROMPT ABLATION")
    print("=" * 80)
    print(f"N paired instances       : {EXPECTED_N}")
    print("")
    print("ZERO-SHOT + THINKING")
    print(f"  Accuracy               : {zero_accuracy:.4f}")
    print(f"  Macro-F1               : {zero_macro_f1:.4f}")
    print(f"  Joint IoU@0.50         : {zero_joint_mean:.4f}")
    print("")
    print("FEW-SHOT + THINKING")
    print(f"  Accuracy               : {few_accuracy:.4f}")
    print(f"  Macro-F1               : {few_macro_f1:.4f}")
    print(f"  Joint IoU@0.50         : {few_joint_mean:.4f}")
    print("")
    print("OBSERVED DELTAS (few - zero)")
    print(f"  Accuracy               : {few_accuracy - zero_accuracy:+.4f}")
    print(f"  Macro-F1               : {few_macro_f1 - zero_macro_f1:+.4f}")
    print(f"  Joint IoU@0.50         : {few_joint_mean - zero_joint_mean:+.4f}")
    print("")
    print("EXACT McNEMAR — ACCURACY")
    print(f"  Zero correct -> Few wrong : {zero_correct_few_wrong}")
    print(f"  Zero wrong -> Few correct : {zero_wrong_few_correct}")
    print(f"  Two-sided exact p         : {mcnemar_p:.6f}")
    print("")
    print("PAIRED BOOTSTRAP 95% CI")
    for key, label in (
        ("accuracy_delta", "Delta Accuracy"),
        ("macro_f1_delta", "Delta Macro-F1"),
        ("joint_iou_at_0_50_delta", "Delta Joint IoU@0.50"),
    ):
        item = bootstrap[key]
        lo, hi = item["ci95_percentile"]
        print(
            f"  {label:22s}: {item['observed']:+.4f} "
            f"[{lo:+.4f}, {hi:+.4f}]"
        )

    print("")
    print("JOINT TRANSITIONS")
    print(f"  0 -> 1 : {joint_0_to_1}")
    print(f"  1 -> 0 : {joint_1_to_0}")
    print("")
    print("Fichiers:")
    print(f"  {out_dir / SUMMARY_FILE}")
    print(f"  {out_dir / DETAIL_FILE}")
    print("")
    print("Aucun appel API n'a été effectué.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
