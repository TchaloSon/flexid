#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_INSTANCES = 180
LABELS = ("entailment", "contradiction", "neutral")
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 2026
END_TOKEN_IS_EXCLUSIVE = False

FIRST_ANNOTATOR_FILE = Path("data/flexid_kappa_rational_first_annotator.jsonl")
SECOND_ANNOTATOR_FILE = Path("data/flexid_kappa_rational_second_annotator.jsonl")
OUTPUT_DIR = Path("data/results_evaluations/iaa_first_vs_second")


def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    for candidate in (script_path.parent, script_path.parent.parent):
        if (candidate / FIRST_ANNOTATOR_FILE).is_file() and (candidate / SECOND_ANNOTATOR_FILE).is_file():
            return candidate
    raise FileNotFoundError(
        "Impossible de trouver les deux fichiers. Place le script à la racine du projet ou dans scripts/."
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}, ligne {lineno}: JSON invalide: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}, ligne {lineno}: objet JSON attendu.")
            rows.append(obj)
    if not rows:
        raise ValueError(f"Fichier vide: {path}")
    return rows


def validate(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    if len(rows) != EXPECTED_INSTANCES:
        raise ValueError(f"{name}: {EXPECTED_INSTANCES} instances attendues, {len(rows)} trouvées.")

    by_id = {}
    required = {"id", "label", "rationale_start_token", "rationale_end_token"}

    for lineno, row in enumerate(rows, 1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{name}, ligne {lineno}: champs manquants {sorted(missing)}")

        rid = row["id"]
        label = row["label"]
        start = row["rationale_start_token"]
        end = row["rationale_end_token"]

        if not isinstance(rid, str) or not rid:
            raise ValueError(f"{name}, ligne {lineno}: id invalide.")
        if rid in by_id:
            raise ValueError(f"{name}: ID dupliqué: {rid}")
        if label not in LABELS:
            raise ValueError(f"{name}, {rid}: label invalide {label!r}")

        if label == "neutral":
            if start is not None or end is not None:
                raise ValueError(f"{name}, {rid}: neutral doit avoir start/end = null.")
        else:
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(f"{name}, {rid}: offsets token entiers requis pour un label non-neutral.")
            if start < 0 or end < 0:
                raise ValueError(f"{name}, {rid}: offsets négatifs interdits.")
            if END_TOKEN_IS_EXCLUSIVE and end <= start:
                raise ValueError(f"{name}, {rid}: end doit être > start (fin exclusive).")
            if not END_TOKEN_IS_EXCLUSIVE and end < start:
                raise ValueError(f"{name}, {rid}: end doit être >= start (fin inclusive).")

        by_id[rid] = row

    return by_id


def align(a1, a2):
    ids1, ids2 = set(a1), set(a2)
    if ids1 != ids2:
        raise ValueError(
            "Les deux fichiers n'ont pas exactement les mêmes IDs.\n"
            f"Seulement A1: {sorted(ids1-ids2)[:20]}\n"
            f"Seulement A2: {sorted(ids2-ids1)[:20]}"
        )
    return [(rid, a1[rid], a2[rid]) for rid in sorted(ids1)]


def raw_agreement(y1, y2):
    return sum(a == b for a, b in zip(y1, y2)) / len(y1)


def cohen_kappa(y1, y2):
    n = len(y1)
    po = raw_agreement(y1, y2)
    c1, c2 = Counter(y1), Counter(y2)
    pe = sum((c1[l] / n) * (c2[l] / n) for l in LABELS)
    if math.isclose(1.0 - pe, 0.0, abs_tol=1e-15):
        return 1.0 if math.isclose(po, 1.0) else float("nan")
    return (po - pe) / (1.0 - pe)


def confusion(y1, y2):
    m = {a: {b: 0 for b in LABELS} for a in LABELS}
    for a, b in zip(y1, y2):
        m[a][b] += 1
    return m


def token_set(start, end):
    if start is None or end is None:
        return set()
    return set(range(start, end if END_TOKEN_IS_EXCLUSIVE else end + 1))


def span_scores(s1: set[int], s2: set[int]):
    if not s1 and not s2:
        return 1.0, 1.0, True
    if not s1 or not s2:
        return 0.0, 0.0, False
    inter = len(s1 & s2)
    precision = inter / len(s2)
    recall = inter / len(s1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    iou = inter / len(s1 | s2)
    return f1, iou, s1 == s2


def rationale_metrics(aligned):
    span_rows = []
    both_neutral = 0
    neutral_vs_nonneutral = 0
    joint_iou_success = 0
    joint_exact_success = 0

    for rid, a1, a2 in aligned:
        l1, l2 = a1["label"], a2["label"]
        s1 = token_set(a1["rationale_start_token"], a1["rationale_end_token"])
        s2 = token_set(a2["rationale_start_token"], a2["rationale_end_token"])

        if l1 == "neutral" and l2 == "neutral":
            both_neutral += 1
            joint_iou_success += 1
            joint_exact_success += 1
            continue

        if (l1 == "neutral") != (l2 == "neutral"):
            neutral_vs_nonneutral += 1
            continue

        f1, iou, exact = span_scores(s1, s2)
        span_rows.append({
            "id": rid,
            "label_a1": l1,
            "label_a2": l2,
            "start_a1": a1["rationale_start_token"],
            "end_a1": a1["rationale_end_token"],
            "start_a2": a2["rationale_start_token"],
            "end_a2": a2["rationale_end_token"],
            "token_f1": f1,
            "iou": iou,
            "exact_match": exact,
        })

        if l1 == l2 and iou >= 0.50:
            joint_iou_success += 1
        if l1 == l2 and exact:
            joint_exact_success += 1

    if not span_rows:
        raise ValueError("Aucune paire de rationales non-neutres à comparer.")

    return {
        "s_span": len(span_rows),
        "both_neutral": both_neutral,
        "neutral_vs_non_neutral_disagreements": neutral_vs_nonneutral,
        "macro_token_f1": sum(r["token_f1"] for r in span_rows) / len(span_rows),
        "macro_iou": sum(r["iou"] for r in span_rows) / len(span_rows),
        "exact_match": sum(bool(r["exact_match"]) for r in span_rows) / len(span_rows),
        "joint_iou_at_0_50": joint_iou_success / len(aligned),
        "joint_exact_match": joint_exact_success / len(aligned),
        "span_rows": span_rows,
    }


def percentile(values, p):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def bootstrap_kappa_ci(y1, y2, iterations, seed):
    rng = random.Random(seed)
    n = len(y1)
    vals = []
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        value = cohen_kappa([y1[i] for i in idx], [y2[i] for i in idx])
        if not math.isnan(value):
            vals.append(value)
    return percentile(vals, 0.025), percentile(vals, 0.975)


def bootstrap_span_f1_ci(span_rows, iterations, seed):
    rng = random.Random(seed)
    n = len(span_rows)
    vals = []
    for _ in range(iterations):
        sample = [span_rows[rng.randrange(n)]["token_f1"] for _ in range(n)]
        vals.append(sum(sample) / n)
    return percentile(vals, 0.025), percentile(vals, 0.975)


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_confusion_csv(path: Path, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["A1 \\ A2", *LABELS])
        for l1 in LABELS:
            w.writerow([l1, *[matrix[l1][l2] for l2 in LABELS]])


def disagreement_rows(aligned):
    rows = []
    for rid, a1, a2 in aligned:
        s1 = token_set(a1["rationale_start_token"], a1["rationale_end_token"])
        s2 = token_set(a2["rationale_start_token"], a2["rationale_end_token"])
        f1, iou, exact = span_scores(s1, s2)
        label_agree = a1["label"] == a2["label"]
        if (not label_agree) or (not exact):
            rows.append({
                "id": rid,
                "label_a1": a1["label"],
                "label_a2": a2["label"],
                "label_agreement": label_agree,
                "rationale_start_token_a1": a1["rationale_start_token"],
                "rationale_end_token_a1": a1["rationale_end_token"],
                "rationale_start_token_a2": a2["rationale_start_token"],
                "rationale_end_token_a2": a2["rationale_end_token"],
                "token_f1": f1,
                "iou": iou,
                "exact_span_match": exact,
                "needs_adjudication": True,
            })
    return rows


def main() -> int:
    root = find_project_root()
    f1 = root / FIRST_ANNOTATOR_FILE
    f2 = root / SECOND_ANNOTATOR_FILE
    out = root / OUTPUT_DIR

    a1 = validate(load_jsonl(f1), "Annotateur 1")
    a2 = validate(load_jsonl(f2), "Annotateur 2")
    aligned = align(a1, a2)

    y1 = [x[1]["label"] for x in aligned]
    y2 = [x[2]["label"] for x in aligned]

    agreement = raw_agreement(y1, y2)
    kappa = cohen_kappa(y1, y2)
    k_lo, k_hi = bootstrap_kappa_ci(y1, y2, BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED)
    matrix = confusion(y1, y2)

    rat = rationale_metrics(aligned)
    f_lo, f_hi = bootstrap_span_f1_ci(rat["span_rows"], BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED + 1)
    disagreements = disagreement_rows(aligned)

    summary = {
        "comparison": "Annotateur 1 vs Annotateur 2 (IAA pré-adjudication)",
        "instances_aligned": len(aligned),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "span_convention": "start inclusive, end exclusive" if END_TOKEN_IS_EXCLUSIVE else "start inclusive, end inclusive",
        "label_agreement": {
            "raw_agreement": agreement,
            "cohen_kappa": kappa,
            "cohen_kappa_ci95": [k_lo, k_hi],
            "confusion_matrix_a1_rows_a2_columns": matrix,
        },
        "rationale_agreement": {
            "s_span_definition": "instances où A1 et A2 donnent tous deux un label non-neutral",
            "s_span": rat["s_span"],
            "both_neutral": rat["both_neutral"],
            "neutral_vs_non_neutral_disagreements": rat["neutral_vs_non_neutral_disagreements"],
            "macro_token_f1": rat["macro_token_f1"],
            "macro_token_f1_ci95": [f_lo, f_hi],
            "macro_iou": rat["macro_iou"],
            "exact_match": rat["exact_match"],
            "joint_iou_at_0_50": rat["joint_iou_at_0_50"],
            "joint_exact_match": rat["joint_exact_match"],
        },
        "adjudication": {
            "instances_flagged": len(disagreements),
            "rule": "labels différents OU spans non exactement identiques",
        },
    }

    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "iaa_summary.json", summary)
    write_confusion_csv(out / "label_confusion_matrix.csv", matrix)
    write_jsonl(out / "span_pair_metrics.jsonl", rat["span_rows"])
    write_jsonl(out / "pre_adjudication_disagreements.jsonl", disagreements)

    print("=" * 80)
    print("IAA PRÉ-ADJUDICATION — ANNOTATEUR 1 vs ANNOTATEUR 2")
    print("=" * 80)
    print(f"Annotateur 1            : {f1}")
    print(f"Annotateur 2            : {f2}")
    print(f"Instances alignées      : {len(aligned)}")
    print(f"Accord brut             : {agreement:.4f}")
    print(f"Cohen's kappa           : {kappa:.4f}")
    print(f"IC 95 % kappa           : [{k_lo:.4f}, {k_hi:.4f}]")
    print(f"Taille de S_span        : {rat['s_span']}")
    print(f"Macro-F1 token          : {rat['macro_token_f1']:.4f}")
    print(f"IC 95 % Macro-F1        : [{f_lo:.4f}, {f_hi:.4f}]")
    print(f"Macro-IoU               : {rat['macro_iou']:.4f}")
    print(f"Exact Match             : {rat['exact_match']:.4f}")
    print(f"Joint IoU@0.50          : {rat['joint_iou_at_0_50']:.4f}")
    print(f"Cas à adjudicer         : {len(disagreements)}")
    print(f"Dossier de sortie       : {out}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        raise SystemExit(1)
