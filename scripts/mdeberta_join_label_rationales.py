#!/usr/bin/env python3
"""FLEXID v2.1: joint supervised NLI + one extractive rationale span.

Backbone: microsoft/mdeberta-v3-base by default.
Architecture: one shared DeBERTa encoder, one standard three-way NLI head,
and one start/end span head over premise tokens only.

Python >= 3.10. Offsets are Python Unicode character indices, zero-based,
end-exclusive. No data splits are generated or modified. Gold labels/spans
never enter forward() and the official test set is evaluated only after
checkpoint selection on validation.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import sys
import unicodedata
from typing import Any, Sequence

VERSION = "FLEXID-JOINT-2.2.0-VSCODE"
LABELS = ("entailment", "contradiction", "neutral")
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
NEUTRAL = LABEL_TO_ID["neutral"]
LOG = logging.getLogger("flexid")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                   + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(path: Path, labelled: bool) -> list[dict]:
    """Read JSONL and canonicalize harmless rationale-boundary whitespace in memory.

    The official files are never rewritten. For a non-neutral rationale, if the
    stored character offsets include only leading/trailing whitespace, those
    boundaries are trimmed before token alignment. Material disagreement between
    rationale_text and the trimmed premise substring remains a hard error.
    """
    records, seen = [], set()
    trimmed_rationales = 0
    neutral_text_normalizations = 0
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            where = f"{path}:{line_number}"
            require(isinstance(row, dict), f"{where}: expected a JSON object")
            for key in ("id", "premise", "hypothesis_facts"):
                require(isinstance(row.get(key), str) and bool(row[key].strip()),
                        f"{where}: missing/empty string {key}")
            require(row["id"] not in seen, f"{where}: duplicate ID {row['id']}")
            seen.add(row["id"])

            if labelled:
                require(row.get("label") in LABELS, f"{where}: invalid label")
                start, end = row.get("rationale_start"), row.get("rationale_end")
                rationale_text = row.get("rationale_text")

                if row["label"] == "neutral":
                    empty_offsets = ((start is None and end is None)
                                     or (type(start) is int and type(end) is int and start == end == 0))
                    text_is_empty = rationale_text is None or (
                        isinstance(rationale_text, str) and not rationale_text.strip()
                    )
                    require(empty_offsets and text_is_empty,
                            f"{where}: neutral must have empty rationale: 0/0/'' or null/null")
                    if isinstance(rationale_text, str) and rationale_text != "":
                        neutral_text_normalizations += 1
                    row.update(rationale_start=0, rationale_end=0, rationale_text="")
                else:
                    require(type(start) is int and type(end) is int,
                            f"{where}: rationale offsets must be integers, not strings/floats")
                    require(0 <= start < end <= len(row["premise"]),
                            f"{where}: invalid rationale [{start}, {end})")

                    original_start, original_end = start, end
                    premise = row["premise"]
                    while start < end and premise[start].isspace():
                        start += 1
                    while end > start and premise[end - 1].isspace():
                        end -= 1
                    require(start < end, f"{where}: rationale contains only whitespace")

                    actual = premise[start:end]
                    require(bool(actual.strip()), f"{where}: empty rationale after whitespace trim")

                    if rationale_text is not None:
                        require(isinstance(rationale_text, str),
                                f"{where}: rationale_text must be a string or null")
                        # Accept the same evidence with harmless boundary whitespace only.
                        require(rationale_text.strip() == actual,
                                f"{where}: rationale_text materially differs from premise[rationale_start:rationale_end]")

                    if (start, end) != (original_start, original_end):
                        trimmed_rationales += 1
                    row.update(rationale_start=start, rationale_end=end, rationale_text=actual)
            else:
                # Never trust or use optional gold fields on the prediction path.
                row = {key: row[key] for key in ("id", "premise", "hypothesis_facts")}

            records.append(row)

    require(bool(records), f"{path}: no instances")
    if labelled and trimmed_rationales:
        LOG.warning(
            "%s: %d rationale span(s) included only boundary whitespace; "
            "trimmed in memory (source file unchanged).",
            path.name, trimmed_rationales,
        )
    if labelled and neutral_text_normalizations:
        LOG.warning(
            "%s: %d neutral rationale_text value(s) contained whitespace only; "
            "normalized in memory (source file unchanged).",
            path.name, neutral_text_normalizations,
        )
    return records


def normalized_group(text: str) -> str:
    # For leakage checks only. Never normalize text used for character offsets.
    return " ".join(unicodedata.normalize("NFC", text).split())


def check_splits(splits: dict[str, list[dict]], group_by: str = "premise") -> dict:
    ids, premises, groups = {}, {}, {}
    report = {}
    for split, rows in splits.items():
        for row in rows:
            rid = row["id"]
            require(rid not in ids, f"ID {rid} appears in both {ids.get(rid)} and {split}")
            ids[rid] = split
            premise = normalized_group(row["premise"])
            previous = premises.setdefault(premise, split)
            require(previous == split,
                    f"Premise leakage: {rid} in {split} shares a premise with {previous}")
            if group_by != "premise":
                meta = row.get("meta")
                field = "law_ref" if group_by == "law_ref" else "group_id"
                group = meta.get(field) if isinstance(meta, dict) else None
                require(isinstance(group, str) and bool(group.strip()),
                        f"{rid}: --group-by {group_by} requires meta.{field}")
                group = normalized_group(group)
                old = groups.setdefault(group, split)
                require(old == split, f"Group {group!r} occurs in {old} and {split}")
        report[split] = {"instances": len(rows),
                         "labels": dict(Counter(row["label"] for row in rows)),
                         "unique_premises": len({normalized_group(r["premise"]) for r in rows})}
    return report


def check_official(directory: Path, splits: dict[str, list[dict]]) -> None:
    sizes = {"train": 701, "validation": 151, "test": 150}
    require(set(splits) == set(sizes), "Official v3 validation requires all three splits")
    summary = json.loads((directory / "split_summary.json").read_text(encoding="utf-8-sig"))
    require(summary.get("protocol_version") == "FLEXID-EXACT-GROUP-SPLIT-v3",
            "Unexpected official split protocol")
    membership = {}
    with (directory / "split_membership.jsonl").open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                require(item.get("id") not in membership, "Duplicate membership ID")
                require(item.get("split") in sizes, "Invalid membership split")
                membership[item["id"]] = item["split"]
    actual = {}
    for split, rows in splits.items():
        require(len(rows) == sizes[split], f"{split}: expected {sizes[split]} instances")
        actual.update({row["id"]: split for row in rows})
    require(actual == membership, "Official membership does not match the input files")


@dataclass(frozen=True)
class Settings:
    max_length: int = 512
    stride: int = 128
    min_premise_tokens: int = 32
    max_windows: int = 64
    alignment_policy: str = "strict"
    max_span_tokens: int = 0  # 0 = no artificial length limit
    # Fixed a priori: rationale extraction is an auxiliary objective.
    # The span CE is first divided by log(K), where K is the number of
    # admissible premise tokens in that window.
    span_loss_weight: float = 0.05


def trim_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def prepare_record(row: dict, tokenizer: Any, settings: Settings,
                   labelled: bool) -> dict:
    require(tokenizer.is_fast, "A fast tokenizer with offset mappings is required")
    require(tokenizer.padding_side == "right", "Only right padding is supported")
    hyp_tokens = tokenizer(row["hypothesis_facts"], add_special_tokens=False)["input_ids"]
    budget = settings.max_length - len(hyp_tokens) - tokenizer.num_special_tokens_to_add(pair=True)
    require(budget >= settings.min_premise_tokens,
            f"{row['id']}: hypothesis leaves only {budget} premise tokens; "
            "shorten the hypothesis explicitly or use a model with a larger context")
    # A tokenizer stride must be smaller than the actual premise token budget.
    stride = min(settings.stride, budget - 1)
    encoded = tokenizer(row["premise"], row["hypothesis_facts"],
                        truncation="only_first", max_length=settings.max_length,
                        stride=stride, return_overflowing_tokens=True,
                        return_offsets_mapping=True, add_special_tokens=True)
    count = len(encoded["input_ids"])
    require(0 < count <= settings.max_windows,
            f"{row['id']}: {count} windows exceeds --max-windows={settings.max_windows}; "
            "no windows were silently discarded")
    windows, positives, exact_positives = [], 0, 0
    for i in range(count):
        offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"][i]]
        seq_ids = encoded.sequence_ids(i)
        mask = [sid == 0 and b > a and bool(row["premise"][a:b].strip())
                for sid, (a, b) in zip(seq_ids, offsets)]
        require(any(mask), f"{row['id']}: window without usable premise tokens")
        features = {key: encoded[key][i]
                    for key in ("input_ids", "attention_mask", "token_type_ids") if key in encoded}
        window = {"features": features, "mask": mask, "offsets": offsets,
                  "start": -100, "end": -100}
        if labelled and row["label"] != "neutral":
            a, b = row["rationale_start"], row["rationale_end"]
            overlapping = [k for k, ((x, y), ok) in enumerate(zip(offsets, mask))
                           if ok and y > a and x < b]
            if overlapping:
                first, last = overlapping[0], overlapping[-1]
                visible_a, visible_b = offsets[first][0], offsets[last][1]
                covered = (visible_a <= a and visible_b >= b)
                if covered:
                    boundary_exact = trim_bounds(row["premise"], visible_a, visible_b) == (a, b)
                    if settings.max_span_tokens:
                        require(last - first + 1 <= settings.max_span_tokens,
                                f"{row['id']}: gold span exceeds --max-span-tokens")
                    window.update(start=first, end=last)
                    positives += 1
                    exact_positives += int(boundary_exact)
                    if settings.alignment_policy == "strict":
                        require(boundary_exact,
                                f"{row['id']}: gold character boundaries are not representable "
                                "by this tokenizer. Inspect annotation or explicitly use "
                                "--alignment-policy expand (character EM may be limited).")
        windows.append(window)
    if labelled and row["label"] != "neutral":
        require(positives > 0,
                f"{row['id']}: no window fully contains the gold span; increase stride "
                "if possible. A span longer than a window requires another model/formulation.")
    return {"record": row, "windows": windows, "stride": stride,
            "positive_windows": positives, "exact_positive_windows": exact_positives,
            "label": LABEL_TO_ID[row["label"]] if labelled else None}


def prepare_records(rows: list[dict], tokenizer: Any, settings: Settings,
                    labelled: bool) -> tuple[list[dict], dict]:
    items = [prepare_record(row, tokenizer, settings, labelled) for row in rows]
    diagnostics = {"instances": len(items),
                   "windows": sum(len(x["windows"]) for x in items),
                   "multi_window_instances": sum(len(x["windows"]) > 1 for x in items),
                   "max_windows": max(len(x["windows"]) for x in items),
                   "adapted_stride_instances": sum(x["stride"] != settings.stride for x in items),
                   "expanded_boundary_instances": sum(x["positive_windows"] >
                                                       x["exact_positive_windows"] for x in items)}
    return items, diagnostics


def make_batches(items: list[dict], batch_size: int, window_budget: int,
                 seed: int | None = None) -> list[list[int]]:
    order = list(range(len(items)))
    if seed is not None:
        random.Random(seed).shuffle(order)
    batches, batch, windows = [], [], 0
    for index in order:
        count = len(items[index]["windows"])
        require(count <= window_budget,
                f"{items[index]['record']['id']}: {count} windows exceeds "
                f"--window-batch-budget={window_budget}; increase the budget if memory permits")
        if batch and (len(batch) >= batch_size or windows + count > window_budget):
            batches.append(batch)
            batch, windows = [], 0
        batch.append(index)
        windows += count
    if batch:
        batches.append(batch)
    return batches


class Collator:
    def __init__(self, tokenizer: Any, torch: Any):
        self.tokenizer, self.torch = tokenizer, torch

    def __call__(self, items: list[dict]) -> dict:
        torch = self.torch
        windows = [w for item in items for w in item["windows"]]
        features = self.tokenizer.pad([w["features"] for w in windows],
                                      padding=True, return_tensors="pt")
        length = features["input_ids"].shape[1]
        features["premise_mask"] = torch.tensor(
            [w["mask"] + [False] * (length - len(w["mask"])) for w in windows], dtype=torch.bool)
        sizes = [len(item["windows"]) for item in items]
        targets = None
        if all(item["label"] is not None for item in items):
            targets = {"labels": torch.tensor([item["label"] for item in items]),
                       "start": torch.tensor([w["start"] for w in windows]),
                       "end": torch.tensor([w["end"] for w in windows])}
        return {"features": features, "sizes": sizes, "targets": targets, "items": items}


def dependencies() -> Any:
    from types import SimpleNamespace
    try:
        import numpy as np
        import torch
        import transformers
        from safetensors.torch import load_file, save_file
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:
        raise RuntimeError("Install dependencies from requirements.txt first") from exc
    return SimpleNamespace(torch=torch, np=np, transformers=transformers,
                           AutoConfig=AutoConfig,
                           AutoModelForSequenceClassification=AutoModelForSequenceClassification,
                           AutoTokenizer=AutoTokenizer, load_file=load_file, save_file=save_file,
                           scheduler=get_linear_schedule_with_warmup)


def build_model_class(torch: Any) -> Any:
    class JointModel(torch.nn.Module):
        def __init__(self, classifier: Any):
            super().__init__()
            require(classifier.config.model_type == "deberta-v2",
                    "This implementation supports the DeBERTa-v2/v3 family only")
            for attr in ("pooler", "dropout", "classifier"):
                require(hasattr(classifier, attr), f"Missing DeBERTa classifier attribute {attr}")
            self.sequence_classifier = classifier
            self.span_head = torch.nn.Linear(classifier.config.hidden_size, 2)
            self.span_dropout = torch.nn.Dropout(classifier.config.hidden_dropout_prob)
            torch.nn.init.normal_(self.span_head.weight, std=classifier.config.initializer_range)
            torch.nn.init.zeros_(self.span_head.bias)

        def forward(self, *, sizes: list[int], premise_mask: Any, **features: Any) -> dict:
            # Deliberately no gold label/span/target argument in this interface.
            sequence = self.sequence_classifier.base_model(
                **features, return_dict=True).last_hidden_state
            pooled = self.sequence_classifier.dropout(self.sequence_classifier.pooler(sequence))
            local_labels = self.sequence_classifier.classifier(pooled).float()

            # Equal-weight aggregation across all windows of one instance.
            # This introduces no third auxiliary task and no gold-dependent gate.
            labels = [local.mean(dim=0) for local in local_labels.split(sizes)]

            span = self.span_head(self.span_dropout(sequence)).float()
            invalid = ~premise_mask.bool()
            start = span[..., 0].masked_fill(invalid, -1e4)
            end = span[..., 1].masked_fill(invalid, -1e4)
            return {"label_logits": torch.stack(labels), "start_logits": start,
                    "end_logits": end}

    return JointModel


def joint_loss(outputs: dict, targets: dict, sizes: list[int], premise_mask: Any,
               settings: Settings, torch: Any) -> dict:
    fn = torch.nn.functional
    label = fn.cross_entropy(outputs["label_logits"], targets["labels"], reduction="none")
    starts, ends = targets["start"], targets["end"]
    positive = starts.ne(-100) & ends.ne(-100)

    # Raw start/end CE for every window. Gold-neutral windows and non-containing
    # overflow windows use ignore_index=-100 and therefore contribute zero.
    per_span_raw = 0.5 * (
        fn.cross_entropy(outputs["start_logits"], starts,
                         reduction="none", ignore_index=-100)
        + fn.cross_entropy(outputs["end_logits"], ends,
                           reduction="none", ignore_index=-100)
    )

    # Task-scale normalization fixed from the output-space cardinality:
    # a uniform random K-way span predictor has CE ~= log(K). Dividing by
    # log(K) keeps the auxiliary span objective on a stable, interpretable scale.
    valid_counts = premise_mask.bool().sum(dim=1).clamp(min=2).float()
    per_span_normalized = per_span_raw / valid_counts.log()

    raw_spans, normalized_spans = [], []
    for raw, normalized, pos in zip(per_span_raw.split(sizes),
                                    per_span_normalized.split(sizes),
                                    positive.split(sizes)):
        if bool(pos.any()):
            raw_spans.append(raw[pos].mean())
            normalized_spans.append(normalized[pos].mean())
        else:
            zero = raw.sum() * 0.0
            raw_spans.append(zero)
            normalized_spans.append(zero)

    span_raw = torch.stack(raw_spans)
    span_normalized = torch.stack(normalized_spans)
    total = label + settings.span_loss_weight * span_normalized

    return {
        "loss": total.mean(),
        "label_loss": label.mean(),
        "span_loss_raw": span_raw.mean(),
        "span_loss_normalized": span_normalized.mean(),
        "weighted_span_loss": (settings.span_loss_weight * span_normalized).mean(),
    }


def best_span(start: Sequence[float], end: Sequence[float], mask: Sequence[bool],
              max_tokens: int = 0) -> tuple[int, int, float]:
    """Exact O(L) argmax of start[s]+end[e], s<=e, with optional length limit."""
    candidates: deque[int] = deque()
    best, pair = -math.inf, None
    for e, ok in enumerate(mask):
        if not ok:
            continue
        if max_tokens:
            while candidates and candidates[0] < e - max_tokens + 1:
                candidates.popleft()
        while candidates and start[candidates[-1]] < start[e]:
            candidates.pop()
        candidates.append(e)
        s = candidates[0]
        value = float(start[s]) + float(end[e])
        if value > best:
            best, pair = value, (s, e)
    require(pair is not None and math.isfinite(best), "No finite valid span could be decoded")
    return pair[0], pair[1], best


def decode_instance(item: dict, label_logits: Any, start: Any, end: Any,
                    settings: Settings, torch: Any) -> dict:
    probabilities = label_logits.float().softmax(-1).tolist()
    predicted_id = max(range(len(LABELS)), key=lambda i: probabilities[i])
    row = item["record"]
    result = {"id": row["id"], "label": LABELS[predicted_id],
              "label_probabilities": dict(zip(LABELS, probabilities)),
              "rationale_start": 0, "rationale_end": 0, "rationale_text": "",
              "window_count": len(item["windows"]), "selected_window": None,
              "rationale_score": None}
    if predicted_id == NEUTRAL:
        return result

    selected, best_score = None, -math.inf
    for i, window in enumerate(item["windows"]):
        length = len(window["mask"])
        mask = torch.tensor(window["mask"], dtype=torch.bool)
        # Softmax over the premise alone, excluding hypothesis, special tokens and padding.
        log_start = start[i, :length].float().masked_fill(~mask, -math.inf).log_softmax(-1)
        log_end = end[i, :length].float().masked_fill(~mask, -math.inf).log_softmax(-1)
        s, e, span_score = best_span(log_start.tolist(), log_end.tolist(),
                                    window["mask"], settings.max_span_tokens)
        if span_score > best_score:
            best_score, selected = span_score, (i, s, e)

    require(selected is not None, f"{row['id']}: no rationale candidate")
    i, s, e = selected
    offsets = item["windows"][i]["offsets"]
    a, b = trim_bounds(row["premise"], offsets[s][0], offsets[e][1])
    require(0 <= a < b <= len(row["premise"]), f"{row['id']}: invalid decoded character span")
    result.update(rationale_start=a, rationale_end=b, rationale_text=row["premise"][a:b],
                  selected_window=i, rationale_score=best_score)
    return result


def overlap_scores(gold: set[int], predicted: set[int]) -> tuple[float, float]:
    if not gold and not predicted:
        return 1.0, 1.0
    overlap = len(gold & predicted)
    denom = len(gold) + len(predicted)
    return (2 * overlap / denom if denom else 0.0,
            overlap / len(gold | predicted) if gold | predicted else 0.0)


def token_indices(text: str, start: int, end: int) -> set[int]:
    return {i for i, match in enumerate(re.finditer(r"\S+", text))
            if match.end() > start and match.start() < end} if end > start else set()


def average(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def score_records(gold_rows: list[dict], predictions: list[dict]) -> dict:
    require(len(gold_rows) == len(predictions), "Prediction count mismatch")
    matrix = {a: {b: 0 for b in LABELS} for a in LABELS}
    nonneutral, conditional, exact_all, iou_all, evidence_free = [], [], [], [], []

    for gold, pred in zip(gold_rows, predictions):
        require(gold["id"] == pred["id"], "Prediction ID/order mismatch")
        g, p = gold["label"], pred["label"]
        matrix[g][p] += 1
        a, b = gold["rationale_start"], gold["rationale_end"]
        x, y = pred["rationale_start"], pred["rationale_end"]
        require(pred["rationale_text"] == gold["premise"][x:y], "Inconsistent prediction text")
        require((p == "neutral" and x == y == 0)
                or (p != "neutral" and 0 <= x < y <= len(gold["premise"])),
                "Prediction label and rationale are inconsistent")

        tf1, iou = overlap_scores(token_indices(gold["premise"], a, b),
                                  token_indices(gold["premise"], x, y))
        overlap = max(0, min(b, y) - max(a, x))
        cf1 = 2 * overlap / ((b - a) + (y - x)) if (b - a) + (y - x) else 1.0
        exact = (a, b) == (x, y)
        correct = g == p

        # End-to-end joint success over all instances. Correct neutral predictions
        # have empty gold/predicted spans, hence exact=True and IoU=1.
        exact_all.append(float(correct and exact))
        iou_all.append(float(correct and iou >= 0.5))

        if g == "neutral":
            evidence_free.append(float(p == "neutral"))
        else:
            record = {"token_f1": tf1, "char_f1": cf1, "token_iou": iou,
                      "char_exact": float(exact), "label_correct": float(correct)}
            nonneutral.append(record)
            # Main rationale metrics follow the paper protocol: score spans only
            # when gold and prediction share the same non-neutral label.
            if correct:
                conditional.append(record)

    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        guessed = sum(matrix[g][label] for g in LABELS)
        precision = tp / guessed if guessed else 0.0
        recall = tp / support if support else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * tp / (support + guessed) if support + guessed else 0.0,
            "support": support,
        }

    n = len(predictions)
    accuracy = sum(matrix[x][x] for x in LABELS) / n
    macro = statistics.mean(per_class[label]["f1"] for label in LABELS)

    # Cohen's kappa from the 3x3 confusion matrix.
    gold_marginals = [sum(matrix[g].values()) for g in LABELS]
    pred_marginals = [sum(matrix[g][p] for g in LABELS) for p in LABELS]
    expected = sum(g * p for g, p in zip(gold_marginals, pred_marginals)) / (n * n)
    kappa = ((accuracy - expected) / (1.0 - expected)) if expected < 1.0 else 0.0

    e2e_f1 = average([r["token_f1"] * r["label_correct"] for r in nonneutral])
    return {
        "n": n,
        "accuracy": accuracy,
        "macro_f1": macro,
        "cohen_kappa": kappa,
        "per_class": per_class,
        "confusion_matrix": matrix,

        # Rationale metrics comparable with the DeepSeek / human protocol.
        "nonneutral_count": len(nonneutral),
        "conditional_span_count": len(conditional),
        "rationale_s_span": len(conditional),
        "conditional_token_f1": average([r["token_f1"] for r in conditional]),
        "conditional_token_iou": average([r["token_iou"] for r in conditional]),
        "conditional_exact_match": average([r["char_exact"] for r in conditional]),
        "no_rationale_agreement": average(evidence_free),

        # End-to-end diagnostics over gold non-neutral instances.
        "e2e_nonneutral_token_f1": e2e_f1,
        "e2e_nonneutral_char_f1":
            average([r["char_f1"] * r["label_correct"] for r in nonneutral]),
        "e2e_nonneutral_char_exact":
            average([r["char_exact"] * r["label_correct"] for r in nonneutral]),

        # Joint metrics over all instances.
        "joint_exact_match": average(exact_all),
        "joint_token_iou_at_0_50": average(iou_all),

        # Backward-compatible aliases.
        "neutral_recall": average(evidence_free),
        "joint_char_exact": average(exact_all),
    }


def device_and_precision(args: Any, torch: Any) -> tuple[Any, str]:
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    require(device.type in ("cpu", "cuda"), "Supported devices: cpu, cuda, cuda:N")
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
        torch.cuda.set_device(device)
    precision = args.precision
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    require(precision == "fp32" or device.type == "cuda", "Mixed precision requires CUDA here")
    if precision == "bf16":
        require(torch.cuda.is_bf16_supported(), "This CUDA device does not support bf16")
    return device, precision


def autocast_context(torch: Any, device: Any, precision: str) -> Any:
    return (nullcontext() if precision == "fp32" else torch.autocast(
        device_type=device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16))


def model_inputs(batch: dict, device: Any) -> dict:
    return {"sizes": batch["sizes"], **{k: v.to(device) for k, v in batch["features"].items()}}


def predict_items(model: Any, items: list[dict], tokenizer: Any, settings: Settings,
                  args: Any, deps: Any, device: Any, precision: str) -> list[dict]:
    torch = deps.torch
    batches = make_batches(items, args.eval_batch_size, args.window_batch_budget)
    loader = torch.utils.data.DataLoader(items, batch_sampler=batches,
                                         collate_fn=Collator(tokenizer, torch), num_workers=0)
    model.eval()
    results = []
    with torch.inference_mode():
        for batch in loader:
            with autocast_context(torch, device, precision):
                outputs = model(**model_inputs(batch, device))
            outputs = {k: v.detach().float().cpu() for k, v in outputs.items()}
            require(all(bool(torch.isfinite(v).all()) for v in outputs.values()),
                    "Non-finite model output; retry fp32 and inspect the checkpoint")
            start = 0
            for i, (item, size) in enumerate(zip(batch["items"], batch["sizes"])):
                end = start + size
                results.append(decode_instance(item, outputs["label_logits"][i],
                    outputs["start_logits"][start:end], outputs["end_logits"][start:end],
                    settings, torch))
                start = end
    return results


def save_checkpoint(directory: Path, model: Any, tokenizer: Any, settings: Settings,
                    metadata: dict, deps: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / "model.safetensors"
    # Atomic weight replacement: an interrupted save preserves the previous weights.
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    deps.save_file(state, str(weights.with_suffix(".tmp")), metadata={"format": "pt", "version": VERSION})
    del state
    os.replace(weights.with_suffix(".tmp"), weights)
    if not (directory / "config.json").is_file():
        model.sequence_classifier.config.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        write_json(directory / "joint_config.json", {"version": VERSION,
                   "settings": asdict(settings), "labels": list(LABELS)})
    write_json(directory / "checkpoint_info.json", metadata)


def load_checkpoint(directory: Path, deps: Any) -> tuple[Any, Any, Settings]:
    manifest = json.loads((directory / "joint_config.json").read_text(encoding="utf-8"))
    require(manifest.get("version") == VERSION and manifest.get("labels") == list(LABELS),
            "Incompatible joint checkpoint version/label mapping")
    tokenizer = deps.AutoTokenizer.from_pretrained(directory, use_fast=True, local_files_only=True)
    tokenizer.padding_side = "right"
    config = deps.AutoConfig.from_pretrained(directory, local_files_only=True)
    require(config.num_labels == len(LABELS), "Invalid checkpoint class count")
    classifier = deps.AutoModelForSequenceClassification.from_config(config)
    model = build_model_class(deps.torch)(classifier)
    model.load_state_dict(deps.load_file(str(directory / "model.safetensors"), device="cpu"), strict=True)
    model = _force_fp32_master_weights(model, deps.torch)
    return model, tokenizer, Settings(**manifest["settings"])


def seed_everything(seed: int, deps: Any, deterministic: bool) -> None:
    random.seed(seed)
    deps.np.random.seed(seed)
    torch = deps.torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False


def _force_fp32_master_weights(model: Any, torch: Any) -> Any:
    """Keep one consistent master dtype for training.

    Some Hugging Face checkpoints are stored in float16 while newly created
    downstream heads are float32. On CPU (and on CUDA when autocast is off),
    that can produce `mat1 and mat2 must have the same dtype`. Converting the
    complete model to fp32 once after loading removes that checkpoint-format
    dependency. CUDA mixed precision, when requested, is still handled by
    autocast during forward/backward.
    """
    model = model.float()
    bad = []
    for name, parameter in model.named_parameters():
        if parameter.is_floating_point() and parameter.dtype != torch.float32:
            bad.append((name, str(parameter.dtype)))
    for name, buffer in model.named_buffers():
        if buffer.is_floating_point() and buffer.dtype != torch.float32:
            bad.append((f"buffer:{name}", str(buffer.dtype)))
    require(not bad, f"Mixed floating dtypes remain after fp32 normalization: {bad[:8]}")
    return model


def new_training_model(args: Any, revision: str | None, deps: Any) -> Any:
    config = deps.AutoConfig.from_pretrained(args.model, revision=revision)
    config.num_labels = len(LABELS)
    config.id2label = dict(enumerate(LABELS))
    config.label2id = LABEL_TO_ID.copy()
    config.problem_type = "single_label_classification"
    classifier = deps.AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=revision, config=config, ignore_mismatched_sizes=True)
    model = build_model_class(deps.torch)(classifier)
    # Always initialize the NLI output head: an arbitrary source checkpoint may
    # use another label order even if it also has three output neurons.
    deps.torch.nn.init.normal_(classifier.classifier.weight, std=config.initializer_range)
    deps.torch.nn.init.zeros_(classifier.classifier.bias)
    # mDeBERTa checkpoints may be stored in fp16 whereas the new NLI/span heads
    # are created in fp32. Normalize all master weights BEFORE optimizer setup.
    model = _force_fp32_master_weights(model, deps.torch)
    if args.gradient_checkpointing:
        classifier.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def make_optimizer(model: Any, args: Any, torch: Any) -> Any:
    encoder_ids = {id(p) for p in model.sequence_classifier.base_model.parameters()}
    groups: dict[tuple[float, float], list] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            lr = args.encoder_lr if id(parameter) in encoder_ids else args.head_lr
            decay = 0.0 if parameter.ndim < 2 or name.endswith("bias") else args.weight_decay
            groups.setdefault((lr, decay), []).append(parameter)
    return torch.optim.AdamW([{"params": values, "lr": lr, "weight_decay": decay}
                              for (lr, decay), values in groups.items()])


def selection_key(metrics: dict) -> tuple[float, float, float]:
    """Validation-only checkpoint rule fixed before test evaluation.

    Primary: three-way NLI Macro-F1.
    Tie-break 1: end-to-end Joint IoU@0.50.
    Tie-break 2: conditional rationale Token-F1.
    """
    rationale_f1 = metrics["conditional_token_f1"]
    return (metrics["macro_f1"],
            metrics["joint_token_iou_at_0_50"],
            -1.0 if rationale_f1 is None else rationale_f1)


def train_seed(seed: int, args: Any, items: dict, tokenizer: Any, settings: Settings,
               deps: Any, revision: str | None, device: Any, precision: str) -> dict:
    torch = deps.torch
    seed_everything(seed, deps, args.deterministic)
    directory = args.output / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    model = new_training_model(args, revision, deps).to(device)
    LOG.info("seed=%s model master dtype=float32; runtime precision=%s; device=%s",
             seed, precision, device)
    optimizer = make_optimizer(model, args, torch)
    epoch_batches = [make_batches(items["train"], args.batch_size, args.window_batch_budget,
                                  seed + epoch) for epoch in range(args.epochs)]
    total_steps = sum(math.ceil(len(batches) / args.grad_accum) for batches in epoch_batches)
    scheduler = deps.scheduler(optimizer, num_warmup_steps=round(total_steps * args.warmup_ratio),
                               num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    best_key, best_epoch, stale, history, step = None, None, 0, [], 0
    checkpoint = directory / "best"
    for epoch, batches in enumerate(epoch_batches, 1):
        loader = torch.utils.data.DataLoader(items["train"], batch_sampler=batches,
            collate_fn=Collator(tokenizer, torch), num_workers=0, pin_memory=device.type == "cuda")
        iterator = iter(loader)
        model.train()
        totals = Counter()
        seen = 0
        while True:
            microbatches = list(itertools.islice(iterator, args.grad_accum))
            if not microbatches:
                break
            # Exact instance denominator for this optimizer step, including a short
            # final accumulation group and unequal dynamic microbatch sizes.
            count = sum(len(batch["items"]) for batch in microbatches)
            optimizer.zero_grad(set_to_none=True)
            for batch in microbatches:
                size = len(batch["items"])
                targets = {k: v.to(device) for k, v in batch["targets"].items()}
                with autocast_context(torch, device, precision):
                    outputs = model(**model_inputs(batch, device))
                    premise_mask = batch["features"]["premise_mask"].to(device)
                    losses = joint_loss(outputs, targets, batch["sizes"], premise_mask,
                                        settings, torch)
                    loss = losses["loss"] * (size / count)
                require(bool(torch.isfinite(loss)), f"Non-finite loss at epoch {epoch}; try --precision fp32")
                scaler.scale(loss).backward()
                for key, value in losses.items():
                    totals[key] += float(value.detach()) * size
                seen += size
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm,
                                                  error_if_nonfinite=not scaler.is_enabled())
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
                step += 1
            else:
                LOG.warning("FP16 overflow: optimizer step skipped; gradient norm=%s", norm)
            if step and step % args.log_every == 0:
                LOG.info("seed=%s epoch=%s step=%s/%s loss=%.4f", seed, epoch, step,
                         total_steps, totals["loss"] / seen)
        predictions = predict_items(model, items["validation"], tokenizer, settings,
                                     args, deps, device, precision)
        metrics = score_records([x["record"] for x in items["validation"]], predictions)
        key = selection_key(metrics)
        entry = {"epoch": epoch, "optimizer_steps": step,
                 "train_losses": {k: v / seen for k, v in totals.items()},
                 "validation": metrics, "learning_rates": [g["lr"] for g in optimizer.param_groups]}
        history.append(entry)
        write_json(directory / "history.json", history)
        LOG.info(
            "seed=%s epoch=%s val Macro-F1=%.4f Joint@.50=%.4f rationale-F1=%s",
            seed, epoch, metrics["macro_f1"], metrics["joint_token_iou_at_0_50"],
            "NA" if metrics["conditional_token_f1"] is None
            else f"{metrics['conditional_token_f1']:.4f}")
        # Deterministic lexicographic tie breaking. Strict improvement only.
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            save_checkpoint(
                checkpoint, model, tokenizer, settings,
                {"seed": seed, "epoch": epoch,
                 "selection": "validation_macro_f1_then_joint_iou_at_0_50",
                 "validation": metrics}, deps)
            write_jsonl(directory / "validation_predictions.jsonl", predictions)
        else:
            stale += 1
            if stale >= args.patience:
                LOG.info("Early stopping seed=%s; best epoch=%s", seed, best_epoch)
                break
    require(best_epoch is not None, "No checkpoint was produced")
    best_metrics = history[best_epoch - 1]["validation"]
    result = {"seed": seed, "best_epoch": best_epoch, "validation": best_metrics,
              "checkpoint": str(checkpoint.resolve())}
    write_json(directory / "result.json", result)
    del model, optimizer, scheduler, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def load_splits(args: Any) -> tuple[dict, dict]:
    paths = {"train": args.train, "validation": args.validation}
    if args.test:
        paths["test"] = args.test
    splits = {name: read_records(path, labelled=True) for name, path in paths.items()}
    report = check_splits(splits, args.group_by)
    if args.official_split_dir:
        # The released v3 protocol groups by exact canonical law_ref OR exact
        # premise. Re-check law_ref explicitly even when the CLI default is premise.
        law_report = check_splits(splits, "law_ref")
        report["official_law_ref_check"] = {
            split: law_report[split] for split in law_report
        }
        check_official(args.official_split_dir, splits)
    return splits, report


def run_train(args: Any) -> None:
    splits, report = load_splits(args)
    require(all(any(r["label"] == label for r in splits["train"]) for label in LABELS),
            "Training requires at least one instance of each NLI label")
    require(any(r["label"] != "neutral" for r in splits["validation"]),
            "Validation needs non-neutral examples for rationale tie-breaking")
    require(not args.output.exists() or not any(args.output.iterdir()),
            f"Output directory {args.output} is not empty; choose a new run directory")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    deps = dependencies()
    device, precision = device_and_precision(args, deps.torch)
    source_config = deps.AutoConfig.from_pretrained(args.model, revision=args.revision)
    require(source_config.model_type == "deberta-v2", "Expected a DeBERTa-v2/v3 model")
    revision = getattr(source_config, "_commit_hash", None) or args.revision
    tokenizer = deps.AutoTokenizer.from_pretrained(args.model, revision=revision, use_fast=True)
    tokenizer.padding_side = "right"
    require(tokenizer.pad_token_id is not None, "Tokenizer has no padding token")
    settings = Settings(**{name: getattr(args, name) for name in Settings.__dataclass_fields__})
    require(settings.span_loss_weight >= 0.0, "--span-loss-weight must be non-negative")
    require(settings.max_length <= source_config.max_position_embeddings,
            "--max-length exceeds the model's configured maximum context")
    items, diagnostics = {}, {}
    for name, rows in splits.items():
        # Test gold spans never affect tokenization checks, window supervision, or
        # configuration choices. An unrepresentable test span remains an honest error.
        items[name], diagnostics[name] = prepare_records(rows, tokenizer, settings,
                                                        labelled=name != "test")
        # Fail before costly training if any instance cannot fit in the window batch budget.
        make_batches(items[name], args.batch_size, args.window_batch_budget)
        LOG.info("%s: %s", name, diagnostics[name])
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest = {"version": VERSION, "arguments": config, "settings": asdict(settings),
                "source_model": args.model, "resolved_revision": revision,
                "source_script_sha256": file_hash(Path(__file__)),
                "split_report": report, "tokenization": diagnostics,
                "data_sha256": {name: file_hash(getattr(args, name)) for name in splits},
                "environment": {"python": platform.python_version(), "torch": deps.torch.__version__,
                                "transformers": deps.transformers.__version__, "numpy": deps.np.__version__,
                                "cuda": deps.torch.version.cuda, "device": str(device), "precision": precision,
                                "master_parameter_dtype": "float32"}}
    write_json(args.output / "run_manifest.json", manifest)
    results = [train_seed(seed, args, items, tokenizer, settings, deps, revision, device, precision)
               for seed in args.seeds]
    # Every seed's checkpoint is frozen using validation BEFORE any test inference.
    if "test" in items:
        for result in results:
            model, saved_tokenizer, saved_settings = load_checkpoint(Path(result["checkpoint"]), deps)
            model.to(device)
            predictions = predict_items(model, items["test"], saved_tokenizer, saved_settings,
                                         args, deps, device, precision)
            result["test"] = score_records(splits["test"], predictions)
            directory = Path(result["checkpoint"]).parent
            write_jsonl(directory / "test_predictions.jsonl", predictions)
            write_json(directory / "result.json", result)
            del model
            if device.type == "cuda":
                deps.torch.cuda.empty_cache()
    aggregate = {}
    aggregate_metrics = (
        "accuracy", "macro_f1", "cohen_kappa",
        "conditional_token_f1", "conditional_token_iou", "conditional_exact_match",
        "no_rationale_agreement", "joint_token_iou_at_0_50", "joint_exact_match",
        "e2e_nonneutral_token_f1", "e2e_nonneutral_char_exact",
    )
    for split in ("validation", "test"):
        if all(split in result for result in results):
            aggregate[split] = {}
            for metric in aggregate_metrics:
                values = [result[split][metric] for result in results]
                aggregate[split][metric] = {
                    "values": values,
                    "mean": average(values) if all(v is not None for v in values) else None,
                    "std": (statistics.stdev(values)
                            if len(values) > 1 and all(v is not None for v in values)
                            else None),
                }
            aggregate[split]["rationale_s_span"] = {
                "values": [result[split]["rationale_s_span"] for result in results]
            }
    write_json(args.output / "summary.json", {"runs": results, "aggregate": aggregate})
    LOG.info("Training complete. Checkpoints kept in %s/seed_*/best", args.output)


def run_prediction(args: Any, labelled: bool) -> None:
    for output in (args.output, getattr(args, "metrics", None)):
        if output is not None:
            require(output.resolve() != args.input.resolve(), "Output must not overwrite input data")
            require(not output.exists(), f"Output already exists: {output}; choose a new filename")
    if labelled:
        require(args.metrics.resolve() != args.output.resolve(), "Metrics and predictions need different paths")
    records = read_records(args.input, labelled=labelled)
    deps = dependencies()
    device, precision = device_and_precision(args, deps.torch)
    model, tokenizer, settings = load_checkpoint(args.checkpoint, deps)
    # Even evaluation uses inference preprocessing: annotations cannot alter windows
    # or gate an output. Unrepresentable gold spans are honestly reflected in metrics.
    items, diagnostics = prepare_records(records, tokenizer, settings, labelled=False)
    LOG.info("Input: %s", diagnostics)
    model.to(device)
    predictions = predict_items(model, items, tokenizer, settings, args, deps, device, precision)
    write_jsonl(args.output, predictions)
    if labelled:
        metrics = score_records(records, predictions)
        write_json(args.metrics, metrics)
        LOG.info(
            "Macro-F1=%.4f; kappa=%.4f; rationale-F1=%s; Joint@.50=%.4f",
            metrics["macro_f1"], metrics["cohen_kappa"],
            "NA" if metrics["conditional_token_f1"] is None
            else f"{metrics['conditional_token_f1']:.4f}",
            metrics["joint_token_iou_at_0_50"])
    LOG.info("Predictions written to %s", args.output)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and >= 0")
    return number


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    main.add_argument("--debug", action="store_true", help="Show tracebacks on errors")
    commands = main.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate JSONL, labels, offsets and split leakage (no ML dependencies)")
    train = commands.add_parser("train", help="Train, select on validation, then optionally evaluate test")
    predict = commands.add_parser("predict", help="Predict labels and rationale spans without gold annotations")
    evaluate = commands.add_parser("evaluate", help="Evaluate a saved checkpoint on labelled JSONL")
    for command in (validate, train):
        command.add_argument("--train", type=Path, required=True)
        command.add_argument("--validation", type=Path, required=True)
        command.add_argument("--test", type=Path)
        command.add_argument("--group-by", choices=("premise", "law_ref", "group_id"), default="premise")
        command.add_argument("--official-split-dir", type=Path,
                             help="Also enforce original FLEXID v3 sizes, protocol and membership")
    for command in (train, predict, evaluate):
        command.add_argument("--output", type=Path, required=True,
                             help="New training directory, or new predictions JSONL for predict/evaluate")
        command.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
        command.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
        command.add_argument("--eval-batch-size", type=positive_int, default=4)
        command.add_argument("--window-batch-budget", type=positive_int, default=8)
    train.add_argument("--model", default="microsoft/mdeberta-v3-base")
    train.add_argument("--revision", help="Prefer a Hugging Face commit hash; resolved hash is recorded")
    train.add_argument("--seeds", type=nonnegative_int, nargs="+", default=[2026, 2027, 2028])
    train.add_argument("--epochs", type=positive_int, default=8)
    train.add_argument("--patience", type=positive_int, default=3)
    train.add_argument("--batch-size", type=positive_int, default=2, help="Instances per microbatch, not windows")
    train.add_argument("--grad-accum", type=positive_int, default=8)
    train.add_argument("--encoder-lr", type=nonnegative_float, default=2e-5)
    train.add_argument("--head-lr", type=nonnegative_float, default=5e-5)
    train.add_argument("--weight-decay", type=nonnegative_float, default=0.01)
    train.add_argument("--warmup-ratio", type=nonnegative_float, default=0.1)
    train.add_argument("--max-grad-norm", type=nonnegative_float, default=1.0)
    train.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--log-every", type=positive_int, default=10)
    train.add_argument("--max-length", type=positive_int, default=512)
    train.add_argument("--stride", type=nonnegative_int, default=128)
    train.add_argument("--min-premise-tokens", type=positive_int, default=32)
    train.add_argument("--max-windows", type=positive_int, default=64)
    train.add_argument("--alignment-policy", choices=("strict", "expand"), default="expand")
    train.add_argument("--max-span-tokens", type=nonnegative_int, default=0)
    train.add_argument(
        "--span-loss-weight", type=nonnegative_float, default=0.05,
        help="Weight applied after span CE normalization by log(valid premise tokens)")
    for command in (predict, evaluate):
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--metrics", type=Path, required=True)
    return main


def _default_vscode_argv() -> list[str]:
    """Arguments used when the file is launched with VS Code's Run button.

    The function searches for the FLEXID project root rather than assuming the
    current working directory, so ``Run Python File`` works whether VS Code
    starts the terminal from the repository root or from ``scripts/``.
    Explicit command-line arguments always take precedence over these defaults.
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [Path.cwd().resolve(), script_dir, *script_dir.parents]
    project_root = None
    for candidate in candidates:
        split_dir = candidate / "data" / "flexid_exact_group_split"
        if (split_dir / "train.jsonl").is_file() and (split_dir / "validation.jsonl").is_file():
            project_root = candidate
            break
    if project_root is None:
        raise RuntimeError(
            "Impossible de trouver data/flexid_exact_group_split. "
            "Ouvre le dépôt FLEXID comme dossier VS Code ou lance le script depuis ce dépôt."
        )

    split_dir = project_root / "data" / "flexid_exact_group_split"
    test_file = split_dir / "test.jsonl"
    require(test_file.is_file(), f"Fichier test officiel absent: {test_file}")

    # VS Code Run should never fail just because an earlier/incomplete run folder
    # already exists. Keep every experiment separate instead of overwriting it.
    base_output = project_root / "data" / "results_mdeberta_joint_final"
    output_dir = base_output
    suffix = 2
    while output_dir.exists() and any(output_dir.iterdir()):
        output_dir = base_output.with_name(f"{base_output.name}_run{suffix:02d}")
        suffix += 1

    return [
        "train",
        "--train", str(split_dir / "train.jsonl"),
        "--validation", str(split_dir / "validation.jsonl"),
        "--test", str(test_file),
        "--official-split-dir", str(split_dir),
        "--output", str(output_dir),
        "--alignment-policy", "expand",
    ]


def main(argv: list[str] | None = None) -> int:
    # VS Code's "Run Python File" button supplies no CLI arguments. In that
    # case, run the frozen official FLEXID experiment directly. The full CLI is
    # preserved for validate/predict/evaluate and for reproducible scripted runs.
    if argv is None and len(sys.argv) == 1:
        argv = _default_vscode_argv()
        print("[FLEXID] Aucun argument CLI détecté (bouton Run VS Code).")
        print("[FLEXID] Lancement automatique du protocole officiel mDeBERTa joint.")
        print("[FLEXID] Tout est configuré dans le script: aucun argument manuel requis.")
        print("[FLEXID] Commande implicite:", " ".join(argv))

    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command == "validate":
            _, report = load_splits(args)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.command == "train":
            require(len(set(args.seeds)) == len(args.seeds), "Seeds must be unique")
            require(all(seed < 2**32 for seed in args.seeds), "Seeds must be smaller than 2**32")
            require(args.warmup_ratio < 1, "--warmup-ratio must be < 1")
            require(args.encoder_lr > 0 and args.head_lr > 0 and args.max_grad_norm > 0,
                    "Learning rates and max gradient norm must be positive")
            run_train(args)
        else:
            run_prediction(args, labelled=args.command == "evaluate")
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as exc:
        if args.debug:
            raise
        LOG.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
