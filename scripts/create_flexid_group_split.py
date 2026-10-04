#!/usr/bin/env python3


from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_RELATIVE_PATH = Path("data") / "flexid_shuffled.jsonl"
OUTPUT_RELATIVE_DIRECTORY = Path("data") / "flexid_exact_group_split"

EXPECTED_INSTANCE_COUNT = 1002

LABELS = (
    "entailment",
    "contradiction",
    "neutral",
)

SPLIT_NAMES = (
    "train",
    "validation",
    "test",
)

SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

SPLIT_SEED = 2026

# Identifiant écrit dans le terminal et dans split_summary.json.
PROTOCOL_VERSION = "FLEXID-EXACT-GROUP-SPLIT-v3"

# Précondition méthodologique :
# les références juridiques de flexid_shuffled.jsonl ont été canonisées en amont.
# Ce script ne modifie ni ne normalise law_ref ; il utilise leur égalité exacte
# (ainsi que l'égalité exacte des prémisses) pour construire les composantes
# group-disjoint. Il conserve aussi l'ordre shufflé de la source dans chaque split.

# Recherche déterministe de la meilleure allocation groupée.
GREEDY_RESTARTS = 500
LOCAL_SEARCH_CANDIDATES = 12
LOCAL_SEARCH_ATTEMPTS = 8000


# ============================================================================
# STRUCTURES
# ============================================================================

class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]

        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)

        if root_left == root_right:
            return

        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left

        self.parent[root_right] = root_left

        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


@dataclass(frozen=True)
class Group:
    group_id: str
    member_ids: tuple[str, ...]
    member_indices: tuple[int, ...]
    size: int
    label_counts: tuple[int, int, int]
    exact_law_refs: tuple[str, ...]
    exact_premises: tuple[str, ...]


@dataclass
class SplitCounts:
    totals: list[int]
    labels: list[list[int]]

    @classmethod
    def empty(cls) -> "SplitCounts":
        return cls(
            totals=[0, 0, 0],
            labels=[
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        )

    def copy(self) -> "SplitCounts":
        return SplitCounts(
            totals=list(self.totals),
            labels=[list(row) for row in self.labels],
        )


# ============================================================================
# FICHIERS ET EMPREINTES
# ============================================================================

def find_project_root() -> Path:
    """
    Organisations acceptées :

    1. Script à la racine :
         PROJET/create_flexid_group_split.py
         PROJET/data/flexid_shuffled.jsonl

    2. Script dans scripts/ :
         PROJET/scripts/create_flexid_group_split.py
         PROJET/data/flexid_shuffled.jsonl

    Le répertoire courant de PowerShell ou de VS Code n'intervient pas.
    """
    script_file = Path(__file__).resolve()
    script_dir = script_file.parent

    candidates = (
        script_dir,
        script_dir.parent,
    )

    checked_paths: list[Path] = []

    for candidate in candidates:
        source_file = candidate / INPUT_RELATIVE_PATH
        checked_paths.append(source_file)

        if source_file.is_file():
            return candidate

    rendered = "\n".join(
        f"  - {path}"
        for path in checked_paths
    )

    raise FileNotFoundError(
        "Fichier source introuvable. Chemins absolus vérifiés :\n"
        f"{rendered}\n\n"
        "Le script doit être placé à la racine du projet ou dans scripts/."
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")

            if line == "":
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}, ligne {line_number}: JSON invalide : {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}, ligne {line_number}: "
                    "chaque ligne doit être un objet JSON."
                )

            records.append(record)

    if not records:
        raise ValueError(f"Aucune instance trouvée dans {path}")

    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def write_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def record_fingerprint(record: dict[str, Any]) -> str:
    serialized = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ============================================================================
# EXTRACTION STRICTE DE LAW_REF
# ============================================================================

def get_exact_law_ref(
    record: dict[str, Any],
    record_id: str | None = None,
) -> str:
    """
    Lit law_ref sans aucune normalisation.

    Emplacements acceptés :
    - record["law_ref"]
    - record["meta"]["law_ref"]

    Si les deux existent, leurs valeurs doivent être strictement identiques.
    Aucun strip, casefold, remplacement ou correction n'est appliqué.
    """
    identifier = record_id if record_id is not None else record.get("id", "<sans id>")

    root_present = "law_ref" in record
    root_value = record.get("law_ref")

    meta = record.get("meta")
    meta_is_object = isinstance(meta, dict)
    meta_present = meta_is_object and "law_ref" in meta
    meta_value = meta.get("law_ref") if meta_present else None

    if root_present and not isinstance(root_value, str):
        raise ValueError(
            f"{identifier!r}: law_ref à la racine doit être une chaîne."
        )

    if meta_present and not isinstance(meta_value, str):
        raise ValueError(
            f"{identifier!r}: meta.law_ref doit être une chaîne."
        )

    if root_present and meta_present:
        if root_value != meta_value:
            raise ValueError(
                f"{identifier!r}: law_ref et meta.law_ref existent mais "
                "leurs valeurs exactes sont différentes. "
                "Le script refuse de choisir ou de normaliser."
            )
        value = root_value
    elif root_present:
        value = root_value
    elif meta_present:
        value = meta_value
    else:
        raise ValueError(
            f"{identifier!r}: champ law_ref absent à la racine "
            "et dans meta."
        )

    if value == "":
        raise ValueError(
            f"{identifier!r}: law_ref existe mais sa valeur est vide."
        )

    return value


# ============================================================================
# VALIDATION STRICTE DE LA SOURCE
# ============================================================================

def validate_source(
    records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, int],
    dict[str, str],
]:
    if len(records) != EXPECTED_INSTANCE_COUNT:
        raise ValueError(
            f"Le corpus doit contenir exactement "
            f"{EXPECTED_INSTANCE_COUNT} instances ; "
            f"{len(records)} ont été trouvées."
        )

    records_by_id: dict[str, dict[str, Any]] = {}
    label_index_by_id: dict[str, int] = {}
    fingerprints_by_id: dict[str, str] = {}

    for source_position, record in enumerate(records, start=1):
        if "id" not in record:
            raise ValueError(
                f"Instance #{source_position}: champ id absent."
            )

        record_id = record["id"]

        if not isinstance(record_id, str) or record_id == "":
            raise ValueError(
                f"Instance #{source_position}: id doit être une chaîne "
                "non vide."
            )

        if record_id in records_by_id:
            raise ValueError(
                f"Identifiant dupliqué dans la source : {record_id!r}"
            )

        if "label" not in record:
            raise ValueError(f"{record_id!r}: champ label absent.")

        label = record["label"]

        if label not in LABELS:
            raise ValueError(
                f"{record_id!r}: label exact invalide {label!r}. "
                f"Valeurs autorisées : {LABELS}"
            )

        if "premise" not in record:
            raise ValueError(f"{record_id!r}: champ premise absent.")

        if not isinstance(record["premise"], str):
            raise ValueError(
                f"{record_id!r}: premise doit être une chaîne."
            )

        # Validation stricte de law_ref, à la racine ou dans meta.
        # La valeur est seulement lue ; elle n'est jamais modifiée.
        get_exact_law_ref(record, record_id)

        records_by_id[record_id] = record
        label_index_by_id[record_id] = LABELS.index(label)
        fingerprints_by_id[record_id] = record_fingerprint(record)

    # IMPORTANT :
    # conserver exactement l'ordre de la source (flexid_shuffled.jsonl).
    # Les dictionnaires ci-dessus servent uniquement aux contrôles d'unicité
    # et aux recherches par ID ; ils ne doivent pas réordonner le corpus.
    return (
        list(records),
        label_index_by_id,
        fingerprints_by_id,
    )


# ============================================================================
# GROUPES EXACTS
# ============================================================================

def union_all(indices: Sequence[int], dsu: DisjointSet) -> None:
    if len(indices) < 2:
        return

    first = indices[0]

    for index in indices[1:]:
        dsu.union(first, index)


def build_exact_groups(
    records: list[dict[str, Any]],
    label_index_by_id: dict[str, int],
) -> list[Group]:
    dsu = DisjointSet(len(records))

    indices_by_exact_law_ref: dict[str, list[int]] = defaultdict(list)
    indices_by_exact_premise: dict[str, list[int]] = defaultdict(list)

    for index, record in enumerate(records):
        # Comparaison exacte des valeurs décodées par json.loads.
        # Aucun strip, casefold, regex ou remplacement.
        indices_by_exact_law_ref[
            get_exact_law_ref(record, record["id"])
        ].append(index)
        indices_by_exact_premise[record["premise"]].append(index)

    for indices in indices_by_exact_law_ref.values():
        union_all(indices, dsu)

    for indices in indices_by_exact_premise.values():
        union_all(indices, dsu)

    component_indices: dict[int, list[int]] = defaultdict(list)

    for index in range(len(records)):
        component_indices[dsu.find(index)].append(index)

    provisional: list[dict[str, Any]] = []

    for members in component_indices.values():
        member_ids = tuple(
            sorted(records[index]["id"] for index in members)
        )

        label_counts = [0, 0, 0]

        for index in members:
            record_id = records[index]["id"]
            label_counts[label_index_by_id[record_id]] += 1

        exact_law_refs = tuple(
            sorted(
                {
                    get_exact_law_ref(
                        records[index],
                        records[index]["id"],
                    )
                    for index in members
                }
            )
        )

        exact_premises = tuple(
            sorted(
                {
                    records[index]["premise"]
                    for index in members
                }
            )
        )

        provisional.append(
            {
                "member_ids": member_ids,
                "member_indices": tuple(sorted(members)),
                "size": len(members),
                "label_counts": tuple(label_counts),
                "exact_law_refs": exact_law_refs,
                "exact_premises": exact_premises,
            }
        )

    provisional.sort(key=lambda item: item["member_ids"])

    groups: list[Group] = []

    for ordinal, item in enumerate(provisional, start=1):
        groups.append(
            Group(
                group_id=f"EXACT-GRP-{ordinal:04d}",
                member_ids=item["member_ids"],
                member_indices=item["member_indices"],
                size=item["size"],
                label_counts=item["label_counts"],
                exact_law_refs=item["exact_law_refs"],
                exact_premises=item["exact_premises"],
            )
        )

    verify_exact_group_constraints(records, groups)

    return groups


def verify_exact_group_constraints(
    records: list[dict[str, Any]],
    groups: list[Group],
) -> None:
    group_by_index: dict[int, str] = {}

    for group in groups:
        for index in group.member_indices:
            if index in group_by_index:
                raise RuntimeError(
                    "Une instance appartient à plusieurs groupes."
                )

            group_by_index[index] = group.group_id

    if len(group_by_index) != len(records):
        raise RuntimeError(
            "Les groupes ne couvrent pas toutes les instances."
        )

    group_by_law_ref: dict[str, set[str]] = defaultdict(set)
    group_by_premise: dict[str, set[str]] = defaultdict(set)

    for index, record in enumerate(records):
        group_id = group_by_index[index]
        group_by_law_ref[
            get_exact_law_ref(record, record["id"])
        ].add(group_id)
        group_by_premise[record["premise"]].add(group_id)

    invalid_law_refs = [
        law_ref
        for law_ref, group_ids in group_by_law_ref.items()
        if len(group_ids) != 1
    ]

    invalid_premises = [
        premise
        for premise, group_ids in group_by_premise.items()
        if len(group_ids) != 1
    ]

    if invalid_law_refs:
        raise RuntimeError(
            "Certaines valeurs law_ref exactement identiques ont été "
            "séparées."
        )

    if invalid_premises:
        raise RuntimeError(
            "Certaines prémisses exactement identiques ont été séparées."
        )


# ============================================================================
# CIBLES DE STRATIFICATION
# ============================================================================

def largest_remainder_allocation(
    total: int,
    ratios: dict[str, float],
) -> list[int]:
    raw = [
        total * ratios[split_name]
        for split_name in SPLIT_NAMES
    ]
    allocated = [math.floor(value) for value in raw]
    remaining = total - sum(allocated)

    order = sorted(
        range(len(SPLIT_NAMES)),
        key=lambda index: (
            -(raw[index] - allocated[index]),
            index,
        ),
    )

    for index in order[:remaining]:
        allocated[index] += 1

    return allocated


def build_targets(
    groups: list[Group],
) -> tuple[list[int], list[list[int]], list[int]]:
    """
    Politique d'arrondi reproductible :

    1. appliquer la méthode des plus grands restes séparément à chaque label ;
    2. additionner les cibles par label pour obtenir la cible totale.

    Aucun second arrondi global indépendant n'est effectué.

    Pour les effectifs FLEXID 337 / 333 / 332 :
      train       = 701
      validation  = 151
      test        = 150
    """
    total_labels = [
        sum(group.label_counts[label_index] for group in groups)
        for label_index in range(len(LABELS))
    ]

    target_labels = [
        [0, 0, 0]
        for _ in SPLIT_NAMES
    ]

    for label_index, label_total in enumerate(total_labels):
        allocation = largest_remainder_allocation(
            label_total,
            SPLIT_RATIOS,
        )

        for split_index in range(len(SPLIT_NAMES)):
            target_labels[split_index][label_index] = allocation[
                split_index
            ]

    target_totals = [
        sum(target_labels[split_index])
        for split_index in range(len(SPLIT_NAMES))
    ]

    if sum(target_totals) != sum(total_labels):
        raise RuntimeError(
            "Les cibles stratifiées ne couvrent pas exactement le corpus."
        )

    for split_index in range(len(SPLIT_NAMES)):
        if target_totals[split_index] != sum(
            target_labels[split_index]
        ):
            raise RuntimeError(
                "Incohérence interne entre cible totale et cibles "
                f"par label pour {SPLIT_NAMES[split_index]}."
            )

    return target_totals, target_labels, total_labels


# ============================================================================
# OPTIMISATION DE L'ALLOCATION GROUPÉE
# ============================================================================

def add_group(
    counts: SplitCounts,
    split_index: int,
    group: Group,
    sign: int = 1,
) -> None:
    counts.totals[split_index] += sign * group.size

    for label_index in range(len(LABELS)):
        counts.labels[split_index][label_index] += (
            sign * group.label_counts[label_index]
        )


def final_score(
    counts: SplitCounts,
    target_totals: list[int],
    target_labels: list[list[int]],
) -> float:
    score = 0.0

    for split_index in range(len(SPLIT_NAMES)):
        total_target = max(1, target_totals[split_index])
        total_delta = (
            counts.totals[split_index]
            - target_totals[split_index]
        ) / total_target

        score += 5.0 * total_delta * total_delta

        if counts.totals[split_index] == 0:
            score += 1000.0

        for label_index in range(len(LABELS)):
            label_target = max(
                1,
                target_labels[split_index][label_index],
            )
            label_delta = (
                counts.labels[split_index][label_index]
                - target_labels[split_index][label_index]
            ) / label_target

            score += 3.0 * label_delta * label_delta

            if counts.labels[split_index][label_index] == 0:
                score += 100.0

    return score


def partial_score(
    counts: SplitCounts,
    assigned_total: int,
    assigned_labels: list[int],
    target_totals: list[int],
) -> float:
    score = 0.0

    for split_index, split_name in enumerate(SPLIT_NAMES):
        ratio = SPLIT_RATIOS[split_name]

        expected_total = assigned_total * ratio
        denominator = max(1.0, expected_total)

        delta_total = (
            counts.totals[split_index] - expected_total
        ) / denominator

        score += 4.0 * delta_total * delta_total

        for label_index in range(len(LABELS)):
            expected_label = assigned_labels[label_index] * ratio
            label_denominator = max(1.0, expected_label)

            delta_label = (
                counts.labels[split_index][label_index]
                - expected_label
            ) / label_denominator

            score += 2.0 * delta_label * delta_label

        # Frein d'overshoot par rapport à la cible finale.
        overflow = max(
            0,
            counts.totals[split_index]
            - target_totals[split_index],
        )

        if overflow:
            score += 20.0 * (
                overflow / max(1, target_totals[split_index])
            ) ** 2

    return score


def greedy_assignment(
    groups: list[Group],
    target_totals: list[int],
    restart: int,
) -> tuple[list[int], SplitCounts]:
    rng = random.Random(SPLIT_SEED + restart * 1009)

    ordered_indices = list(range(len(groups)))
    rng.shuffle(ordered_indices)

    # Les grands groupes passent d'abord ; l'aléa départage les groupes
    # de taille égale.
    random_tie = {
        index: rng.random()
        for index in ordered_indices
    }

    ordered_indices.sort(
        key=lambda index: (
            -groups[index].size,
            -max(groups[index].label_counts),
            random_tie[index],
        )
    )

    assignments = [-1] * len(groups)
    counts = SplitCounts.empty()
    assigned_total = 0
    assigned_labels = [0, 0, 0]

    for group_index in ordered_indices:
        group = groups[group_index]
        best_candidates: list[tuple[float, int]] = []

        for split_index in range(len(SPLIT_NAMES)):
            add_group(counts, split_index, group, sign=1)

            candidate_total = assigned_total + group.size
            candidate_labels = [
                assigned_labels[label_index]
                + group.label_counts[label_index]
                for label_index in range(len(LABELS))
            ]

            score = partial_score(
                counts,
                candidate_total,
                candidate_labels,
                target_totals,
            )

            add_group(counts, split_index, group, sign=-1)
            best_candidates.append((score, split_index))

        best_score = min(score for score, _ in best_candidates)
        tied = [
            split_index
            for score, split_index in best_candidates
            if math.isclose(score, best_score, abs_tol=1e-12)
        ]
        selected_split = rng.choice(tied)

        assignments[group_index] = selected_split
        add_group(counts, selected_split, group, sign=1)

        assigned_total += group.size

        for label_index in range(len(LABELS)):
            assigned_labels[label_index] += (
                group.label_counts[label_index]
            )

    return assignments, counts


def rebuild_counts(
    groups: list[Group],
    assignments: list[int],
) -> SplitCounts:
    counts = SplitCounts.empty()

    for group_index, split_index in enumerate(assignments):
        add_group(counts, split_index, groups[group_index], sign=1)

    return counts


def improve_assignment(
    groups: list[Group],
    assignments: list[int],
    target_totals: list[int],
    target_labels: list[list[int]],
    seed_offset: int,
) -> tuple[list[int], SplitCounts, float]:
    rng = random.Random(SPLIT_SEED + 500_000 + seed_offset)
    assignments = list(assignments)
    counts = rebuild_counts(groups, assignments)
    current_score = final_score(
        counts,
        target_totals,
        target_labels,
    )

    for _ in range(LOCAL_SEARCH_ATTEMPTS):
        if rng.random() < 0.65:
            # Déplacement d'un groupe.
            group_index = rng.randrange(len(groups))
            old_split = assignments[group_index]
            new_split = rng.randrange(len(SPLIT_NAMES))

            if new_split == old_split:
                continue

            group = groups[group_index]

            add_group(counts, old_split, group, sign=-1)
            add_group(counts, new_split, group, sign=1)

            candidate_score = final_score(
                counts,
                target_totals,
                target_labels,
            )

            if candidate_score + 1e-12 < current_score:
                assignments[group_index] = new_split
                current_score = candidate_score
            else:
                add_group(counts, new_split, group, sign=-1)
                add_group(counts, old_split, group, sign=1)

        else:
            # Échange de deux groupes affectés à des splits différents.
            left = rng.randrange(len(groups))
            right = rng.randrange(len(groups))

            if left == right:
                continue

            left_split = assignments[left]
            right_split = assignments[right]

            if left_split == right_split:
                continue

            left_group = groups[left]
            right_group = groups[right]

            add_group(counts, left_split, left_group, sign=-1)
            add_group(counts, right_split, right_group, sign=-1)
            add_group(counts, right_split, left_group, sign=1)
            add_group(counts, left_split, right_group, sign=1)

            candidate_score = final_score(
                counts,
                target_totals,
                target_labels,
            )

            if candidate_score + 1e-12 < current_score:
                assignments[left] = right_split
                assignments[right] = left_split
                current_score = candidate_score
            else:
                add_group(counts, right_split, left_group, sign=-1)
                add_group(counts, left_split, right_group, sign=-1)
                add_group(counts, left_split, left_group, sign=1)
                add_group(counts, right_split, right_group, sign=1)

    return assignments, counts, current_score


def optimise_group_assignment(
    groups: list[Group],
) -> tuple[
    list[int],
    SplitCounts,
    list[int],
    list[list[int]],
    float,
]:
    target_totals, target_labels, _ = build_targets(groups)

    candidate_pool: list[tuple[float, tuple[int, ...], list[int]]] = []

    for restart in range(GREEDY_RESTARTS):
        assignments, counts = greedy_assignment(
            groups,
            target_totals,
            restart,
        )

        score = final_score(
            counts,
            target_totals,
            target_labels,
        )

        signature = tuple(assignments)
        candidate_pool.append(
            (score, signature, assignments)
        )

    candidate_pool.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    best_assignments: list[int] | None = None
    best_counts: SplitCounts | None = None
    best_score = float("inf")
    best_signature: tuple[int, ...] | None = None

    unique_candidates: list[list[int]] = []
    seen_signatures: set[tuple[int, ...]] = set()

    for _, signature, assignments in candidate_pool:
        if signature in seen_signatures:
            continue

        seen_signatures.add(signature)
        unique_candidates.append(assignments)

        if len(unique_candidates) >= LOCAL_SEARCH_CANDIDATES:
            break

    for candidate_index, assignments in enumerate(unique_candidates):
        (
            improved_assignments,
            improved_counts,
            improved_score,
        ) = improve_assignment(
            groups,
            assignments,
            target_totals,
            target_labels,
            candidate_index,
        )

        signature = tuple(improved_assignments)

        if (
            improved_score < best_score - 1e-12
            or (
                math.isclose(
                    improved_score,
                    best_score,
                    abs_tol=1e-12,
                )
                and (
                    best_signature is None
                    or signature < best_signature
                )
            )
        ):
            best_assignments = improved_assignments
            best_counts = improved_counts
            best_score = improved_score
            best_signature = signature

    if best_assignments is None or best_counts is None:
        raise RuntimeError(
            "Impossible de construire une allocation groupée."
        )

    return (
        best_assignments,
        best_counts,
        target_totals,
        target_labels,
        best_score,
    )


# ============================================================================
# VALIDATION DU SPLIT GROUPÉ
# ============================================================================

def validate_split(
    records: list[dict[str, Any]],
    groups: list[Group],
    assignments: list[int],
    label_index_by_id: dict[str, int],
) -> dict[str, Any]:
    split_ids: list[set[str]] = [
        set(),
        set(),
        set(),
    ]
    split_group_ids: list[set[str]] = [
        set(),
        set(),
        set(),
    ]

    for group_index, group in enumerate(groups):
        split_index = assignments[group_index]
        split_group_ids[split_index].add(group.group_id)

        for record_id in group.member_ids:
            if record_id in split_ids[split_index]:
                raise RuntimeError(
                    f"ID répété dans un split : {record_id!r}"
                )

            split_ids[split_index].add(record_id)

    for left in range(len(SPLIT_NAMES)):
        for right in range(left + 1, len(SPLIT_NAMES)):
            id_overlap = split_ids[left] & split_ids[right]
            group_overlap = (
                split_group_ids[left]
                & split_group_ids[right]
            )

            if id_overlap:
                raise RuntimeError(
                    "Chevauchement d'instances entre "
                    f"{SPLIT_NAMES[left]} et {SPLIT_NAMES[right]}."
                )

            if group_overlap:
                raise RuntimeError(
                    "Chevauchement de groupes entre "
                    f"{SPLIT_NAMES[left]} et {SPLIT_NAMES[right]}."
                )

    all_source_ids = {
        record["id"]
        for record in records
    }
    all_split_ids = set().union(*split_ids)

    if all_split_ids != all_source_ids:
        raise RuntimeError(
            "Les partitions ne couvrent pas exactement la source."
        )

    statistics: dict[str, Any] = {}

    for split_index, split_name in enumerate(SPLIT_NAMES):
        label_counts = [0, 0, 0]

        for record_id in split_ids[split_index]:
            label_counts[label_index_by_id[record_id]] += 1

        statistics[split_name] = {
            "instances": len(split_ids[split_index]),
            "groups": len(split_group_ids[split_index]),
            "label_distribution": {
                LABELS[label_index]: label_counts[label_index]
                for label_index in range(len(LABELS))
            },
            "proportion": (
                len(split_ids[split_index]) / len(records)
            ),
        }

    return {
        "id_overlap_train_validation": len(
            split_ids[0] & split_ids[1]
        ),
        "id_overlap_train_test": len(
            split_ids[0] & split_ids[2]
        ),
        "id_overlap_validation_test": len(
            split_ids[1] & split_ids[2]
        ),
        "group_overlap_train_validation": len(
            split_group_ids[0] & split_group_ids[1]
        ),
        "group_overlap_train_test": len(
            split_group_ids[0] & split_group_ids[2]
        ),
        "group_overlap_validation_test": len(
            split_group_ids[1] & split_group_ids[2]
        ),
        "all_instances_covered": True,
        "each_instance_exactly_once": (
            sum(len(ids) for ids in split_ids)
            == len(records)
        ),
        "statistics": statistics,
    }


# ============================================================================
# SORTIES
# ============================================================================

def write_outputs(
    project_root: Path,
    records: list[dict[str, Any]],
    groups: list[Group],
    assignments: list[int],
    fingerprints_by_id: dict[str, str],
    validation: dict[str, Any],
    counts: SplitCounts,
    target_totals: list[int],
    target_labels: list[list[int]],
    optimisation_score: float,
    source_file: Path,
) -> None:
    output_directory = project_root / OUTPUT_RELATIVE_DIRECTORY
    output_directory.mkdir(parents=True, exist_ok=True)

    record_by_id = {
        record["id"]: record
        for record in records
    }

    split_by_group_id: dict[str, str] = {}
    split_by_id: dict[str, str] = {}

    for group_index, group in enumerate(groups):
        split_name = SPLIT_NAMES[assignments[group_index]]
        split_by_group_id[group.group_id] = split_name

        for record_id in group.member_ids:
            split_by_id[record_id] = split_name

    # L'ordre shufflé de la source est conservé à l'intérieur de chaque split.
    # Chaque fichier train/validation/test est donc une sous-séquence de
    # flexid_shuffled.jsonl, et non un nouveau tri par ID.
    expected_ids_by_split: dict[str, list[str]] = {}

    for split_name in SPLIT_NAMES:
        split_records = [
            record
            for record in records
            if split_by_id[record["id"]] == split_name
        ]

        expected_ids_by_split[split_name] = [
            record["id"]
            for record in split_records
        ]

        write_jsonl(
            output_directory / f"{split_name}.jsonl",
            split_records,
        )

    # Audit de membership également écrit dans l'ordre source.
    group_by_member_id = {
        record_id: group.group_id
        for group in groups
        for record_id in group.member_ids
    }

    position_within_split = {
        split_name: {
            record_id: position
            for position, record_id in enumerate(
                expected_ids_by_split[split_name],
                start=1,
            )
        }
        for split_name in SPLIT_NAMES
    }

    membership_rows = [
        {
            "id": record["id"],
            "label": record["label"],
            "split": split_by_id[record["id"]],
            "exact_group": group_by_member_id[record["id"]],
            "source_position": source_position,
            "position_within_split": position_within_split[
                split_by_id[record["id"]]
            ][record["id"]],
        }
        for source_position, record in enumerate(records, start=1)
    ]

    write_jsonl(
        output_directory / "split_membership.jsonl",
        membership_rows,
    )

    group_membership_rows = [
        {
            "exact_group": group.group_id,
            "split": split_by_group_id[group.group_id],
            "size": group.size,
            "label_distribution": {
                LABELS[label_index]: group.label_counts[label_index]
                for label_index in range(len(LABELS))
            },
            "member_ids": list(group.member_ids),
        }
        for group in groups
    ]

    write_jsonl(
        output_directory / "group_membership.jsonl",
        group_membership_rows,
    )

    group_audit_rows = [
        {
            "exact_group": group.group_id,
            "split": split_by_group_id[group.group_id],
            "size": group.size,
            "member_ids": list(group.member_ids),
            "exact_law_refs": list(group.exact_law_refs),
            "exact_premises": list(group.exact_premises),
            "label_distribution": {
                LABELS[label_index]: group.label_counts[label_index]
                for label_index in range(len(LABELS))
            },
            "normalization_applied": False,
        }
        for group in groups
    ]

    write_jsonl(
        output_directory / "group_audit.jsonl",
        group_audit_rows,
    )

    # Vérifier que les objets écrits sont identiques aux objets source.
    written_ids: set[str] = set()

    for split_name in SPLIT_NAMES:
        written = load_jsonl(
            output_directory / f"{split_name}.jsonl"
        )

        written_order = [
            record["id"]
            for record in written
        ]

        if written_order != expected_ids_by_split[split_name]:
            raise RuntimeError(
                f"{split_name}: l'ordre shufflé de la source "
                "n'a pas été conservé dans le split."
            )

        for record in written:
            record_id = record["id"]

            if record_id in written_ids:
                raise RuntimeError(
                    f"ID écrit plusieurs fois : {record_id!r}"
                )

            written_ids.add(record_id)

            if record_fingerprint(record) != fingerprints_by_id[record_id]:
                raise RuntimeError(
                    f"Objet source modifié : {record_id!r}"
                )

    if written_ids != set(fingerprints_by_id):
        raise RuntimeError(
            "Les sorties ne couvrent pas exactement la source."
        )

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "source": {
            "path": str(source_file),
            "sha256": file_sha256(source_file),
            "instances": len(records),
        },
        "grouping": {
            "method": (
                "transitive connected components over exact equality of "
                "pre-normalized law_ref strings OR exact premise strings"
            ),
            "same_exact_law_ref_linked": True,
            "law_ref_locations_accepted": ["law_ref", "meta.law_ref"],
            "same_exact_premise_linked": True,
            "transitive_components": True,
            "normalization_applied_in_split_script": False,
            "input_law_refs_pre_normalized_upstream": True,
            "normalization_verified_by_split_script": False,
            "grouping_uses_exact_pre_normalized_law_ref": True,
            "strip_applied_in_split_script": False,
            "case_conversion_applied_in_split_script": False,
            "regex_applied_in_split_script": False,
            "typographical_correction_applied_in_split_script": False,
            "law_ref_modified_by_split_script": False,
            "premise_modified_by_split_script": False,
            "groups": len(groups),
            "singletons": sum(group.size == 1 for group in groups),
            "multi_instance_groups": sum(
                group.size > 1 for group in groups
            ),
            "largest_group": max(group.size for group in groups),
            "preprocessing_note": (
                "Legal-reference canonicalisation is an upstream preprocessing "
                "step. This split script performs no additional normalization "
                "and groups by exact equality of the resulting canonical "
                "law_ref strings, with exact premise equality as an additional "
                "linking criterion."
            ),
            "remaining_limitation": (
                "The split script does not infer semantic equivalence beyond "
                "the upstream canonical law_ref strings and exact premise "
                "equality."
            ),
        },
        "stratification": {
            "seed": SPLIT_SEED,
            "requested_ratios": SPLIT_RATIOS,
            "rounding_policy": (
                "largest remainder independently within each label; "
                "total targets are sums of label targets"
            ),
            "target_instances_derived_from_target_labels": True,
            "target_instances": {
                SPLIT_NAMES[index]: target_totals[index]
                for index in range(len(SPLIT_NAMES))
            },
            "target_labels": {
                SPLIT_NAMES[split_index]: {
                    LABELS[label_index]: target_labels[
                        split_index
                    ][label_index]
                    for label_index in range(len(LABELS))
                }
                for split_index in range(len(SPLIT_NAMES))
            },
            "actual_instances": {
                SPLIT_NAMES[index]: counts.totals[index]
                for index in range(len(SPLIT_NAMES))
            },
            "actual_labels": {
                SPLIT_NAMES[split_index]: {
                    LABELS[label_index]: counts.labels[
                        split_index
                    ][label_index]
                    for label_index in range(len(LABELS))
                }
                for split_index in range(len(SPLIT_NAMES))
            },
            "optimisation_score": optimisation_score,
            "group_constraints_may_prevent_exact_targets": True,
        },
        "validation": validation,
        "records_preserved_exactly": True,
        "ordering": {
            "source_file_is_shuffled_input": True,
            "source_order_preserved_within_each_split": True,
            "split_files_are_source_order_subsequences": True,
            "sorting_by_id_applied_to_split_files": False,
        },
        "outputs": {
            split_name: str(
                output_directory / f"{split_name}.jsonl"
            )
            for split_name in SPLIT_NAMES
        }
        | {
            "split_membership": str(
                output_directory / "split_membership.jsonl"
            ),
            "group_membership": str(
                output_directory / "group_membership.jsonl"
            ),
            "group_audit": str(
                output_directory / "group_audit.jsonl"
            ),
        },
    }

    for split_name in SPLIT_NAMES:
        total_target = summary["stratification"][
            "target_instances"
        ][split_name]
        label_target_sum = sum(
            summary["stratification"]["target_labels"][
                split_name
            ].values()
        )

        if total_target != label_target_sum:
            raise RuntimeError(
                "Résumé refusé : target_instances et target_labels "
                f"sont incohérents pour {split_name}."
            )

    write_json(
        output_directory / "split_summary.json",
        summary,
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    project_root = find_project_root()
    source_file = project_root / INPUT_RELATIVE_PATH

    source_records = load_jsonl(source_file)

    (
        records,
        label_index_by_id,
        fingerprints_by_id,
    ) = validate_source(source_records)

    groups = build_exact_groups(
        records,
        label_index_by_id,
    )

    (
        assignments,
        counts,
        target_totals,
        target_labels,
        optimisation_score,
    ) = optimise_group_assignment(groups)

    validation = validate_split(
        records,
        groups,
        assignments,
        label_index_by_id,
    )

    write_outputs(
        project_root=project_root,
        records=records,
        groups=groups,
        assignments=assignments,
        fingerprints_by_id=fingerprints_by_id,
        validation=validation,
        counts=counts,
        target_totals=target_totals,
        target_labels=target_labels,
        optimisation_score=optimisation_score,
        source_file=source_file,
    )

    output_directory = project_root / OUTPUT_RELATIVE_DIRECTORY

    print("Split FLEXID par groupes exacts terminé.")
    print(f"Version du protocole       : {PROTOCOL_VERSION}")
    print(f"Source unique              : {source_file}")
    print(f"Instances                  : {len(records)}")
    print(f"Groupes exacts             : {len(groups)}")
    print(
        "Groupes multi-instances    : "
        f"{sum(group.size > 1 for group in groups)}"
    )
    print(
        "Plus grand groupe          : "
        f"{max(group.size for group in groups)}"
    )
    print("Normalisation dans ce script: NON")
    print("law_ref pré-normalisées      : OUI (étape amont)")
    print("Groupement law_ref exacte    : OUI")
    print("Modification de law_ref      : NON")
    print("Ordre shufflé dans les splits: CONSERVÉ")
    print("")
    print(
        "Cibles stratifiées         : "
        f"train={target_totals[0]}, "
        f"validation={target_totals[1]}, "
        f"test={target_totals[2]}"
    )
    print("")

    for split_index, split_name in enumerate(SPLIT_NAMES):
        labels = counts.labels[split_index]

        print(
            f"{split_name:<11}: "
            f"{counts.totals[split_index]:>4} instances | "
            f"entailment={labels[0]} | "
            f"contradiction={labels[1]} | "
            f"neutral={labels[2]}"
        )

    print("")
    print(
        "Chevauchement groupes train/test : "
        f"{validation['group_overlap_train_test']}"
    )
    print(
        "Chevauchement ID train/test      : "
        f"{validation['id_overlap_train_test']}"
    )
    print(
        "Chaque instance apparaît une fois: "
        f"{validation['each_instance_exactly_once']}"
    )
    print(f"Sorties                            : {output_directory}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        raise SystemExit(1)
