#!/usr/bin/env python3


from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils.validation import check_is_fitted


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = Path("data/flexid.jsonl")
PREDICT_FILE = Path("data/pairs_to_predict.jsonl")
OUTPUT_DIR = Path("results/full_nli_audit_final")

RUN_EXTERNAL_PREDICTION = True
RUN_HYPOTHESIS_ONLY_CONTROL = True  # Déjà évalué dans hypothesis_only_classifier.py
RUN_PREMISE_ONLY_CONTROL = True
RUN_SHUFFLED_PREMISE_CONTROL = True

ID_FIELD = "id"
PREMISE_FIELD = "premise"
HYPOTHESIS_FIELD = "hypothesis_facts"
LABEL_FIELD = "label"

LABELS = ["entailment", "contradiction", "neutral"]
EXPECTED_INSTANCE_COUNT = 1_002
EXPECTED_LABEL_COUNTS = {
    "entailment": 337,
    "contradiction": 333,
    "neutral": 332,
}
EXPECTED_GROUP_COUNT = 335
PROTOCOL_VERSION = "FLEXID-FINAL-RELATIONAL-AUDIT-v2"

OUTER_FOLDS = 5
OUTER_SEEDS = [1701, 2718, 3141]
INNER_FOLDS = 4

C_VALUES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
CLASS_WEIGHTS = [None, "balanced"]

SINGLE_WORD_MAX_FEATURES = 80_000
SINGLE_CHAR_MAX_FEATURES = 120_000
PAIR_WORD_MAX_FEATURES = 60_000
PAIR_CHAR_MAX_FEATURES = 90_000

TOP_FEATURES_PER_CLASS = 60
N_JOBS = -1
FINAL_MODEL_SEED = 8_675_309
PAIRED_BOOTSTRAP_SEED = 2026
PAIRED_BOOTSTRAP_ITERATIONS = 10_000

VARIANT_FULL = "premise_plus_hypothesis"
VARIANT_SHUFFLED = "shuffled_premise"
VARIANT_HYPOTHESIS = "hypothesis_only"
VARIANT_PREMISE = "premise_only"


# ============================================================================
# JSONL ET VALIDATION
# ============================================================================

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON invalide dans {path}, ligne {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Objet JSON invalide dans {path}, ligne {line_number}."
                )

            records.append(record)

    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("Le corpus est vide.")

    seen_ids: set[str] = set()

    for index, record in enumerate(records, start=1):
        for field in (ID_FIELD, PREMISE_FIELD, HYPOTHESIS_FIELD, LABEL_FIELD):
            if field not in record:
                raise ValueError(
                    f"Champ {field!r} absent à l'instance {index}."
                )

        record_id = str(record[ID_FIELD]).strip()
        premise = record[PREMISE_FIELD]
        hypothesis = record[HYPOTHESIS_FIELD]
        label = record[LABEL_FIELD]

        if not record_id:
            raise ValueError(f"ID vide à l'instance {index}.")
        if record_id in seen_ids:
            raise ValueError(f"ID dupliqué : {record_id}")
        seen_ids.add(record_id)

        if not isinstance(premise, str) or not premise.strip():
            raise ValueError(f"Prémisse invalide pour {record_id}.")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError(f"Hypothèse invalide pour {record_id}.")
        if label not in LABELS:
            raise ValueError(
                f"Label invalide pour {record_id}: {label!r}"
            )

    if len(records) != EXPECTED_INSTANCE_COUNT:
        raise ValueError(
            f"Le corpus doit contenir {EXPECTED_INSTANCE_COUNT} instances, "
            f"pas {len(records)}."
        )

    observed_counts = Counter(
        str(record[LABEL_FIELD])
        for record in records
    )
    if dict(observed_counts) != EXPECTED_LABEL_COUNTS:
        raise ValueError(
            "Distribution des labels inattendue : "
            f"{dict(observed_counts)} ; attendu : {EXPECTED_LABEL_COUNTS}."
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ============================================================================
# GROUPES ANTI-FUITE
# ============================================================================

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    return re.sub(r"\s+", " ", text).strip()


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)

        if left_root == right_root:
            return

        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root

        self.parent[right_root] = left_root

        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def law_reference(record: dict[str, Any]) -> str:
    values: list[str] = []

    direct = record.get("law_ref")
    if direct:
        values.append(normalize_text(str(direct)))

    meta = record.get("meta")
    if isinstance(meta, dict):
        value = meta.get("law_ref")
        if value:
            values.append(normalize_text(str(value)))

    distinct_values = set(values)
    if len(distinct_values) > 1:
        raise ValueError(
            f"Références juridiques incohérentes pour {record.get(ID_FIELD)!r}."
        )

    return values[0] if values else ""


def build_groups(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Fusionne les instances partageant :
    - meta.law_ref ;
    - la même prémisse normalisée ;
    - la même hypothèse normalisée.

    Les groupes sont calculés une seule fois sur les données originales et
    réutilisés dans toutes les variantes.
    """
    uf = UnionFind(len(records))
    first_seen: dict[tuple[str, str], int] = {}
    duplicate_hypotheses = Counter()

    for index, record in enumerate(records):
        keys: list[tuple[str, str]] = []

        ref = law_reference(record)
        if ref:
            keys.append(("law_ref", ref))

        premise = normalize_text(str(record[PREMISE_FIELD]))
        keys.append(("premise", premise))

        hypothesis = normalize_text(str(record[HYPOTHESIS_FIELD]))
        keys.append(("hypothesis", hypothesis))
        duplicate_hypotheses[hypothesis] += 1

        for key in keys:
            if key in first_seen:
                uf.union(index, first_seen[key])
            else:
                first_seen[key] = index

    roots = [uf.find(index) for index in range(len(records))]
    root_names = {
        root: f"group-{number:04d}"
        for number, root in enumerate(sorted(set(roots)), start=1)
    }
    groups = np.asarray([root_names[root] for root in roots], dtype=object)
    sizes = Counter(groups.tolist())

    diagnostics = {
        "instances": len(records),
        "groups": len(sizes),
        "grouping_method": (
            "transitive Union-Find over normalized law_ref OR normalized "
            "premise OR normalized hypothesis"
        ),
        "normalization": "NFKC + casefold + whitespace collapse",
        "minimum_group_size": min(sizes.values()),
        "maximum_group_size": max(sizes.values()),
        "mean_group_size": float(np.mean(list(sizes.values()))),
        "groups_with_multiple_instances": sum(
            size > 1 for size in sizes.values()
        ),
        "duplicated_hypothesis_forms": sum(
            count > 1 for count in duplicate_hypotheses.values()
        ),
    }

    if diagnostics["groups"] != EXPECTED_GROUP_COUNT:
        raise ValueError(
            f"Le protocole attend {EXPECTED_GROUP_COUNT} groupes, "
            f"mais {diagnostics['groups']} ont été construits."
        )

    return groups, diagnostics


# ============================================================================
# REPRÉSENTATION DE LA PAIRE PRÉMISSE–HYPOTHÈSE
# ============================================================================

class PairTfidfFeatures(BaseEstimator, TransformerMixin):
    """
    Représentation lexicale relationnelle d'une paire de textes.

    Les deux textes sont projetés dans le même espace TF-IDF. Pour chaque
    analyseur, la sortie contient : P, H, |P-H| et P*H.
    """

    def __init__(
        self,
        word_max_features: int = PAIR_WORD_MAX_FEATURES,
        char_max_features: int = PAIR_CHAR_MAX_FEATURES,
        min_df: int = 2,
        max_df: float = 0.995,
    ) -> None:
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features
        self.min_df = min_df
        self.max_df = max_df

    @staticmethod
    def _extract_columns(X: Any) -> tuple[list[str], list[str]]:
        if isinstance(X, pd.DataFrame):
            premises = X[PREMISE_FIELD].astype(str).tolist()
            hypotheses = X[HYPOTHESIS_FIELD].astype(str).tolist()
            return premises, hypotheses

        array = np.asarray(X, dtype=object)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(
                "PairTfidfFeatures attend un DataFrame ou une matrice à deux colonnes."
            )
        return array[:, 0].astype(str).tolist(), array[:, 1].astype(str).tolist()

    def fit(self, X: Any, y: Any = None) -> "PairTfidfFeatures":
        premises, hypotheses = self._extract_columns(X)
        corpus = premises + hypotheses

        self.word_vectorizer_ = TfidfVectorizer(
            analyzer="word",
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.word_max_features,
            sublinear_tf=True,
        )
        self.char_vectorizer_ = TfidfVectorizer(
            analyzer="char_wb",
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(3, 5),
            min_df=self.min_df,
            max_features=self.char_max_features,
            sublinear_tf=True,
        )

        self.word_vectorizer_.fit(corpus)
        self.char_vectorizer_.fit(corpus)
        return self

    def transform(self, X: Any) -> sparse.csr_matrix:
        check_is_fitted(
            self,
            attributes=["word_vectorizer_", "char_vectorizer_"],
        )
        premises, hypotheses = self._extract_columns(X)

        premise_word = self.word_vectorizer_.transform(premises)
        hypothesis_word = self.word_vectorizer_.transform(hypotheses)
        premise_char = self.char_vectorizer_.transform(premises)
        hypothesis_char = self.char_vectorizer_.transform(hypotheses)

        word_difference = abs(premise_word - hypothesis_word)
        word_overlap = premise_word.multiply(hypothesis_word)
        char_difference = abs(premise_char - hypothesis_char)
        char_overlap = premise_char.multiply(hypothesis_char)

        return sparse.hstack(
            [
                premise_word,
                hypothesis_word,
                word_difference,
                word_overlap,
                premise_char,
                hypothesis_char,
                char_difference,
                char_overlap,
            ],
            format="csr",
        )

    def get_feature_names_out(
        self,
        input_features: Sequence[str] | None = None,
    ) -> np.ndarray:
        check_is_fitted(
            self,
            attributes=["word_vectorizer_", "char_vectorizer_"],
        )
        word_names = self.word_vectorizer_.get_feature_names_out()
        char_names = self.char_vectorizer_.get_feature_names_out()

        names: list[str] = []
        for prefix in (
            "premise_word",
            "hypothesis_word",
            "absdiff_word",
            "overlap_word",
        ):
            names.extend(f"{prefix}__{name}" for name in word_names)

        for prefix in (
            "premise_char",
            "hypothesis_char",
            "absdiff_char",
            "overlap_char",
        ):
            names.extend(f"{prefix}__{name}" for name in char_names)

        return np.asarray(names, dtype=object)


# ============================================================================
# MODÈLES
# ============================================================================

def make_classifier(seed: int) -> LogisticRegression:
    return LogisticRegression(
        solver="saga",
        max_iter=10_000,
        tol=1e-4,
        random_state=seed,
    )


def make_text_pipeline(seed: int) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.995,
                    max_features=SINGLE_WORD_MAX_FEATURES,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=SINGLE_CHAR_MAX_FEATURES,
                    sublinear_tf=True,
                ),
            ),
        ],
        n_jobs=1,
    )

    return Pipeline(
        [
            ("features", features),
            ("classifier", make_classifier(seed)),
        ]
    )


def make_pair_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("features", PairTfidfFeatures()),
            ("classifier", make_classifier(seed)),
        ]
    )


PARAM_GRID = {
    "classifier__C": C_VALUES,
    "classifier__class_weight": CLASS_WEIGHTS,
}


def align_probabilities(model: Pipeline, probabilities: np.ndarray) -> np.ndarray:
    model_classes = model.named_steps["classifier"].classes_
    class_to_column = {
        str(label): column
        for column, label in enumerate(model_classes)
    }

    aligned = np.zeros((len(probabilities), len(LABELS)), dtype=float)
    for target_column, label in enumerate(LABELS):
        aligned[:, target_column] = probabilities[
            :, class_to_column[label]
        ]
    return aligned


def labels_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.asarray(
        [LABELS[index] for index in np.argmax(probabilities, axis=1)],
        dtype=object,
    )


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def paired_group_bootstrap_accuracy(
    labels: np.ndarray,
    first_predictions: np.ndarray,
    second_predictions: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int = PAIRED_BOOTSTRAP_ITERATIONS,
    seed: int = PAIRED_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """
    Intervalle percentile apparié pour une différence d'accuracy.

    Les groupes anti-fuite, et non les instances individuelles, sont
    rééchantillonnés avec remise. Toutes les instances d'un groupe tiré sont
    conservées ensemble. L'estimation reste pondérée par le nombre
    d'instances, comme l'accuracy corpus-level rapportée dans l'article.
    """
    if iterations < 1:
        raise ValueError("Le nombre d'itérations bootstrap doit être positif.")

    unique_groups = np.asarray(sorted(set(groups.tolist())), dtype=object)
    group_delta_sums: list[int] = []
    group_sizes: list[int] = []

    first_correct = first_predictions == labels
    second_correct = second_predictions == labels

    for group in unique_groups:
        group_mask = groups == group
        group_delta_sums.append(
            int(
                np.sum(
                    first_correct[group_mask].astype(int)
                    - second_correct[group_mask].astype(int)
                )
            )
        )
        group_sizes.append(int(np.sum(group_mask)))

    delta_array = np.asarray(group_delta_sums, dtype=float)
    size_array = np.asarray(group_sizes, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)

    for iteration in range(iterations):
        sampled = rng.integers(0, len(unique_groups), len(unique_groups))
        estimates[iteration] = (
            float(np.sum(delta_array[sampled]))
            / float(np.sum(size_array[sampled]))
        )

    point_estimate = float(
        accuracy_score(labels, first_predictions)
        - accuracy_score(labels, second_predictions)
    )

    return {
        "metric": "accuracy",
        "first_minus_second": point_estimate,
        "confidence_interval_95": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "method": "paired percentile cluster bootstrap over leakage-control groups",
        "groups": int(len(unique_groups)),
        "iterations": iterations,
        "seed": seed,
    }


# ============================================================================
# PLIS FIXES ET CONTRÔLE DE PRÉMISSES MÉLANGÉES
# ============================================================================

def hash_ids(ids: np.ndarray, indices: np.ndarray) -> str:
    payload = "\n".join(str(ids[index]) for index in indices)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_fixed_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    ids: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_specs: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    dummy = np.zeros(len(labels), dtype=np.int8)

    for repeat, outer_seed in enumerate(OUTER_SEEDS, start=1):
        outer_cv = StratifiedGroupKFold(
            n_splits=OUTER_FOLDS,
            shuffle=True,
            random_state=outer_seed,
        )

        for fold, (train_idx, test_idx) in enumerate(
            outer_cv.split(dummy, labels, groups),
            start=1,
        ):
            train_groups = set(groups[train_idx].tolist())
            test_groups = set(groups[test_idx].tolist())
            if train_groups & test_groups:
                raise RuntimeError("Fuite de groupes entre train et test.")

            inner_cv = StratifiedGroupKFold(
                n_splits=INNER_FOLDS,
                shuffle=True,
                random_state=outer_seed * 100 + fold,
            )
            inner_splits = list(
                inner_cv.split(
                    np.zeros(len(train_idx), dtype=np.int8),
                    labels[train_idx],
                    groups[train_idx],
                )
            )

            spec = {
                "repeat": repeat,
                "fold": fold,
                "seed": outer_seed,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "inner_splits": inner_splits,
            }
            split_specs.append(spec)

            audit_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "seed": outer_seed,
                    "train_instances": len(train_idx),
                    "test_instances": len(test_idx),
                    "train_groups": len(train_groups),
                    "test_groups": len(test_groups),
                    "train_ids_sha256": hash_ids(ids, train_idx),
                    "test_ids_sha256": hash_ids(ids, test_idx),
                }
            )

    return split_specs, audit_rows


def find_cross_group_permutation(
    test_idx: np.ndarray,
    groups: np.ndarray,
    normalized_premises: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Cherche une permutation déterministe où chaque prémisse provient d'une
    autre instance, d'un autre groupe et d'une prémisse normalisée différente.
    """
    size = len(test_idx)
    if size < 2:
        raise RuntimeError("Pli de test trop petit pour mélanger les prémisses.")

    local = np.arange(size)
    test_groups = groups[test_idx]
    test_premises = normalized_premises[test_idx]
    rng = np.random.default_rng(seed)

    best_permutation: np.ndarray | None = None
    best_valid = -1

    for _ in range(10_000):
        permutation = rng.permutation(size)
        valid = (
            (permutation != local)
            & (test_groups[permutation] != test_groups)
            & (test_premises[permutation] != test_premises)
        )
        valid_count = int(np.sum(valid))

        if valid_count > best_valid:
            best_valid = valid_count
            best_permutation = permutation.copy()

        if valid_count == size:
            return permutation, {
                "test_instances": size,
                "valid_mismatches": size,
                "mismatch_rate": 1.0,
                "perfect_derangement": True,
            }

    raise RuntimeError(
        "Impossible de construire une permutation parfaite des prémisses "
        f"après 10 000 essais ({best_valid}/{size} appariements valides)."
    )


def take_rows(X: Any, indices: np.ndarray) -> Any:
    if isinstance(X, pd.DataFrame):
        return X.iloc[indices].reset_index(drop=True)
    return np.asarray(X, dtype=object)[indices]


# ============================================================================
# VALIDATION CROISÉE
# ============================================================================

def evaluate_single_text_variant(
    variant: str,
    texts: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    split_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    probability_sum = np.zeros((len(texts), len(LABELS)), dtype=float)
    prediction_counts = np.zeros(len(texts), dtype=int)
    fold_rows: list[dict[str, Any]] = []

    for spec in split_specs:
        train_idx = spec["train_idx"]
        test_idx = spec["test_idx"]
        seed = spec["seed"] * 100 + spec["fold"]

        search = GridSearchCV(
            estimator=make_text_pipeline(seed),
            param_grid=PARAM_GRID,
            scoring="balanced_accuracy",
            cv=spec["inner_splits"],
            n_jobs=N_JOBS,
            refit=True,
            error_score="raise",
        )
        search.fit(
            texts[train_idx],
            labels[train_idx],
            groups=groups[train_idx],
        )

        model = search.best_estimator_
        probabilities = align_probabilities(
            model,
            model.predict_proba(texts[test_idx]),
        )
        predictions = labels_from_probabilities(probabilities)

        probability_sum[test_idx] += probabilities
        prediction_counts[test_idx] += 1

        fold_rows.append(
            {
                "variant": variant,
                "repeat": spec["repeat"],
                "fold": spec["fold"],
                "seed": spec["seed"],
                "best_inner_balanced_accuracy": float(search.best_score_),
                "best_C": float(search.best_params_["classifier__C"]),
                "best_class_weight": str(
                    search.best_params_["classifier__class_weight"]
                ),
                **scores(labels[test_idx], predictions),
            }
        )

    expected = len(OUTER_SEEDS)
    if not np.all(prediction_counts == expected):
        raise RuntimeError(
            f"Prédictions OOF incomplètes pour la variante {variant}."
        )

    probabilities = probability_sum / prediction_counts[:, np.newaxis]
    predictions = labels_from_probabilities(probabilities)
    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "fold_rows": fold_rows,
        "scores": scores(labels, predictions),
    }


def evaluate_full_and_shuffled(
    pairs: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    ids: np.ndarray,
    normalized_premises: np.ndarray,
    split_specs: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    full_probability_sum = np.zeros((len(pairs), len(LABELS)), dtype=float)
    shuffled_probability_sum = np.zeros_like(full_probability_sum)
    prediction_counts = np.zeros(len(pairs), dtype=int)

    full_fold_rows: list[dict[str, Any]] = []
    shuffled_fold_rows: list[dict[str, Any]] = []
    shuffle_map_rows: list[dict[str, Any]] = []

    for spec in split_specs:
        train_idx = spec["train_idx"]
        test_idx = spec["test_idx"]
        seed = spec["seed"] * 100 + spec["fold"]

        search = GridSearchCV(
            estimator=make_pair_pipeline(seed),
            param_grid=PARAM_GRID,
            scoring="balanced_accuracy",
            cv=spec["inner_splits"],
            n_jobs=N_JOBS,
            refit=True,
            error_score="raise",
        )
        search.fit(
            take_rows(pairs, train_idx),
            labels[train_idx],
            groups=groups[train_idx],
        )
        model = search.best_estimator_

        correct_test_pairs = take_rows(pairs, test_idx)
        full_probabilities = align_probabilities(
            model,
            model.predict_proba(correct_test_pairs),
        )
        full_predictions = labels_from_probabilities(full_probabilities)

        full_probability_sum[test_idx] += full_probabilities
        prediction_counts[test_idx] += 1

        common_fold_metadata = {
            "repeat": spec["repeat"],
            "fold": spec["fold"],
            "seed": spec["seed"],
            "best_inner_balanced_accuracy": float(search.best_score_),
            "best_C": float(search.best_params_["classifier__C"]),
            "best_class_weight": str(
                search.best_params_["classifier__class_weight"]
            ),
        }
        full_fold_rows.append(
            {
                "variant": VARIANT_FULL,
                **common_fold_metadata,
                **scores(labels[test_idx], full_predictions),
            }
        )

        if RUN_SHUFFLED_PREMISE_CONTROL:
            permutation, permutation_diagnostics = find_cross_group_permutation(
                test_idx,
                groups,
                normalized_premises,
                seed=spec["seed"] * 10_000 + spec["fold"],
            )
            source_idx = test_idx[permutation]

            shuffled_test_pairs = correct_test_pairs.copy()
            shuffled_test_pairs[PREMISE_FIELD] = pairs.iloc[source_idx][
                PREMISE_FIELD
            ].to_numpy()

            shuffled_probabilities = align_probabilities(
                model,
                model.predict_proba(shuffled_test_pairs),
            )
            shuffled_predictions = labels_from_probabilities(
                shuffled_probabilities
            )
            shuffled_probability_sum[test_idx] += shuffled_probabilities

            shuffled_fold_rows.append(
                {
                    "variant": VARIANT_SHUFFLED,
                    **common_fold_metadata,
                    **permutation_diagnostics,
                    **scores(labels[test_idx], shuffled_predictions),
                }
            )

            for target_position, source_position in zip(test_idx, source_idx):
                shuffle_map_rows.append(
                    {
                        "repeat": spec["repeat"],
                        "fold": spec["fold"],
                        "target_id": str(ids[target_position]),
                        "premise_source_id": str(ids[source_position]),
                        "target_group": str(groups[target_position]),
                        "premise_source_group": str(groups[source_position]),
                        "same_group": bool(
                            groups[target_position] == groups[source_position]
                        ),
                        "same_normalized_premise": bool(
                            normalized_premises[target_position]
                            == normalized_premises[source_position]
                        ),
                    }
                )

    expected = len(OUTER_SEEDS)
    if not np.all(prediction_counts == expected):
        raise RuntimeError("Prédictions OOF incomplètes pour le modèle complet.")

    full_probabilities = full_probability_sum / prediction_counts[:, np.newaxis]
    full_predictions = labels_from_probabilities(full_probabilities)
    full_result = {
        "predictions": full_predictions,
        "probabilities": full_probabilities,
        "fold_rows": full_fold_rows,
        "scores": scores(labels, full_predictions),
    }

    if RUN_SHUFFLED_PREMISE_CONTROL:
        shuffled_probabilities = (
            shuffled_probability_sum / prediction_counts[:, np.newaxis]
        )
        shuffled_predictions = labels_from_probabilities(shuffled_probabilities)
        shuffled_result = {
            "predictions": shuffled_predictions,
            "probabilities": shuffled_probabilities,
            "fold_rows": shuffled_fold_rows,
            "scores": scores(labels, shuffled_predictions),
        }
    else:
        shuffled_result = {}

    return full_result, shuffled_result, shuffle_map_rows


# ============================================================================
# MODÈLE FINAL ET INTERPRÉTATION
# ============================================================================

def fit_final_full_model(
    pairs: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[Pipeline, dict[str, Any]]:
    cv = StratifiedGroupKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=FINAL_MODEL_SEED,
    )

    search = GridSearchCV(
        estimator=make_pair_pipeline(FINAL_MODEL_SEED),
        param_grid=PARAM_GRID,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=N_JOBS,
        refit=True,
        error_score="raise",
    )
    search.fit(pairs, labels, groups=groups)

    metadata = {
        "best_grouped_cv_balanced_accuracy": float(search.best_score_),
        "best_parameters": {
            key: (
                float(value)
                if isinstance(value, (int, float, np.number))
                else value
            )
            for key, value in search.best_params_.items()
        },
        "representation": [
            "premise_tfidf",
            "hypothesis_tfidf",
            "absolute_difference",
            "elementwise_overlap",
        ],
    }
    return search.best_estimator_, metadata


def top_features(model: Pipeline) -> dict[str, Any]:
    names = np.asarray(
        model.named_steps["features"].get_feature_names_out(),
        dtype=object,
    )
    classifier = model.named_steps["classifier"]
    output: dict[str, Any] = {}

    for class_index, label in enumerate(classifier.classes_):
        coefficients = classifier.coef_[class_index]
        positive = np.argsort(coefficients)[-TOP_FEATURES_PER_CLASS:][::-1]
        negative = np.argsort(coefficients)[:TOP_FEATURES_PER_CLASS]

        output[str(label)] = {
            "most_positive": [
                {
                    "feature": str(names[index]),
                    "coefficient": float(coefficients[index]),
                }
                for index in positive
            ],
            "most_negative": [
                {
                    "feature": str(names[index]),
                    "coefficient": float(coefficients[index]),
                }
                for index in negative
            ],
        }

    return output


# ============================================================================
# SORTIES
# ============================================================================

def variant_diagnostics(
    variant: str,
    labels: np.ndarray,
    result: dict[str, Any],
) -> dict[str, Any]:
    predictions = result["predictions"]
    return {
        "variant": variant,
        **result["scores"],
        "classification_report": classification_report(
            labels,
            predictions,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=LABELS,
        ).tolist(),
    }


def write_confusion_matrix(
    variant: str,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> None:
    matrix = confusion_matrix(labels, predictions, labels=LABELS)
    pd.DataFrame(
        matrix,
        index=[f"gold_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    ).to_csv(
        OUTPUT_DIR / f"05_confusion_matrix_{variant}.csv",
        encoding="utf-8",
    )


def build_oof_rows(
    records: list[dict[str, Any]],
    groups: np.ndarray,
    results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        row: dict[str, Any] = {
            "index": index,
            "id": record[ID_FIELD],
            "group": str(groups[index]),
            "premise": record[PREMISE_FIELD],
            "hypothesis_facts": record[HYPOTHESIS_FIELD],
            "gold_label": record[LABEL_FIELD],
        }

        for variant, result in results.items():
            probabilities = result["probabilities"]
            prediction = str(result["predictions"][index])
            row[f"predicted_{variant}"] = prediction
            row[f"correct_{variant}"] = bool(
                prediction == record[LABEL_FIELD]
            )
            row[f"confidence_{variant}"] = float(
                np.max(probabilities[index])
            )
            for label_index, label in enumerate(LABELS):
                row[f"probability_{variant}_{label}"] = float(
                    probabilities[index, label_index]
                )

        rows.append(row)

    return rows


# ============================================================================
# PRÉDICTION EXTERNE
# ============================================================================

def predict_file(model: Pipeline) -> int:
    records = read_jsonl(PREDICT_FILE)
    rows: list[dict[str, str]] = []

    for index, record in enumerate(records, start=1):
        premise = record.get(PREMISE_FIELD)
        hypothesis = record.get(HYPOTHESIS_FIELD)
        if not isinstance(premise, str) or not premise.strip():
            raise ValueError(
                f"Prémisse absente dans {PREDICT_FILE}, ligne {index}."
            )
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError(
                f"Hypothèse absente dans {PREDICT_FILE}, ligne {index}."
            )
        rows.append(
            {
                PREMISE_FIELD: premise,
                HYPOTHESIS_FIELD: hypothesis,
            }
        )

    pairs = pd.DataFrame(rows)
    probabilities = align_probabilities(
        model,
        model.predict_proba(pairs),
    )
    predicted_columns = np.argmax(probabilities, axis=1)

    output: list[dict[str, Any]] = []
    for index, (record, column) in enumerate(
        zip(records, predicted_columns),
        start=1,
    ):
        row: dict[str, Any] = {"prediction_index": index}
        if ID_FIELD in record:
            row[ID_FIELD] = record[ID_FIELD]
        row[PREMISE_FIELD] = record[PREMISE_FIELD]
        row[HYPOTHESIS_FIELD] = record[HYPOTHESIS_FIELD]
        row["predicted_label"] = LABELS[int(column)]
        row["confidence"] = float(probabilities[index - 1, column])
        row["probabilities"] = {
            label: float(probabilities[index - 1, label_index])
            for label_index, label in enumerate(LABELS)
        }
        output.append(row)

    write_jsonl(OUTPUT_DIR / "12_external_predictions.jsonl", output)
    return len(output)


# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================

def main() -> int:
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if not INPUT_FILE.exists():
            raise FileNotFoundError(
                f"Corpus introuvable : {INPUT_FILE.resolve()}"
            )

        records = read_jsonl(INPUT_FILE)
        validate_records(records)

        ids = np.asarray(
            [str(record[ID_FIELD]) for record in records],
            dtype=object,
        )
        labels = np.asarray(
            [record[LABEL_FIELD] for record in records],
            dtype=object,
        )
        premises = np.asarray(
            [record[PREMISE_FIELD] for record in records],
            dtype=object,
        )
        hypotheses = np.asarray(
            [record[HYPOTHESIS_FIELD] for record in records],
            dtype=object,
        )
        pairs = pd.DataFrame(
            {
                PREMISE_FIELD: premises,
                HYPOTHESIS_FIELD: hypotheses,
            }
        )
        normalized_premises = np.asarray(
            [normalize_text(text) for text in premises],
            dtype=object,
        )

        groups, group_diagnostics = build_groups(records)
        split_specs, split_audit = build_fixed_splits(labels, groups, ids)

        print("Validation croisée FLEXID premise + hypothesis...")
        print(f"Instances : {len(records)}")
        print(f"Groupes   : {len(np.unique(groups))}")
        print(f"sklearn   : {sklearn.__version__}")
        print()

        results: dict[str, dict[str, Any]] = {}
        all_fold_rows: list[dict[str, Any]] = []

        if RUN_HYPOTHESIS_ONLY_CONTROL:
            print("- Contrôle hypothesis-only...")
            result = evaluate_single_text_variant(
                VARIANT_HYPOTHESIS,
                hypotheses,
                labels,
                groups,
                split_specs,
            )
            results[VARIANT_HYPOTHESIS] = result
            all_fold_rows.extend(result["fold_rows"])

        if RUN_PREMISE_ONLY_CONTROL:
            print("- Contrôle premise-only...")
            result = evaluate_single_text_variant(
                VARIANT_PREMISE,
                premises,
                labels,
                groups,
                split_specs,
            )
            results[VARIANT_PREMISE] = result
            all_fold_rows.extend(result["fold_rows"])

        print("- Modèle premise + hypothesis...")
        full_result, shuffled_result, shuffle_map_rows = (
            evaluate_full_and_shuffled(
                pairs,
                labels,
                groups,
                ids,
                normalized_premises,
                split_specs,
            )
        )
        results[VARIANT_FULL] = full_result
        all_fold_rows.extend(full_result["fold_rows"])

        if RUN_SHUFFLED_PREMISE_CONTROL:
            results[VARIANT_SHUFFLED] = shuffled_result
            all_fold_rows.extend(shuffled_result["fold_rows"])
            write_jsonl(
                OUTPUT_DIR / "04_shuffled_premise_map.jsonl",
                shuffle_map_rows,
            )

        label_counts = Counter(labels.tolist())
        majority_label, majority_count = label_counts.most_common(1)[0]
        majority_accuracy = majority_count / len(records)

        diagnostics = {
            variant: variant_diagnostics(variant, labels, result)
            for variant, result in results.items()
        }

        comparison_rows: list[dict[str, Any]] = []
        for variant, result in results.items():
            row = {
                "variant": variant,
                **result["scores"],
                "delta_vs_majority": (
                    result["scores"]["accuracy"] - majority_accuracy
                ),
            }
            comparison_rows.append(row)

        full_accuracy = results[VARIANT_FULL]["scores"]["accuracy"]
        hypothesis_accuracy = (
            results.get(VARIANT_HYPOTHESIS, {})
            .get("scores", {})
            .get("accuracy")
        )
        premise_accuracy = (
            results.get(VARIANT_PREMISE, {})
            .get("scores", {})
            .get("accuracy")
        )
        shuffled_accuracy = (
            results.get(VARIANT_SHUFFLED, {})
            .get("scores", {})
            .get("accuracy")
        )

        paired_bootstrap: dict[str, Any] = {}
        if VARIANT_HYPOTHESIS in results:
            paired_bootstrap["correct_pair_minus_hypothesis_only"] = (
                paired_group_bootstrap_accuracy(
                    labels,
                    results[VARIANT_FULL]["predictions"],
                    results[VARIANT_HYPOTHESIS]["predictions"],
                    groups,
                )
            )
        if VARIANT_SHUFFLED in results:
            paired_bootstrap["correct_pair_minus_shuffled_premise"] = (
                paired_group_bootstrap_accuracy(
                    labels,
                    results[VARIANT_FULL]["predictions"],
                    results[VARIANT_SHUFFLED]["predictions"],
                    groups,
                )
            )

        relational_audit = {
            "full_accuracy": full_accuracy,
            "hypothesis_only_accuracy": hypothesis_accuracy,
            "premise_only_accuracy": premise_accuracy,
            "shuffled_premise_accuracy": shuffled_accuracy,
            "full_gain_over_hypothesis_only": (
                None
                if hypothesis_accuracy is None
                else full_accuracy - hypothesis_accuracy
            ),
            "full_gain_over_premise_only": (
                None
                if premise_accuracy is None
                else full_accuracy - premise_accuracy
            ),
            "drop_when_premise_is_shuffled": (
                None
                if shuffled_accuracy is None
                else full_accuracy - shuffled_accuracy
            ),
            "paired_group_bootstrap": paired_bootstrap,
        }

        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "input_file": str(INPUT_FILE.resolve()),
            "input_sha256": file_sha256(INPUT_FILE),
            "instances": len(records),
            "labels": dict(label_counts),
            "groups": group_diagnostics,
            "validation": {
                "outer_folds": OUTER_FOLDS,
                "outer_seeds": OUTER_SEEDS,
                "inner_folds": INNER_FOLDS,
                "fixed_splits_reused_across_variants": True,
            },
            "baselines": {
                "majority_label": majority_label,
                "majority_accuracy": majority_accuracy,
                "uniform_random_accuracy": 1 / len(LABELS),
                "stratified_random_expected_accuracy": sum(
                    (label_counts[label] / len(records)) ** 2
                    for label in LABELS
                ),
            },
            "variant_results": diagnostics,
            "relational_audit": relational_audit,
            "software": {
                "python": sys.version,
                "scikit_learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }

        write_json(OUTPUT_DIR / "01_summary.json", summary)
        pd.DataFrame(all_fold_rows).to_csv(
            OUTPUT_DIR / "02_fold_metrics.csv",
            index=False,
            encoding="utf-8",
        )
        write_jsonl(
            OUTPUT_DIR / "03_oof_predictions.jsonl",
            build_oof_rows(records, groups, results),
        )
        write_json(
            OUTPUT_DIR / "09_fixed_split_audit.json",
            {
                "splits": split_audit,
                "variants": list(results),
                "strictly_identical_outer_and_inner_splits": True,
            },
        )
        pd.DataFrame(comparison_rows).to_csv(
            OUTPUT_DIR / "10_variant_comparison.csv",
            index=False,
            encoding="utf-8",
        )
        write_json(
            OUTPUT_DIR / "10_variant_comparison.json",
            {
                "comparison": comparison_rows,
                "relational_audit": relational_audit,
            },
        )

        for variant, result in results.items():
            write_confusion_matrix(
                variant,
                labels,
                result["predictions"],
            )

        print("Entraînement du modèle complet final...")
        final_model, final_metadata = fit_final_full_model(
            pairs,
            labels,
            groups,
        )
        write_json(
            OUTPUT_DIR / "06_top_features_full.json",
            top_features(final_model),
        )
        write_json(
            OUTPUT_DIR / "07_final_model_metadata.json",
            final_metadata,
        )
        final_features = final_model.named_steps["features"]
        joblib.dump(
            {
                # Le bundle ne contient que des objets sklearn/scipy standards.
                # Il reste donc chargeable sans devoir importer cette classe
                # personnalisée sous le nom __main__.
                "word_vectorizer": final_features.word_vectorizer_,
                "char_vectorizer": final_features.char_vectorizer_,
                "classifier": final_model.named_steps["classifier"],
                "labels": LABELS,
                "premise_field": PREMISE_FIELD,
                "hypothesis_field": HYPOTHESIS_FIELD,
                "feature_order": [
                    "premise_word",
                    "hypothesis_word",
                    "absdiff_word",
                    "overlap_word",
                    "premise_char",
                    "hypothesis_char",
                    "absdiff_char",
                    "overlap_char",
                ],
                "metadata": final_metadata,
                "group_diagnostics": group_diagnostics,
            },
            OUTPUT_DIR / "08_final_full_model.joblib",
            compress=3,
        )

        if RUN_EXTERNAL_PREDICTION:
            if PREDICT_FILE.exists():
                count = predict_file(final_model)
                print(f"Prédictions externes : {count}")
            else:
                print(
                    "Prédiction externe ignorée, fichier absent : "
                    f"{PREDICT_FILE}"
                )

        print()
        print("=" * 76)
        print("RÉSULTATS FLEXID — PRÉMISSE + HYPOTHÈSE")
        print("=" * 76)
        print(f"Baseline majoritaire : {majority_accuracy:.4f}")
        for row in comparison_rows:
            print(
                f"{row['variant']:<27} "
                f"accuracy={row['accuracy']:.4f}  "
                f"balanced={row['balanced_accuracy']:.4f}  "
                f"macro-F1={row['macro_f1']:.4f}"
            )
        print("-" * 76)

        gain_hypothesis = relational_audit["full_gain_over_hypothesis_only"]
        drop_shuffled = relational_audit["drop_when_premise_is_shuffled"]
        if gain_hypothesis is not None:
            print(
                "Gain du modèle complet sur hypothesis-only : "
                f"{gain_hypothesis:+.4f}"
            )
        if drop_shuffled is not None:
            print(
                "Chute avec prémisses mélangées            : "
                f"{drop_shuffled:+.4f}"
            )
        print(f"Résultats : {OUTPUT_DIR.resolve()}")
        print("=" * 76)

        return 0

    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
