#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FLEXID — audit de sensibilité aux prénoms (name-sensitivity audit)

Usage prévu:
    Ouvrir ce fichier dans VS Code puis cliquer sur "Run Python File".

Le script:
1) localise automatiquement le dépôt FLEXID;
2) charge les splits officiels v3;
3) réalise un audit descriptif d'association prénom/label sur l'ensemble du corpus;
4) construit un challenge set CONTREFACTUEL sur le test officiel:
       même prémisse + mêmes faits + même label gold,
       seul le prénom d'une personne est remplacé;
5) utilise uniquement l'inventaire manuel collé dans NAME_INVENTORY_JSON;
   les substitutions restent dans la même catégorie grammaticale afin d'éviter
   de modifier les accords français;
6) exclut automatiquement les cas où le prénom/nom/sexe/identité est lui-même
   juridiquement pertinent;
7) évalue les 3 checkpoints mDeBERTa joint (2026/2027/2028), s'ils sont présents;
8) rapporte flip rate, consistency, variation des probabilités, accuracy des variantes,
   harmful/beneficial flips, ainsi que des IC bootstrap groupés par cas de base.

IMPORTANT:
- Cet audit mesure la sensibilité lexicale aux prénoms.
- Il ne prétend PAS constituer un audit complet de fairness démographique.
- Aucun fichier source FLEXID n'est modifié.
- Aucun hyperparamètre n'est choisi à partir des résultats du challenge.
"""

from __future__ import annotations

import csv
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
# CONFIGURATION FIGÉE
# ============================================================================

AUDIT_VERSION = "FLEXID-NAME-SENSITIVITY-v1.1-MANUAL-INVENTORY"
AUDIT_SEED = 2026

SPLIT_RELATIVE_DIR = Path("data") / "flexid_exact_group_split"
EXPECTED_SPLIT_SIZES = {"train": 701, "validation": 151, "test": 150}
EXPECTED_LABELS = ("entailment", "contradiction", "neutral")

OUTPUT_DIR_NAME = "fairness_name_sensitivity_audit_v1_1"

# Nombre maximal de CAS DE BASE par label dans le challenge.
# Le script prend le minimum disponible entre les trois labels pour rester équilibré.
TARGET_BASE_CASES_PER_LABEL = 20

# Nombre de substitutions de prénom pour chaque cas de base.
VARIANTS_PER_CASE = 4

# Bootstrap au niveau du CAS DE BASE, pas au niveau des variantes corrélées.
BOOTSTRAP_REPS = 5000
PERMUTATION_REPS = 5000

# Termes qui rendent le prénom / nom / sexe / identité potentiellement
# juridiquement pertinent. Ces cas sont volontairement exclus.
EXCLUSION_PATTERNS = (
    "prénom",
    "prénoms",
    "changer de nom",
    "changement de nom",
    "nom de famille",
    "nom patronymique",
    "patronyme",
    "sexe revendiqué",
    "mention relative à son sexe",
    "modification de son sexe",
    "modification du sexe",
    "identité soit préservée",
    "secret de son identité",
)

# ============================================================================
# INVENTAIRE MANUEL DES PRÉNOMS — À REMPLIR PAR TOI
# ============================================================================
#
# COLLE ICI la sortie JSON obtenue après inspection du corpus FLEXID.
# Tu peux coller soit le JSON brut, soit le bloc ```json ... ```.
#
# Le script n'ajoute AUCUN prénom et n'essaie PAS de deviner des noms.
# Les prénoms de "ambiguous_or_uncertain" sont détectés mais JAMAIS utilisés
# comme cas de base ni comme remplacements.
#
# Format attendu EXACT:
# {
#   "masculine": [...],
#   "feminine": [...],
#   "ambiguous_or_uncertain": [...]
# }
#
NAME_INVENTORY_JSON = r"""
{
  "masculine": [
    "Abdourazak",
    "Abel",
    "Adam",
    "Adil",
    "Ahmed",
    "Alejandro",
    "Amadou",
    "Amaury",
    "Amine",
    "Amir",
    "Andrés",
    "Anis",
    "Antoine",
    "Armand",
    "Arsène",
    "Augustin",
    "Aurélien",
    "Ayoub",
    "Azzedine",
    "Balthazar",
    "Bastien",
    "Benjamin",
    "Benoît",
    "Bernard",
    "Bertrand",
    "Bilal",
    "Boris",
    "Brice",
    "Bruno",
    "Caio",
    "Charles",
    "Christophe",
    "Césaire",
    "César",
    "Côme",
    "Daiki",
    "Damien",
    "Didier",
    "Diego",
    "Dimitri",
    "Donghyun",
    "Dongmin",
    "Dragan",
    "Désiré",
    "Emiliano",
    "Emmanuel",
    "Enguerrand",
    "Esteban",
    "Ewen",
    "Farid",
    "Felipe",
    "Firmin",
    "Florent",
    "Félix",
    "Gaston",
    "Gautier",
    "Grégoire",
    "Gustavo",
    "Gérald",
    "Hadrien",
    "Hakim",
    "Hamza",
    "Hans",
    "Haruto",
    "Hector",
    "Henrique",
    "Hiroshi",
    "Honoré",
    "Hyunwoo",
    "Ibrahim",
    "Idriss",
    "Idrissa",
    "Ignacio",
    "Ilyes",
    "Iskander",
    "Issam",
    "Jacques",
    "Jean",
    "Joachim",
    "Joaquín",
    "Joonho",
    "Joris",
    "João",
    "Julien",
    "Junseo",
    "Justin",
    "Kaito",
    "Kamel",
    "Kilian",
    "Kossi",
    "Kota",
    "Kwame",
    "Laurent",
    "Lazare",
    "Lucas",
    "Léandre",
    "Malik",
    "Malo",
    "Mamadou",
    "Marcel",
    "Marius",
    "Martin",
    "Marwan",
    "Mateo",
    "Mehdi",
    "Minho",
    "Minjun",
    "Mohamed",
    "Mourad",
    "Moussa",
    "Murilo",
    "Nabil",
    "Nadir",
    "Nassim",
    "Naël",
    "Noé",
    "Numa",
    "Octave",
    "Octavien",
    "Osman",
    "Pacôme",
    "Pierrick",
    "Rachid",
    "Rafael",
    "Rami",
    "Raphaël",
    "Rayan",
    "Riad",
    "Riku",
    "Rodrigo",
    "Ruben",
    "Rémy",
    "Salim",
    "Salissou",
    "Sami",
    "Samir",
    "Samuel",
    "Samy",
    "Santiago",
    "Selim",
    "Seojun",
    "Seungwoo",
    "Silvère",
    "Sofiane",
    "Sota",
    "Souleymane",
    "Sélim",
    "Taeyang",
    "Takumi",
    "Thiago",
    "Thomas",
    "Théodore",
    "Tiago",
    "Timothée",
    "Tom",
    "Ulysse",
    "Vadim",
    "Valentin",
    "Vinícius",
    "Walid",
    "Wandrille",
    "William",
    "Xavier",
    "Yacine",
    "Yanis",
    "Yassine",
    "Younès",
    "Youssef",
    "Yuto",
    "Yvan",
    "Zakaria",
    "Zinedine",
    "Zéphyr",
    "Éloi",
    "Émeric",
    "Émile",
    "Éric"
  ],
  "feminine": [
    "Abena",
    "Abril",
    "Adelaïde",
    "Adeline",
    "Akari",
    "Akiko",
    "Akouvi",
    "Albane",
    "Alice",
    "Aline",
    "Aliénor",
    "Amina",
    "Amélie",
    "Anissa",
    "Araceli",
    "Ariane",
    "Audrey",
    "Aurore",
    "Ayaka",
    "Azucena",
    "Aïcha",
    "Aïssata",
    "Aïssatou",
    "Beatriz",
    "Bertille",
    "Blanche",
    "Bruna",
    "Béatrice",
    "Bérénice",
    "Camila",
    "Capucine",
    "Carine",
    "Carolina",
    "Cecília",
    "Chaewon",
    "Chloé",
    "Christine",
    "Citlali",
    "Clara",
    "Coline",
    "Constance",
    "Célestine",
    "Daniela",
    "Daphné",
    "Delphine",
    "Diane",
    "Domitille",
    "Dorcas",
    "Dulce",
    "Débora",
    "Déborah",
    "Elena",
    "Elvire",
    "Emelyne",
    "Emi",
    "Esmeralda",
    "Estelle",
    "Esther",
    "Eugénie",
    "Eulalie",
    "Eunji",
    "Eva",
    "Fabiana",
    "Fabiola",
    "Farah",
    "Fatou",
    "Faustine",
    "Fernanda",
    "Flavie",
    "Frida",
    "Félicie",
    "Gaeun",
    "Garance",
    "Gloria",
    "Graziella",
    "Haeun",
    "Hana",
    "Hortense",
    "Hyejin",
    "Hyeri",
    "Héloïse",
    "Hélène",
    "Imane",
    "Inès",
    "Inês",
    "Irène",
    "Isabela",
    "Isaline",
    "Itzel",
    "Josiane",
    "Joséphine",
    "Joëlle",
    "Juliana",
    "Julie",
    "Juliette",
    "Kaori",
    "Keiko",
    "Larissa",
    "Letícia",
    "Leïla",
    "Lily",
    "Linda",
    "Livia",
    "Louise",
    "Luana",
    "Lucie",
    "Léonie",
    "Léontine",
    "Mahaut",
    "Maicha",
    "Manon",
    "Manuela",
    "Mariam",
    "Mariana",
    "Marina",
    "Marisol",
    "Maritza",
    "Martine",
    "Mathilde",
    "Maya",
    "Maëlle",
    "Maïa",
    "Meryem",
    "Minji",
    "Miyu",
    "Momoka",
    "Montserrat",
    "Muriel",
    "Mélisande",
    "Nadine",
    "Nadège",
    "Nanami",
    "Naomie",
    "Nari",
    "Nayeli",
    "Naëlle",
    "Nora",
    "Noémie",
    "Olympe",
    "Oriane",
    "Paloma",
    "Patrícia",
    "Paulina",
    "Prisca",
    "Priscila",
    "Priscille",
    "Quitterie",
    "Rahel",
    "Rassikatou",
    "Renata",
    "Rina",
    "Rosalie",
    "Roxane",
    "Ruth",
    "Saba",
    "Sakura",
    "Salma",
    "Sama",
    "Samia",
    "Samira",
    "Sandrine",
    "Sarah",
    "Sayaka",
    "Shiori",
    "Sibylle",
    "Socorro",
    "Sofia",
    "Solange",
    "Solène",
    "Soojin",
    "Sophie",
    "Séraphine",
    "Tainá",
    "Thaïs",
    "Tiphaine",
    "Tomoko",
    "Valeria",
    "Victoire",
    "Violette",
    "Virginie",
    "Vitória",
    "Wassila",
    "Ximena",
    "Yara",
    "Yasmine",
    "Yeji",
    "Yejin",
    "Yerin",
    "Yetunde",
    "Yui",
    "Yuna",
    "Yvette",
    "Zélie",
    "Élodie",
    "Éva"
  ],
  "ambiguous_or_uncertain": [
    "Aboudourazakou",
    "Adhiambo",
    "Adingra",
    "Akinyi",
    "Akliouch",
    "Akoth",
    "Alex",
    "Almaz",
    "Anyango",
    "Aoi",
    "Atieno",
    "Awali",
    "Ayeva",
    "Bekele",
    "Bethlehem",
    "Bih",
    "Birhanu",
    "Bora",
    "Busisiwe",
    "Camille",
    "Chancelle",
    "Charlie",
    "Chebet",
    "Cherono",
    "Chihiro",
    "Chinedu",
    "Cyrille",
    "Dawit",
    "Divine",
    "Eden",
    "Eposi",
    "Espérance",
    "Ewube",
    "Faouzane",
    "Faouziatou",
    "Grâce",
    "Guadalupe",
    "Haruka",
    "Henok",
    "Hikari",
    "Hirut",
    "Hlengiwe",
    "Jepkemoi",
    "Jisoo",
    "Kahambu",
    "Kalkidan",
    "Ketsia",
    "Lemlem",
    "Li",
    "Liang",
    "Lindiwe",
    "Mademba",
    "Mado",
    "Mahlet",
    "Makosso",
    "Manka",
    "Manyi",
    "Maxence",
    "Mekdes",
    "Meron",
    "Merveille",
    "Meseret",
    "Misaki",
    "Morgan",
    "Mumbi",
    "Muthoni",
    "Naliaka",
    "Nandi",
    "Natsuki",
    "Ngono",
    "Ngozi",
    "Ngum",
    "Njeri",
    "Njoki",
    "Noluthando",
    "Nomfundo",
    "Nompumelelo",
    "Nomvula",
    "Nozomi",
    "Ntombifuthi",
    "Nyambura",
    "Olumide",
    "Phumzile",
    "Ren",
    "Robin",
    "Sacha",
    "Sam",
    "Sani",
    "Sekoulou",
    "Senait",
    "Seoyeon",
    "Seulgi",
    "Sibongile",
    "Siphokazi",
    "Slindile",
    "Sumin",
    "Sètondji",
    "Tadesse",
    "Tesfaye",
    "Tetereou",
    "Thandeka",
    "Thembisile",
    "Thulisile",
    "Tigist",
    "Tinah",
    "Toundé",
    "Tshala",
    "Wacera",
    "Wahabou",
    "Wairimu",
    "Wambui",
    "Wangari",
    "Wanjiku",
    "Wanjiru",
    "Yannick",
    "Yordanos",
    "Zanele",
    "Zinhle",
    "Zodwa",
    "Zouberatou"
  ]
}
"""


def _extract_json_object(raw: str) -> dict:
    """Accepte JSON brut ou bloc Markdown ```json ... ```."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Tolère une courte phrase autour du JSON, mais n'invente rien.
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first < 0 or last < first:
        raise RuntimeError(
            "NAME_INVENTORY_JSON ne contient pas d'objet JSON { ... }."
        )
    cleaned = cleaned[first:last + 1]

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"NAME_INVENTORY_JSON invalide: {exc}"
        ) from exc

    if not isinstance(obj, dict):
        raise RuntimeError("L'inventaire des prénoms doit être un objet JSON.")
    return obj


def _clean_name_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, list):
        raise RuntimeError(f"{key!r} doit être une liste JSON.")

    names = []
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise RuntimeError(f"{key}[{i}] doit être une chaîne.")
        name = item.strip()
        if not name:
            raise RuntimeError(f"{key}[{i}] est vide.")
        if name != item:
            raise RuntimeError(
                f"{key}[{i}] contient des espaces autour de {item!r}. "
                "Conserve l'orthographe exacte du corpus."
            )
        names.append(name)

    duplicates = sorted(name for name, n in Counter(names).items() if n > 1)
    if duplicates:
        raise RuntimeError(
            f"Doublons dans {key}: {duplicates}. Supprime-les du JSON."
        )
    return names


def load_manual_name_inventory() -> dict[str, list[str]]:
    raw = _extract_json_object(NAME_INVENTORY_JSON)

    expected = {"masculine", "feminine", "ambiguous_or_uncertain"}
    missing = sorted(expected - set(raw))
    extra = sorted(set(raw) - expected)
    if missing:
        raise RuntimeError(f"Clés manquantes dans le JSON: {missing}")
    if extra:
        raise RuntimeError(
            f"Clés inattendues dans le JSON: {extra}. "
            f"Clés admises: {sorted(expected)}"
        )

    inventory = {
        key: _clean_name_list(raw[key], key)
        for key in ("masculine", "feminine", "ambiguous_or_uncertain")
    }

    # Les trois listes doivent être disjointes.
    owner = {}
    overlaps = []
    for key, names in inventory.items():
        for name in names:
            if name in owner:
                overlaps.append((name, owner[name], key))
            else:
                owner[name] = key
    if overlaps:
        raise RuntimeError(
            "Un même prénom apparaît dans plusieurs catégories: "
            + "; ".join(f"{n!r} ({a}/{b})" for n, a, b in overlaps)
        )

    if not inventory["masculine"] and not inventory["feminine"]:
        raise RuntimeError(
            "Les listes masculine et feminine sont toutes deux vides. "
            "Colle d'abord l'inventaire réel des prénoms FLEXID."
        )
    return inventory


MANUAL_NAME_INVENTORY = load_manual_name_inventory()

# Catégories grammaticales utilisées pour conserver les accords français.
NAME_TO_CATEGORY = {
    name: "masculine" for name in MANUAL_NAME_INVENTORY["masculine"]
}
NAME_TO_CATEGORY.update({
    name: "feminine" for name in MANUAL_NAME_INVENTORY["feminine"]
})

AMBIGUOUS_NAMES = set(MANUAL_NAME_INVENTORY["ambiguous_or_uncertain"])
ALL_INVENTORY_NAMES = set(NAME_TO_CATEGORY) | AMBIGUOUS_NAMES

# Les pools de remplacement viennent UNIQUEMENT de l'inventaire fourni.
# Aucun prénom externe n'est ajouté.
REPLACEMENT_POOLS = {
    "masculine": tuple(MANUAL_NAME_INVENTORY["masculine"]),
    "feminine": tuple(MANUAL_NAME_INVENTORY["feminine"]),
}


# ============================================================================
# I/O ET DÉCOUVERTE DU PROJET
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
                raise RuntimeError(f"{path}:{line_no}: JSON invalide: {exc}") from exc
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
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


def find_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [Path.cwd().resolve(), script_dir, *script_dir.parents]
    for candidate in candidates:
        split_dir = candidate / SPLIT_RELATIVE_DIR
        if all((split_dir / f"{name}.jsonl").is_file()
               for name in EXPECTED_SPLIT_SIZES):
            return candidate
    raise RuntimeError(
        "Impossible de trouver data/flexid_exact_group_split. "
        "Ouvre le dépôt FLEXID comme dossier VS Code."
    )


def unique_output_dir(project_root: Path) -> Path:
    base = project_root / "data" / OUTPUT_DIR_NAME
    if not base.exists():
        return base
    for i in range(2, 100):
        candidate = project_root / "data" / f"{OUTPUT_DIR_NAME}_run{i:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Trop de dossiers d'audit existants.")


# ============================================================================
# DÉTECTION ET SUBSTITUTION DE PRÉNOMS
# ============================================================================

def token_present(text: str, name: str) -> bool:
    # Frontières Unicode simples: le caractère avant/après ne doit pas être
    # alphanumérique ni underscore.
    import re
    pattern = rf"(?<![\wÀ-ÖØ-öø-ÿ]){re.escape(name)}(?![\wÀ-ÖØ-öø-ÿ])"
    return re.search(pattern, text) is not None


def replace_name(text: str, old: str, new: str) -> str:
    import re
    pattern = rf"(?<![\wÀ-ÖØ-öø-ÿ]){re.escape(old)}(?![\wÀ-ÖØ-öø-ÿ])"
    replaced, count = re.subn(pattern, new, text)
    require(count >= 1, f"Substitution impossible: {old!r} absent du texte.")
    return replaced


def detected_inventory_names(text: str) -> list[str]:
    """Détecte uniquement les prénoms fournis manuellement dans le JSON."""
    return sorted(name for name in ALL_INVENTORY_NAMES if token_present(text, name))


def excluded_semantic_context(text: str) -> bool:
    folded = text.casefold()
    return any(pattern.casefold() in folded for pattern in EXCLUSION_PATTERNS)


def eligible_record(row: dict) -> tuple[bool, str | None, str | None, str]:
    text = row.get("hypothesis_facts", "")
    hits = detected_inventory_names(text)

    if len(hits) == 0:
        return False, None, None, "no_inventory_name"

    # Un prénom déclaré ambigu/incertain suffit à exclure le cas.
    ambiguous_hits = [name for name in hits if name in AMBIGUOUS_NAMES]
    if ambiguous_hits:
        return False, None, None, "ambiguous_or_uncertain_name"

    active_hits = [name for name in hits if name in NAME_TO_CATEGORY]
    if len(active_hits) != 1 or len(hits) != 1:
        return False, None, None, "multiple_inventory_names"

    name = active_hits[0]
    category = NAME_TO_CATEGORY[name]

    if excluded_semantic_context(text):
        return False, name, category, "name_legally_relevant_context"

    return True, name, category, "eligible"


def deterministic_order(values: list[str], key: str) -> list[str]:
    def h(value: str) -> str:
        raw = f"{AUDIT_SEED}|{key}|{value}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()
    return sorted(values, key=h)


def choose_replacements(original: str, category: str, base_id: str) -> list[str]:
    pool = [x for x in REPLACEMENT_POOLS[category] if x != original]
    pool = deterministic_order(pool, base_id)
    require(len(pool) >= VARIANTS_PER_CASE, "Pool de remplacements trop petit.")
    return pool[:VARIANTS_PER_CASE]



def validate_manual_inventory_against_corpus(all_rows: list[dict]) -> dict:
    """
    Vérifie que chaque prénom fourni apparaît réellement comme token entier
    dans au moins un hypothesis_facts FLEXID.
    """
    hypotheses = [str(row.get("hypothesis_facts", "")) for row in all_rows]
    counts = {}
    absent = []

    for name in sorted(ALL_INVENTORY_NAMES):
        count = sum(token_present(text, name) for text in hypotheses)
        counts[name] = count
        if count == 0:
            absent.append(name)

    if absent:
        raise RuntimeError(
            "Le JSON contient des prénoms absents de hypothesis_facts dans "
            f"le corpus officiel: {absent}. Vérifie la sortie avant l'audit."
        )

    return {
        "inventory_source": "manual JSON pasted into NAME_INVENTORY_JSON",
        "masculine_count": len(MANUAL_NAME_INVENTORY["masculine"]),
        "feminine_count": len(MANUAL_NAME_INVENTORY["feminine"]),
        "ambiguous_or_uncertain_count":
            len(MANUAL_NAME_INVENTORY["ambiguous_or_uncertain"]),
        "total_unique_names": len(ALL_INVENTORY_NAMES),
        "occurrences_by_name": counts,
        "all_listed_names_found_in_hypothesis_facts": True,
    }


# ============================================================================
# STATISTIQUES D'ASSOCIATION DATASET PRÉNOM/LABEL
# ============================================================================

def contingency_chi2(rows: list[tuple[str, str]]) -> tuple[float, int, int]:
    groups = sorted({g for g, _ in rows})
    labels = list(EXPECTED_LABELS)
    counts = {g: Counter() for g in groups}
    for group, label in rows:
        counts[group][label] += 1

    n = len(rows)
    group_totals = {g: sum(counts[g].values()) for g in groups}
    label_totals = Counter(label for _, label in rows)

    chi2 = 0.0
    for g in groups:
        for label in labels:
            expected = group_totals[g] * label_totals[label] / n
            if expected > 0:
                observed = counts[g][label]
                chi2 += (observed - expected) ** 2 / expected
    return chi2, len(groups), len(labels)


def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    denom = n * min(r - 1, c - 1)
    return math.sqrt(chi2 / denom) if denom > 0 else 0.0


def permutation_association(rows: list[tuple[str, str]], reps: int, seed: int) -> dict:
    require(rows, "Aucune ligne pour le test d'association.")
    observed, r, c = contingency_chi2(rows)
    groups = [g for g, _ in rows]
    labels = [y for _, y in rows]
    rng = random.Random(seed)
    exceed = 0
    shuffled = labels[:]
    for _ in range(reps):
        rng.shuffle(shuffled)
        stat, _, _ = contingency_chi2(list(zip(groups, shuffled)))
        exceed += int(stat >= observed - 1e-12)
    p = (exceed + 1) / (reps + 1)
    return {
        "n": len(rows),
        "groups": r,
        "labels": c,
        "chi_square_statistic": observed,
        "cramers_v": cramers_v(observed, len(rows), r, c),
        "permutation_reps": reps,
        "permutation_p_value": p,
    }


def dataset_name_association(all_rows: list[dict]) -> dict:
    eligible = []
    exclusions = Counter()
    for row in all_rows:
        ok, name, category, reason = eligible_record(row)
        exclusions[reason] += 1
        if ok:
            eligible.append((row, name, category))

    # Catégorie grammaticale fournie manuellement (masculine/feminine) vs label.
    category_pairs = [(category, row["label"]) for row, _, category in eligible]

    # Nom exact vs label: garder seulement les prénoms présents >= 3 fois.
    freq = Counter(name for _, name, _ in eligible)
    exact_pairs = [
        (name, row["label"])
        for row, name, _ in eligible
        if freq[name] >= 3
    ]

    by_category = {
        cat: dict(Counter(row["label"] for row, _, c in eligible if c == cat))
        for cat in ("feminine", "masculine")
    }
    by_name = {
        name: dict(Counter(row["label"] for row, n, _ in eligible if n == name))
        for name in sorted(freq)
        if freq[name] >= 3
    }

    return {
        "eligible_instances": len(eligible),
        "eligibility_reasons": dict(exclusions),
        "category_label_counts": by_category,
        "exact_name_label_counts_min_frequency_3": by_name,
        "category_label_permutation_test":
            permutation_association(category_pairs, PERMUTATION_REPS, AUDIT_SEED)
            if len(set(g for g, _ in category_pairs)) >= 2 else None,
        "exact_name_label_permutation_test_min_frequency_3":
            permutation_association(exact_pairs, PERMUTATION_REPS, AUDIT_SEED + 1)
            if len(set(g for g, _ in exact_pairs)) >= 2 else None,
        "interpretation_warning":
            "These tests describe association in the manually reviewed FLEXID name inventory. "
            "They do not establish demographic identity or comprehensive fairness.",
    }


# ============================================================================
# CONSTRUCTION DU CHALLENGE SET OFFICIEL
# ============================================================================

def select_balanced_test_cases(test_rows: list[dict]) -> tuple[list[dict], dict]:
    candidates = defaultdict(list)
    exclusion_reasons = Counter()

    for row in test_rows:
        ok, name, category, reason = eligible_record(row)
        exclusion_reasons[reason] += 1
        if not ok:
            continue
        enriched = dict(row)
        enriched["_audit_name"] = name
        enriched["_audit_category"] = category
        candidates[row["label"]].append(enriched)

    counts = {label: len(candidates[label]) for label in EXPECTED_LABELS}
    require(all(counts[label] > 0 for label in EXPECTED_LABELS),
            f"Aucun cas éligible pour au moins un label: {counts}")

    n_per_label = min(TARGET_BASE_CASES_PER_LABEL, *(counts[label] for label in EXPECTED_LABELS))
    require(n_per_label >= 5,
            f"Trop peu de cas de base équilibrés pour un audit utile: {counts}")

    selected = []
    for label in EXPECTED_LABELS:
        ordered = sorted(
            candidates[label],
            key=lambda row: hashlib.sha256(
                f"{AUDIT_SEED}|{row['id']}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(ordered[:n_per_label])

    selected.sort(key=lambda row: row["id"])
    return selected, {
        "eligible_test_by_label": counts,
        "selected_per_label": n_per_label,
        "selected_total": len(selected),
        "exclusion_reasons_on_test": dict(exclusion_reasons),
    }


def build_challenge(selected: list[dict]) -> tuple[list[dict], list[dict]]:
    base_rows = []
    variants = []

    for row in selected:
        original = row["_audit_name"]
        category = row["_audit_category"]

        base_rows.append({
            "id": row["id"],
            "premise": row["premise"],
            "hypothesis_facts": row["hypothesis_facts"],
            "label": row["label"],
            "original_name": original,
            "name_category": category,
            "law_ref": (row.get("meta") or {}).get("law_ref"),
        })

        replacements = choose_replacements(original, category, row["id"])
        for idx, replacement in enumerate(replacements, 1):
            new_hypothesis = replace_name(row["hypothesis_facts"], original, replacement)
            require(new_hypothesis != row["hypothesis_facts"],
                    f"{row['id']}: substitution sans effet.")
            variants.append({
                "id": f"{row['id']}__name_{idx:02d}",
                "base_id": row["id"],
                "variant_index": idx,
                "original_name": original,
                "replacement_name": replacement,
                "name_category": category,
                "premise": row["premise"],
                "hypothesis_facts": new_hypothesis,
                "label": row["label"],
                "law_ref": (row.get("meta") or {}).get("law_ref"),
            })

    return base_rows, variants


# ============================================================================
# BOOTSTRAP GROUPÉ PAR CAS DE BASE
# ============================================================================

def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def bootstrap_case_mean(values_by_case: dict[str, float], seed: int) -> dict:
    ids = sorted(values_by_case)
    require(ids, "Bootstrap sans cas.")
    observed = statistics.mean(values_by_case[i] for i in ids)
    rng = random.Random(seed)
    reps = []
    for _ in range(BOOTSTRAP_REPS):
        sampled = [rng.choice(ids) for _ in ids]
        reps.append(statistics.mean(values_by_case[i] for i in sampled))
    reps.sort()
    return {
        "mean": observed,
        "ci95": [percentile(reps, 0.025), percentile(reps, 0.975)],
        "bootstrap_reps": BOOTSTRAP_REPS,
        "cluster_unit": "base_case",
    }


# ============================================================================
# IMPORT DU MODÈLE mDeBERTa DÉJÀ ENTRAÎNÉ
# ============================================================================

def find_model_script(project_root: Path) -> Path:
    candidates = [
        project_root / "mdeberta_join_label_rationales.py",
        project_root / "scripts" / "mdeberta_join_label_rationales.py",
        project_root / "train_mdeberta_joint_nli_rationale_final.py",
        project_root / "scripts" / "train_mdeberta_joint_nli_rationale_final.py",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise RuntimeError(
        "Script mDeBERTa introuvable. Place mdeberta_join_label_rationales.py "
        "à la racine du dépôt ou dans scripts/."
    )


def import_model_module(path: Path):
    """
    Importe dynamiquement le script mDeBERTa.

    Python 3.13 / dataclasses exige que le module soit déjà enregistré dans
    sys.modules pendant son exécution. Sans cela, les décorateurs @dataclass
    peuvent échouer avec:
        AttributeError: 'NoneType' object has no attribute '__dict__'
    """
    module_name = "flexid_mdeberta_joint"
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None,
            f"Impossible d'importer {path}")

    module = importlib.util.module_from_spec(spec)

    # IMPORTANT pour @dataclass sous Python 3.13:
    # rendre le module visible via sys.modules AVANT exec_module().
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
    data_dir = project_root / "data"
    candidates = sorted(
        [p for p in data_dir.glob("results_mdeberta_joint_final*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for directory in candidates:
        if all((directory / f"seed_{seed}" / "best" / "model.safetensors").is_file()
               for seed in (2026, 2027, 2028)):
            return directory
    raise RuntimeError(
        "Aucun dossier mDeBERTa complet avec seed_2026/2027/2028/best trouvé."
    )


def to_model_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "id": row["id"],
            "premise": row["premise"],
            "hypothesis_facts": row["hypothesis_facts"],
        }
        for row in rows
    ]


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    return 0.5 * sum(abs(float(p[label]) - float(q[label]))
                     for label in EXPECTED_LABELS)


def evaluate_one_seed(module: Any, checkpoint: Path,
                      base_rows: list[dict], variants: list[dict],
                      seed: int) -> tuple[dict, list[dict]]:
    deps = module.dependencies()

    # Reproductibilité / CPU ou CUDA auto selon le helper du modèle.
    args = SimpleNamespace(
        device="auto",
        precision="auto",
        eval_batch_size=4,
        window_batch_budget=8,
    )
    device, precision = module.device_and_precision(args, deps.torch)

    model, tokenizer, settings = module.load_checkpoint(checkpoint, deps)
    model = model.float().to(device)

    base_model_rows = to_model_rows(base_rows)
    variant_model_rows = to_model_rows(variants)

    base_items, _ = module.prepare_records(
        base_model_rows, tokenizer, settings, labelled=False
    )
    variant_items, _ = module.prepare_records(
        variant_model_rows, tokenizer, settings, labelled=False
    )

    base_preds = module.predict_items(
        model, base_items, tokenizer, settings, args, deps, device, precision
    )
    variant_preds = module.predict_items(
        model, variant_items, tokenizer, settings, args, deps, device, precision
    )

    base_gold = {row["id"]: row["label"] for row in base_rows}
    base_pred = {pred["id"]: pred for pred in base_preds}
    variant_by_id = {row["id"]: row for row in variants}

    details = []
    case_flip = defaultdict(list)
    case_tv = defaultdict(list)
    case_abs_gold_shift = defaultdict(list)
    case_harm = defaultdict(list)
    case_benefit = defaultdict(list)
    case_variant_correct = defaultdict(list)

    for pred in variant_preds:
        meta = variant_by_id[pred["id"]]
        base_id = meta["base_id"]
        original = base_pred[base_id]
        gold = base_gold[base_id]

        orig_probs = original["label_probabilities"]
        var_probs = pred["label_probabilities"]

        flip = pred["label"] != original["label"]
        original_correct = original["label"] == gold
        variant_correct = pred["label"] == gold
        tv = total_variation(orig_probs, var_probs)
        gold_shift = float(var_probs[gold]) - float(orig_probs[gold])

        harmful = original_correct and not variant_correct
        beneficial = (not original_correct) and variant_correct

        case_flip[base_id].append(float(flip))
        case_tv[base_id].append(tv)
        case_abs_gold_shift[base_id].append(abs(gold_shift))
        case_harm[base_id].append(float(harmful))
        case_benefit[base_id].append(float(beneficial))
        case_variant_correct[base_id].append(float(variant_correct))

        details.append({
            "seed": seed,
            "base_id": base_id,
            "variant_id": pred["id"],
            "gold_label": gold,
            "original_name": meta["original_name"],
            "replacement_name": meta["replacement_name"],
            "name_category": meta["name_category"],
            "original_prediction": original["label"],
            "variant_prediction": pred["label"],
            "label_flip": int(flip),
            "original_correct": int(original_correct),
            "variant_correct": int(variant_correct),
            "harmful_flip": int(harmful),
            "beneficial_flip": int(beneficial),
            "original_gold_probability": float(orig_probs[gold]),
            "variant_gold_probability": float(var_probs[gold]),
            "gold_probability_shift": gold_shift,
            "abs_gold_probability_shift": abs(gold_shift),
            "total_variation_distance": tv,
        })

    # Agrégation au niveau cas de base.
    case_flip_mean = {k: statistics.mean(v) for k, v in case_flip.items()}
    case_tv_mean = {k: statistics.mean(v) for k, v in case_tv.items()}
    case_abs_gold_mean = {k: statistics.mean(v) for k, v in case_abs_gold_shift.items()}
    case_harm_mean = {k: statistics.mean(v) for k, v in case_harm.items()}
    case_benefit_mean = {k: statistics.mean(v) for k, v in case_benefit.items()}
    case_variant_acc = {k: statistics.mean(v) for k, v in case_variant_correct.items()}

    all_consistent = {}
    for base_id in case_flip:
        all_consistent[base_id] = float(all(x == 0.0 for x in case_flip[base_id]))

    original_accuracy = statistics.mean(
        float(base_pred[row["id"]]["label"] == row["label"])
        for row in base_rows
    )

    summary = {
        "seed": seed,
        "checkpoint": str(checkpoint),
        "device": str(device),
        "precision": precision,
        "base_cases": len(base_rows),
        "variants": len(variants),
        "original_accuracy": original_accuracy,
        "variant_accuracy": statistics.mean(case_variant_acc.values()),
        "flip_rate": bootstrap_case_mean(case_flip_mean, AUDIT_SEED + seed),
        "all_variants_consistent_with_original_rate":
            bootstrap_case_mean(all_consistent, AUDIT_SEED + seed + 11),
        "mean_total_variation_distance":
            bootstrap_case_mean(case_tv_mean, AUDIT_SEED + seed + 23),
        "mean_absolute_gold_probability_shift":
            bootstrap_case_mean(case_abs_gold_mean, AUDIT_SEED + seed + 37),
        "harmful_flip_rate":
            bootstrap_case_mean(case_harm_mean, AUDIT_SEED + seed + 41),
        "beneficial_flip_rate":
            bootstrap_case_mean(case_benefit_mean, AUDIT_SEED + seed + 53),
    }

    del model
    if device.type == "cuda":
        deps.torch.cuda.empty_cache()

    return summary, details


# ============================================================================
# AGRÉGATION MULTI-SEEDS
# ============================================================================

def seed_metric(summary: dict, key: str) -> float:
    value = summary[key]
    if isinstance(value, dict) and "mean" in value:
        return float(value["mean"])
    return float(value)


def aggregate_seed_summaries(summaries: list[dict]) -> dict:
    keys = (
        "original_accuracy",
        "variant_accuracy",
        "flip_rate",
        "all_variants_consistent_with_original_rate",
        "mean_total_variation_distance",
        "mean_absolute_gold_probability_shift",
        "harmful_flip_rate",
        "beneficial_flip_rate",
    )
    result = {}
    for key in keys:
        values = [seed_metric(s, key) for s in summaries]
        result[key] = {
            "values": values,
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return result


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    project_root = find_project_root()
    split_dir = project_root / SPLIT_RELATIVE_DIR
    output_dir = unique_output_dir(project_root)
    output_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 78)
    print("FLEXID — NAME-SENSITIVITY AUDIT")
    print("=" * 78)
    print(f"Projet          : {project_root}")
    print(f"Sortie          : {output_dir}")
    print(f"Version audit   : {AUDIT_VERSION}")
    print(f"Seed audit      : {AUDIT_SEED}")

    splits = {
        name: read_jsonl(split_dir / f"{name}.jsonl")
        for name in EXPECTED_SPLIT_SIZES
    }
    for name, expected in EXPECTED_SPLIT_SIZES.items():
        require(len(splits[name]) == expected,
                f"{name}: {len(splits[name])} != {expected}")
        require(set(row["label"] for row in splits[name]) <= set(EXPECTED_LABELS),
                f"{name}: label invalide")

    all_rows = splits["train"] + splits["validation"] + splits["test"]

    # 0) Vérification de l'inventaire manuel contre le corpus officiel.
    inventory_report = validate_manual_inventory_against_corpus(all_rows)
    write_json(output_dir / "name_inventory_used.json", MANUAL_NAME_INVENTORY)
    write_json(output_dir / "name_inventory_validation.json", inventory_report)

    print("\nInventaire manuel")
    print(f"  Masculine               : {inventory_report['masculine_count']}")
    print(f"  Feminine                : {inventory_report['feminine_count']}")
    print(f"  Ambiguous/uncertain     : {inventory_report['ambiguous_or_uncertain_count']}")
    print(f"  Total unique names      : {inventory_report['total_unique_names']}")
    print("  Tous présents corpus    : oui")

    # 1) Audit descriptif dataset.
    association = dataset_name_association(all_rows)
    write_json(output_dir / "dataset_name_label_association.json", association)

    # 2) Challenge test équilibré.
    selected, selection_report = select_balanced_test_cases(splits["test"])
    base_rows, variants = build_challenge(selected)

    write_json(output_dir / "selection_report.json", selection_report)
    write_jsonl(output_dir / "base_cases.jsonl", base_rows)
    write_jsonl(output_dir / "name_counterfactual_challenge.jsonl", variants)

    print("\nChallenge set")
    print(f"  Cas éligibles test par label : {selection_report['eligible_test_by_label']}")
    print(f"  Cas retenus par label        : {selection_report['selected_per_label']}")
    print(f"  Cas de base                  : {len(base_rows)}")
    print(f"  Variantes                    : {len(variants)}")

    # 3) Évaluation mDeBERTa joint déjà entraîné.
    model_script = find_model_script(project_root)
    result_dir = find_latest_complete_result_dir(project_root)
    module = import_model_module(model_script)

    print(f"\nScript modèle    : {model_script}")
    print(f"Checkpoints      : {result_dir}")

    seed_summaries = []
    all_details = []

    for seed in (2026, 2027, 2028):
        checkpoint = result_dir / f"seed_{seed}" / "best"
        print(f"\n--- Seed {seed} ---")
        summary, details = evaluate_one_seed(
            module, checkpoint, base_rows, variants, seed
        )
        seed_summaries.append(summary)
        all_details.extend(details)

        print(f"Original accuracy : {summary['original_accuracy']:.4f}")
        print(f"Variant accuracy  : {summary['variant_accuracy']:.4f}")
        print(f"Flip rate         : {summary['flip_rate']['mean']:.4f} "
              f"[{summary['flip_rate']['ci95'][0]:.4f}, "
              f"{summary['flip_rate']['ci95'][1]:.4f}]")
        print(f"Consistency       : "
              f"{summary['all_variants_consistent_with_original_rate']['mean']:.4f}")
        print(f"Mean TV distance  : "
              f"{summary['mean_total_variation_distance']['mean']:.4f}")
        print(f"Harmful flip rate : {summary['harmful_flip_rate']['mean']:.4f}")

    aggregate = aggregate_seed_summaries(seed_summaries)

    final = {
        "audit_version": AUDIT_VERSION,
        "audit_seed": AUDIT_SEED,
        "scope": {
            "claim": "name-sensitivity / lexical invariance",
            "not_a_claim": "comprehensive demographic fairness",
            "split_used_for_model_challenge": "official test only",
            "same_category_substitution_reason":
                "avoid introducing French grammatical agreement changes",
        },
        "name_inventory": {
            "source": "manual JSON pasted into script",
            "validation": inventory_report,
            "inventory": MANUAL_NAME_INVENTORY,
        },
        "selection": selection_report,
        "dataset_association": association,
        "model_result_directory": str(result_dir),
        "per_seed": seed_summaries,
        "aggregate_across_model_seeds": aggregate,
    }

    write_json(output_dir / "summary.json", final)
    write_csv(output_dir / "variant_level_results.csv", all_details)

    print("\n" + "=" * 78)
    print("RÉSUMÉ AGRÉGÉ SUR LES 3 SEEDS")
    print("=" * 78)
    for key in (
        "original_accuracy",
        "variant_accuracy",
        "flip_rate",
        "all_variants_consistent_with_original_rate",
        "mean_total_variation_distance",
        "mean_absolute_gold_probability_shift",
        "harmful_flip_rate",
        "beneficial_flip_rate",
    ):
        m = aggregate[key]["mean"]
        s = aggregate[key]["std"]
        print(f"{key:45s} {m:.4f} ± {s:.4f}")

    print("\nFichiers principaux:")
    print(f"  {output_dir / 'summary.json'}")
    print(f"  {output_dir / 'name_inventory_used.json'}")
    print(f"  {output_dir / 'name_inventory_validation.json'}")
    print(f"  {output_dir / 'variant_level_results.csv'}")
    print(f"  {output_dir / 'name_counterfactual_challenge.jsonl'}")
    print("\nAudit terminé.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[FLEXID AUDIT ERROR] {exc}", file=sys.stderr)
        raise
