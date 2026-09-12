#!/usr/bin/env python3
"""
experiments/analyze_consistency.py

Reads all JSON result files produced by consistency_runner.py and reports:

  1. Sensitive attribute agreement rate per (model, dataset)
  2. Inter-model Jaccard similarity on attribute sets
  3. Discretization bin-label exact-match rate per (model, dataset, attr)
  4. Pair selection agreement rate per (model, dataset)

Usage (from project root):
    python experiments/analyze_consistency.py
    python experiments/analyze_consistency.py --results experiments/results/
    python experiments/analyze_consistency.py --csv experiments/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str) -> List[dict]:
    runs: List[dict] = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json") and not fname.startswith("_"):
            path = os.path.join(results_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    runs.append(json.load(fh))
            except Exception as e:
                print(f"  Warning: could not read {fname}: {e}")
    return runs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def normalize_pair(p: str) -> str:
    """Sort the two attribute names in a pair string so 'A + B' == 'B + A'."""
    parts = [x.strip() for x in p.split("+")]
    return " + ".join(sorted(parts))


def normalize_list(lst: List) -> frozenset:
    """Normalize a list of pairs/attrs into a frozenset for comparison."""
    return frozenset(normalize_pair(s) if " + " in s or "+" in s else s for s in lst)


def mode_of_sets(lists: List[List]) -> frozenset:
    counter: Counter = Counter(normalize_list(s) for s in lists)
    return counter.most_common(1)[0][0] if counter else frozenset()


def agreement_rate(lists: List[List]) -> float:
    if not lists:
        return 0.0
    mode = mode_of_sets(lists)
    return sum(1 for lst in lists if normalize_list(lst) == mode) / len(lists)


# ---------------------------------------------------------------------------
# Grouping helper
# ---------------------------------------------------------------------------

def group_runs(runs: List[dict]) -> Dict[Tuple[str, str], List[dict]]:
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for run in runs:
        key = (run["model_config"], run["dataset"])
        groups[key].append(run)
    return dict(groups)


def short(name: str) -> str:
    return name.replace("config_", "").replace(".yml", "")


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def section_attrs(groups: Dict[Tuple[str, str], List[dict]], csv_rows: List[dict]):
    print("\n" + "=" * 72)
    print("1. SENSITIVE ATTRIBUTE AGREEMENT")
    print("=" * 72)
    fmt = f"{'Model':<28} {'Dataset':<22} {'N':>3} {'Agree%':>8}  Mode attribute set"
    print(fmt)
    print("-" * 72)

    for (model, dataset), group in sorted(groups.items()):
        lists = [r["decisions"]["sensitive_columns"] for r in group]
        rate = agreement_rate(lists)
        mode = sorted(mode_of_sets(lists))
        print(
            f"{short(model):<28} {os.path.splitext(dataset)[0]:<22} "
            f"{len(lists):>3} {rate*100:>7.1f}%  {mode}"
        )
        csv_rows.append({
            "section": "attr_agreement",
            "model": short(model),
            "dataset": dataset,
            "n_runs": len(lists),
            "agree_pct": round(rate * 100, 1),
            "mode_value": str(mode),
        })


def section_jaccard(groups: Dict[Tuple[str, str], List[dict]], csv_rows: List[dict]):
    print("\n" + "=" * 72)
    print("2. INTER-MODEL JACCARD SIMILARITY (on mode attribute sets)")
    print("=" * 72)

    by_dataset: Dict[str, Dict[str, frozenset]] = defaultdict(dict)
    for (model, dataset), group in groups.items():
        lists = [r["decisions"]["sensitive_columns"] for r in group]
        by_dataset[dataset][model] = mode_of_sets(lists)  # already normalized

    for dataset, model_modes in sorted(by_dataset.items()):
        print(f"\n  Dataset: {dataset}")
        models = sorted(model_modes.keys())
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                j = jaccard(model_modes[m1], model_modes[m2])
                print(f"    {short(m1)} vs {short(m2)}: Jaccard = {j:.3f}")
                csv_rows.append({
                    "section": "jaccard",
                    "model": f"{short(m1)} vs {short(m2)}",
                    "dataset": dataset,
                    "n_runs": "",
                    "agree_pct": round(j, 3),
                    "mode_value": "",
                })


def section_discretization(groups: Dict[Tuple[str, str], List[dict]], csv_rows: List[dict]):
    print("\n" + "=" * 72)
    print("3. DISCRETIZATION BIN-LABEL CONSISTENCY (exact-match rate)")
    print("=" * 72)
    fmt = f"{'Model':<28} {'Dataset':<22} {'Attribute':<18} {'N':>3} {'Exact%':>8}"
    print(fmt)
    print("-" * 72)

    for (model, dataset), group in sorted(groups.items()):
        disc_by_attr: Dict[str, List[tuple]] = defaultdict(list)
        for run in group:
            for attr, info in run["decisions"]["discretization"].items():
                disc_by_attr[attr].append(tuple(info.get("labels", [])))

        for attr, label_runs in sorted(disc_by_attr.items()):
            counter: Counter = Counter(label_runs)
            exact = counter.most_common(1)[0][1] / len(label_runs)
            print(
                f"{short(model):<28} {os.path.splitext(dataset)[0]:<22} "
                f"{attr:<18} {len(label_runs):>3} {exact*100:>7.1f}%"
            )
            csv_rows.append({
                "section": "discretization",
                "model": short(model),
                "dataset": dataset,
                "n_runs": len(label_runs),
                "agree_pct": round(exact * 100, 1),
                "mode_value": str(list(counter.most_common(1)[0][0])),
            })


def section_pairs(groups: Dict[Tuple[str, str], List[dict]], csv_rows: List[dict]):
    print("\n" + "=" * 72)
    print("4. PAIR SELECTION CONSISTENCY")
    print("=" * 72)
    fmt = f"{'Model':<28} {'Dataset':<22} {'N':>3} {'Agree%':>8}  Mode pairs"
    print(fmt)
    print("-" * 72)

    for (model, dataset), group in sorted(groups.items()):
        lists = [r["decisions"]["pair_selection"]["selected_pairs"] for r in group]
        rate = agreement_rate(lists)
        mode = sorted(mode_of_sets(lists))
        print(
            f"{short(model):<28} {os.path.splitext(dataset)[0]:<22} "
            f"{len(lists):>3} {rate*100:>7.1f}%  {mode}"
        )
        csv_rows.append({
            "section": "pair_selection",
            "model": short(model),
            "dataset": dataset,
            "n_runs": len(lists),
            "agree_pct": round(rate * 100, 1),
            "mode_value": str(mode),
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze EquiAudit consistency study results"
    )
    parser.add_argument(
        "--results",
        default="experiments/results",
        help="Directory containing JSON result files",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to save a summary CSV (e.g. experiments/summary.csv)",
    )
    args = parser.parse_args()

    runs = load_results(args.results)
    if not runs:
        print(f"No result files found in: {args.results}")
        return

    print(f"Loaded {len(runs)} runs from {args.results}")

    groups = group_runs(runs)
    csv_rows: List[dict] = []

    section_attrs(groups, csv_rows)
    section_jaccard(groups, csv_rows)
    section_discretization(groups, csv_rows)
    section_pairs(groups, csv_rows)

    print(f"\n{'=' * 72}")
    print(
        f"Total: {len(runs)} runs across "
        f"{len(set(r['model_config'] for r in runs))} model(s), "
        f"{len(set(r['dataset'] for r in runs))} dataset(s)"
    )

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        fields = ["section", "model", "dataset", "n_runs", "agree_pct", "mode_value"]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nCSV summary saved: {args.csv}")


if __name__ == "__main__":
    main()
