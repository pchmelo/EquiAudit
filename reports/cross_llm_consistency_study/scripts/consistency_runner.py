#!/usr/bin/env python3
"""
experiments/consistency_runner.py

Runs ONLY Stages 0-3.5 of the EquiAudit pipeline for one (LLM, dataset, run).
Records the agentic decisions: sensitive attribute detection, discretization
bin edges, and intersectional pair selection.

No ML training is performed (Stages 4, 4.5, 5, 6 are skipped), so each run
is fast (roughly 30-90 seconds depending on the model and dataset size).

Usage (from project root):
    python experiments/consistency_runner.py \
        --config experiments/configs/config_qwen.yml \
        --dataset adult-all.csv \
        --run-id 1

    python experiments/consistency_runner.py \
        --config experiments/configs/config_deepseek.yml \
        --dataset GermanCredit.csv \
        --run-id 3 \
        --output experiments/results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

# Allow running from project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Load .env from project root so API keys (OPENROUTER_API_KEY etc.) are available
try:
    from dotenv import load_dotenv
    _root = os.path.join(os.path.dirname(__file__), "..")
    for _env in (os.path.join(_root, ".env"), os.path.join(_root, "examples", ".env")):
        if os.path.exists(_env):
            load_dotenv(_env)
            break
except ImportError:
    pass

from equiaudit.pipeline.pipeline import DatasetEvaluationPipeline
from equiaudit.pipeline.stage import NavigationAction


STUDY_STAGES = {
    "0_loading",
    "1_objective",
    "2_quality",
    "3_sensitive",
    "3_5_discretization",
}

DATASET_TARGETS = {
    "adult-all.csv": "Income",
    "GermanCredit.csv": "credit_risk",
}

MAX_PAIRS = 2


def extract_decisions(pipeline: DatasetEvaluationPipeline) -> dict:
    results = pipeline.evaluation_results.get("stages", {})

    stage3 = results.get("3_sensitive", {})
    stage3_5 = results.get("3_5_discretization", {})

    sensitive_columns = stage3.get("sensitive_columns", [])

    pair_info = stage3.get("pair_selection", {})
    pair_selection = {
        "selected_pairs": pair_info.get("selected_pairs", []),
        "total_possible": pair_info.get("total_possible_pairs", 0),
        "mode": pair_info.get("mode", ""),
        "reasoning_excerpt": (pair_info.get("reasoning", "") or "")[:400],
    }

    bins_per_attr: dict = {}
    disc_cols = stage3_5.get("discretized_columns", [])
    for item in disc_cols:
        if isinstance(item, dict) and item.get("status") == "success":
            attr = item.get("column") or item.get("column_name", "unknown")
            bins_per_attr[attr] = {
                "edges": item.get("bin_edges", []),
                "labels": item.get("labels", []),
                "method": item.get("method", ""),
                "n_bins": len(item.get("labels", [])),
            }

    return {
        "sensitive_columns": sensitive_columns,
        "num_sensitive_columns": len(sensitive_columns),
        "pair_selection": pair_selection,
        "discretization": bins_per_attr,
        "stage3_tool_used": stage3.get("tool_used", ""),
        "stage3_agent_excerpt": (stage3.get("agent_analysis", "") or "")[:500],
    }


def run_study(
    config_path: str,
    dataset: str,
    target_col: str,
    run_id: int,
    output_dir: str,
) -> str:
    start = time.time()
    config_stem = os.path.splitext(os.path.basename(config_path))[0]
    print(
        f"[{config_stem}] dataset={dataset} run={run_id} | "
        f"config={os.path.abspath(config_path)}"
    )

    pipeline = DatasetEvaluationPipeline(config_path=os.path.abspath(config_path))

    tmp_report_dir = os.path.join(output_dir, "_tmp_pipeline_reports")
    pipeline.build_stages(
        dataset_name=dataset,
        target_column=target_col,
        user_prompt=f"Evaluate {dataset} for fairness biases. Target={target_col}",
        report_base_dir=tmp_report_dir,
    )

    # Force auto-detection mode and agent-driven pair selection
    ctx = pipeline._pipeline_ctx
    ctx["confirmed_sensitive_columns"] = None
    ctx["discretization_method"] = "auto"
    ctx["discretization_enabled"] = True
    ctx["max_pairs"] = MAX_PAIRS
    ctx["user_specified_pairs"] = None

    stages_run: list[str] = []
    while not pipeline.is_finished:
        stage = pipeline.current_stage
        if stage.key not in STUDY_STAGES:
            break
        print(f"  [{stage.key}] {stage.name}...")
        pipeline.navigate(NavigationAction.FORWARD)
        stages_run.append(stage.key)

    elapsed = round(time.time() - start, 1)
    decisions = extract_decisions(pipeline)

    record = {
        "run_id": run_id,
        "model_config": config_stem,
        "dataset": dataset,
        "target_column": target_col,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "stages_run": stages_run,
        "decisions": decisions,
    }

    os.makedirs(output_dir, exist_ok=True)
    dataset_stem = os.path.splitext(dataset)[0]
    out_name = f"{config_stem}__{dataset_stem}__run{run_id:02d}.json"
    out_path = os.path.join(output_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)

    n_attrs = decisions["num_sensitive_columns"]
    n_pairs = len(decisions["pair_selection"]["selected_pairs"])
    n_disc = len(decisions["discretization"])
    print(
        f"  Done in {elapsed}s | attrs={n_attrs} pairs={n_pairs} disc={n_disc} | {out_path}"
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="EquiAudit consistency study: single run (stages 0-3.5 only)"
    )
    parser.add_argument("--config", required=True, help="Path to agent config YAML")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_TARGETS.keys()),
        help="Dataset filename",
    )
    parser.add_argument(
        "--target",
        help="Target column (auto-resolved if omitted for known datasets)",
    )
    parser.add_argument(
        "--run-id", type=int, default=1, metavar="N", help="Run index 1-10"
    )
    parser.add_argument(
        "--output",
        default="experiments/results",
        help="Directory to save the JSON result",
    )
    args = parser.parse_args()

    target_col = args.target or DATASET_TARGETS.get(args.dataset)
    if not target_col:
        parser.error(
            f"Cannot resolve target column for '{args.dataset}'. "
            "Specify with --target."
        )

    run_study(args.config, args.dataset, target_col, args.run_id, args.output)


if __name__ == "__main__":
    main()
