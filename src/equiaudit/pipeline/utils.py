"""
Pipeline utility functions and shared constants.

Centralises markdown-formatting helpers and lookup tables that were
previously scattered across pipeline.py and evaluator.py.
"""
from __future__ import annotations

from typing import Any, Dict, List


# ── Mitigation-technique display-name mapping ──────────────────────────
# Canonical mapping from user-facing / config keys to display names.
# Used by both the CLI evaluator and the GUI to build mitigation configs.
TECHNIQUE_DISPLAY: Dict[str, str] = {
    "reweighting": "Reweighting",
    "resampling": "SMOTE",
    "smote": "SMOTE",
    "oversampling": "Random Oversampling",
    "undersampling": "Random Undersampling",
    "aif360 reweighing": "AIF360 Reweighing",
    "aif360": "AIF360 Reweighing",
    "aif360_reweighing": "AIF360 Reweighing",
}


# ── Markdown report helpers ────────────────────────────────────────────

def format_mitigation_markdown(lines: List[str], stage_data: Dict[str, Any]) -> None:
    """Format bias mitigation section as markdown (appends to *lines*)."""
    methods_results = stage_data.get("methods", {})
    applied = stage_data.get("applied_methods", list(methods_results.keys()))

    lines.append(f"**Status:** {stage_data.get('status', 'unknown')}")
    lines.append(f"**Applied Methods:** {', '.join(applied)}")
    lines.append("")

    for method in applied:
        mr = methods_results.get(method, {})
        lines.append(f"### {method}")
        lines.append("")

        if mr.get("status") == "error":
            lines.append(f"**Error:** {mr.get('error', 'Unknown error')}")
            lines.append("")
            continue

        mitigation_result = mr.get("mitigation_result", {})
        if mitigation_result:
            lines.append("#### Mitigation Results")
            lines.append("")
            if "method" in mitigation_result:
                lines.append(f"- **Technique:** {mitigation_result['method']}")
            if "original_rows" in mitigation_result and "new_rows" in mitigation_result:
                orig = mitigation_result["original_rows"]
                new = mitigation_result["new_rows"]
                change = new - orig
                pct = (change / orig * 100) if orig > 0 else 0
                lines.append(f"- **Dataset Size:** {orig:,} → {new:,} ({pct:+.1f}%)")
            if "rows_added" in mitigation_result:
                lines.append(f"- **Samples Added:** +{mitigation_result['rows_added']:,}")
            lines.append("")

        fairness_comparison = mr.get("fairness_comparison") or mitigation_result.get("fairness_comparison")
        
        if fairness_comparison:
            mitigated_metrics = fairness_comparison.get("mitigated_metrics", {})
            if mitigated_metrics and mitigated_metrics.get("status") == "success":
                format_ml_model_markdown(lines, mitigated_metrics, title=f"Evaluation ML Model ({method})", header_level=4)

        comparison_result = mr.get("comparison_result") or mitigation_result.get("comparison_result")
        if comparison_result:
            imb = comparison_result.get("imbalance_metrics", {})
            if imb:
                lines.append("#### Mitigation Scorecard")
                lines.append("")
                lines.append("| Metric | Before Mitigation | After Mitigation | Improved? | Diff |")
                lines.append("|--------|-------------------|------------------|-----------|------|")
                
                orig_ratio = float(imb.get("original_imbalance_ratio", 0))
                new_ratio = float(imb.get("mitigated_imbalance_ratio", 0))
                diff = new_ratio - orig_ratio
                improvement = imb.get("improvement", "No")
                
                lines.append(f"| Imbalance Ratio | {orig_ratio:.2f} | {new_ratio:.2f} | {improvement} | {diff:+.2f} |")

                if fairness_comparison:
                    per_attr = fairness_comparison.get("per_attribute_comparison", {})
                    for attr, metrics in per_attr.items():
                        spd = metrics.get("statistical_parity_difference", {})
                        if spd:
                            sb = float(spd.get("baseline", 0))
                            sm = float(spd.get("mitigated", 0))
                            si = "Yes" if spd.get("improved") else "No"
                            lines.append(f"| {attr} (Stat Parity) | {sb:.4f} | {sm:.4f} | {si} | {float(spd.get('change', 0)):+.4f} |")
                            
                        di = metrics.get("disparate_impact", {})
                        if di:
                            db = float(di.get("baseline", 0))
                            dm = float(di.get("mitigated", 0))
                            di_imp = "Yes" if di.get("improved") else "No"
                            lines.append(f"| {attr} (Disp Impact) | {db:.4f} | {dm:.4f} | {di_imp} | {float(di.get('change', 0)):+.4f} |")

                lines.append("")

            if "agent_analysis" in comparison_result:
                lines.append("#### Agent Analysis")
                lines.append("")
                agent_text = comparison_result["agent_analysis"]
                # Strip the redundant h1 title the LLM generates (e.g. "# Detailed Analysis...")
                # The method is already identified by the ### heading above.
                agent_lines = agent_text.split('\n')
                if agent_lines and agent_lines[0].startswith('# ') and not agent_lines[0].startswith('## '):
                    agent_lines = agent_lines[1:]
                    while agent_lines and not agent_lines[0].strip():
                        agent_lines = agent_lines[1:]
                    agent_text = '\n'.join(agent_lines)
                lines.append(agent_text)
                lines.append("")

    _format_mitigation_comparison(lines, methods_results, applied)


def _format_mitigation_comparison(
    lines: List[str], methods_results: Dict[str, Any], applied: List[str]
) -> None:
    """Append a cross-method comparative summary table at the end of Stage 6."""
    method_data: Dict[str, Any] = {}
    baseline_perf: Dict[str, Any] = {}

    for method in applied:
        mr = methods_results.get(method, {})
        if mr.get("status") == "error":
            continue
        fc = mr.get("fairness_comparison") or mr.get("mitigation_result", {}).get("fairness_comparison")
        if not fc:
            continue

        if not baseline_perf:
            bm = fc.get("baseline_metrics", {})
            baseline_perf = bm.get("performance", {})

        mm = fc.get("mitigated_metrics", {})
        method_data[method] = {
            "perf": mm.get("performance", {}),
            "per_attr": fc.get("per_attribute_comparison", {}),
        }

    if not method_data:
        return

    active = [m for m in applied if m in method_data]

    lines.append("### Method Comparison")
    lines.append("")
    lines.append("Side-by-side summary of all mitigation techniques applied.")
    lines.append("")

    # Model performance table
    lines.append("#### Model Performance")
    lines.append("")
    col_seps = "|----------" * len(active)
    lines.append("| Metric | Baseline |" + "".join(f" {m} |" for m in active))
    lines.append(f"|--------|----------{col_seps}|")

    for key, label in [("accuracy", "Accuracy"), ("f1_macro", "F1 Macro"), ("f1_weighted", "F1 Weighted")]:
        bv = baseline_perf.get(key)
        b_str = f"{bv:.4f}" if isinstance(bv, float) else "N/A"
        row = f"| {label} | {b_str} |"
        for m in active:
            v = method_data[m]["perf"].get(key)
            row += f" {f'{v:.4f}' if isinstance(v, float) else 'N/A'} |"
        lines.append(row)
    lines.append("")

    # Collect all attributes preserving first-seen order
    all_attrs: List[str] = []
    seen: set = set()
    for m in active:
        for attr in method_data[m]["per_attr"]:
            if attr not in seen:
                all_attrs.append(attr)
                seen.add(attr)

    if not all_attrs:
        return

    def _attr_display(attr: str) -> str:
        return attr.replace("_combined", "").replace("_", " + ")

    def _spd(m: str, attr: str) -> str:
        v = method_data[m]["per_attr"].get(attr, {}).get("statistical_parity_difference", {}).get("mitigated")
        return f"{float(v):.4f}" if v is not None else "N/A"

    def _di(m: str, attr: str) -> str:
        v = method_data[m]["per_attr"].get(attr, {}).get("disparate_impact", {}).get("mitigated")
        return f"{float(v):.4f}" if v is not None else "N/A"

    def _base_spd(attr: str) -> str:
        for m in active:
            v = method_data[m]["per_attr"].get(attr, {}).get("statistical_parity_difference", {}).get("baseline")
            if v is not None:
                return f"{float(v):.4f}"
        return "N/A"

    def _base_di(attr: str) -> str:
        for m in active:
            v = method_data[m]["per_attr"].get(attr, {}).get("disparate_impact", {}).get("baseline")
            if v is not None:
                return f"{float(v):.4f}"
        return "N/A"

    # SPD table
    lines.append("#### Statistical Parity Difference (lower is better)")
    lines.append("")
    lines.append("| Sensitive Attribute | Baseline |" + "".join(f" {m} |" for m in active))
    lines.append(f"|---------------------|----------{col_seps}|")
    for attr in all_attrs:
        row = f"| {_attr_display(attr)} | {_base_spd(attr)} |"
        for m in active:
            row += f" {_spd(m, attr)} |"
        lines.append(row)
    lines.append("")

    # DI table
    lines.append("#### Disparate Impact (higher is better, ideal >= 0.8)")
    lines.append("")
    lines.append("| Sensitive Attribute | Baseline |" + "".join(f" {m} |" for m in active))
    lines.append(f"|---------------------|----------{col_seps}|")
    for attr in all_attrs:
        row = f"| {_attr_display(attr)} | {_base_di(attr)} |"
        for m in active:
            row += f" {_di(m, attr)} |"
        lines.append(row)
    lines.append("")


def _format_stage_2_tool_markdown(lines: List[str], tool_result: Dict[str, Any]) -> None:
    """Format Data Quality Results into a markdown table."""
    details = tool_result.get("details", [])
    if not details:
        lines.append("**No severe data quality issues detected.**")
        lines.append("")
        return

    lines.append("### Data Quality Issues Detected")
    lines.append("")
    lines.append("| Column | Type | Missing Count | Missing % | Detected Issues |")
    lines.append("|--------|------|---------------|-----------|-----------------|")
    
    for col in details:
        name = col.get("column", "Unknown")
        dtype = col.get("data_type", "N/A")
        m_count = col.get("missing_count", 0)
        m_pct = col.get("missing_percentage", 0.0)
        issues = col.get("detected_issues", "None")
        lines.append(f"| {name} | {dtype} | {m_count} | {m_pct:.1f}% | {issues} |")
        
    lines.append("")


def format_pair_selection_markdown(lines: List[str], pair_selection: Dict[str, Any]) -> None:
    """Format pair selection info as markdown (appends to *lines*)."""
    lines.append("### Intersectional Pair Selection")
    lines.append("")
    lines.append(f"**Max Pairs Limit:** {pair_selection.get('max_pairs_limit', 'N/A')}")
    lines.append(f"**Total Possible Pairs:** {pair_selection.get('total_possible_pairs', 'N/A')}")
    lines.append("")
    lines.append("**Selected Pairs for Analysis:**")
    for pair in pair_selection.get("selected_pairs", []):
        lines.append(f"- {pair}")
    lines.append("")
    lines.append("**Selection Reasoning:**")
    lines.append("")
    lines.append(pair_selection.get("reasoning", "No reasoning provided."))
    lines.append("")


def _format_stage_3_tool_markdown(lines: List[str], stage_data: Dict[str, Any]) -> None:
    """Format Sensitive Attribute Detection results into a markdown table."""
    sensitive_columns = stage_data.get("sensitive_columns", [])
    reasons = stage_data.get("sensitive_reasons", {})
    
    if not sensitive_columns:
        lines.append("**No sensitive attributes detected.**")
        lines.append("")
        return

    lines.append("### Detected Sensitive Attributes")
    lines.append("")
    lines.append("| Column | Reason |")
    lines.append("|--------|--------|")
    
    for col in sensitive_columns:
        reason = reasons.get(col, "Identified as a protected demographic or socioeconomic attribute.")
        lines.append(f"| {col} | {reason} |")
        
    lines.append("")


def _format_stage_4_tool_markdown(lines: List[str], tool_result: Dict[str, Any]) -> None:
    """Format Class Imbalance tool results into a markdown table."""
    if not tool_result or tool_result.get("status") != "success":
        return
        
    details = tool_result.get("details", [])
    if not details:
        return
        
    lines.append("### Class Imbalance Details")
    lines.append("")
    lines.append("| Column | Dominant Value | Percentage | Top Distribution |")
    lines.append("|--------|----------------|------------|------------------|")
    
    for item in details:
        col = item.get("column", "N/A")
        dom_val = item.get("dominant_value", "N/A")
        pct = item.get("dominant_percentage", 0)
        
        dist = item.get("distribution", {})
        dist_str = ", ".join(f"{k}: {v:.1f}%" for k, v in dist.items())
        
        lines.append(f"| {col} | {dom_val} | {pct:.1f}% | {dist_str} |")
        
    lines.append("")


def _format_stage_4_5_tool_markdown(lines: List[str], tool_result: Dict[str, Any]) -> None:
    """Format Target Fairness tool results into a markdown table."""
    if not tool_result or tool_result.get("status") != "success":
        return
        
    rates = tool_result.get("target_rates_by_group", {})
    if not rates:
        return
        
    lines.append("### Target Variable Rates by Sensitive Group")
    lines.append("")
    lines.append("| Sensitive Feature | Group Level | Total Count | Target Distribution |")
    lines.append("|-------------------|-------------|-------------|---------------------|")
    
    for sensitive_col, groups in rates.items():
        for group_val, details in groups.items():
            count = details.get("total_count", 0)
            target_pcts = details.get("target_percentages", {})
            dist_str = ", ".join(f"{k}: {v:.1f}%" for k, v in target_pcts.items())
            
            lines.append(f"| {sensitive_col} | {group_val} | {count} | {dist_str} |")
            
    lines.append("")


def _format_discretization_markdown(lines: List[str], stage_data: Dict[str, Any]) -> None:
    """Format the discretization stage results as markdown (appends to *lines*)."""
    status = stage_data.get("status", "")

    if status == "skipped":
        lines.append(f"**Status:** Skipped — {stage_data.get('message', '')}")
        lines.append("")
        return

    if status == "error":
        lines.append(f"**Status:** Error — {stage_data.get('message', '')}")
        lines.append("")
        return

    method = stage_data.get("method", "unknown")
    discretized = stage_data.get("discretized_columns", [])

    if not discretized:
        lines.append(stage_data.get("message", "No continuous sensitive columns found."))
        lines.append("")
        return

    lines.append(f"**Method:** {method}")
    lines.append(f"**Columns Discretized:** {len(discretized)}")
    lines.append("")

    # Per-column details
    for col_info in discretized:
        col_name = col_info.get("column", "unknown")
        lines.append(f"### {col_name}")
        lines.append("")

        if col_info.get("status") == "error":
            lines.append(f"**Error:** {col_info.get('message', 'Unknown error')}")
            lines.append("")
            continue

        col_method = col_info.get("method", method)
        labels = col_info.get("labels", [])
        bin_edges = col_info.get("bin_edges", [])
        dist = col_info.get("distribution", {})

        lines.append(f"- **Binning Method:** {col_method}")
        if bin_edges:
            lines.append(f"- **Bin Edges:** {bin_edges}")
        if labels:
            lines.append(f"- **Labels:** {', '.join(str(l) for l in labels)}")
        lines.append("")

        if dist:
            lines.append("**Bin Distribution:**")
            lines.append("")
            lines.append("| Bin | Count |")
            lines.append("|-----|-------|")
            for bin_label, count in dist.items():
                lines.append(f"| {bin_label} | {count} |")
            lines.append("")

    # Agent reasoning
    agent_analysis = stage_data.get("agent_analysis", "")
    if agent_analysis and agent_analysis.strip():
        lines.append("### Agent Reasoning")
        lines.append("")
        lines.append(agent_analysis)
        lines.append("")


def _get_csv_folder_and_prefix(title: str):
    """Determine the subfolder name and file prefix for exported CSVs."""
    title_lower = title.lower()
    if "base" in title_lower:
        return "base_fairness", "base"
    elif "intersectional" in title_lower:
        return "intersectional_fairness", "intersectional"
    elif "evaluation" in title_lower:
        import re
        m = re.search(r'\((.*?)\)', title)
        method = m.group(1).lower().replace(" ", "_") if m else "unknown"
        return f"mitigation_{method}", f"mitigated_{method}"
    else:
        prefix = title_lower.replace(" ", "_").replace("(", "").replace(")", "")
        return "other_fairness", prefix


def format_fairness_board_markdown(lines: List[str], ml_results: Dict[str, Any], title: str = "", header_level: int = 4) -> None:
    """Format the fairness evaluation metrics into a markdown table, omitting raw data."""
    fairness = ml_results.get("fairness_analysis", {})
    if not fairness:
        return

    folder_name, prefix = _get_csv_folder_and_prefix(title)

    lines.append(f"{'#' * header_level} Evaluated Fairness Metrics")
    lines.append("")
    lines.append("| Sensitive Attribute | Stat Parity Diff | Disparate Impact | Highest Rate Group | Lowest Rate Group |")
    lines.append("|---------------------|------------------|------------------|--------------------|-------------------|")

    for col_name, data in fairness.items():
        metrics_data = data.get("metrics", {})
        spd_value = metrics_data.get("statistical_parity_difference", 0)
        di_value = metrics_data.get("disparate_impact", 0)
        max_group = metrics_data.get("max_positive_rate_group", "N/A")
        min_group = metrics_data.get("min_positive_rate_group", "N/A")

        attr_display = col_name.replace("_combined", "").replace("_", " + ")

        lines.append(f"| {attr_display} | {spd_value:.4f} | {di_value:.4f} | {max_group} | {min_group} |")

    lines.append("")


def format_ml_model_markdown(lines: List[str], ml_results: Dict[str, Any], title: str = "Machine Learning Evaluation Model", header_level: int = 3) -> None:
    """Format ML model details into markdown (appends to *lines*)."""
    if not ml_results or ml_results.get("status") != "success":
        return
    model_type = ml_results.get("model_type")
    if not model_type:
        return

    lines.append(f"{'#' * header_level} {title}")
    lines.append("")
    lines.append(f"- **Algorithm:** {model_type}")

    test_size = ml_results.get("test_size")
    if test_size is not None:
        lines.append(f"- **Test Size:** {test_size}")

    accuracy = ml_results.get("performance", {}).get("accuracy")
    if accuracy is not None:
        lines.append(f"- **Accuracy:** {accuracy:.4f}")

    params = ml_results.get("model_params") or {}
    if params:
        params_str = ", ".join(f"`{k}={v}`" for k, v in params.items())
        lines.append(f"- **Parameters:** {params_str}")
    else:
        lines.append("- **Parameters:** Default settings")
    lines.append("")

    format_fairness_board_markdown(lines, ml_results, title, header_level=header_level + 1)


# ── Report Generation ──────────────────────────────────────────────────

import os
import hashlib
from datetime import datetime
from equiaudit.pipeline.stages.base import safe_json_dumps

def generate_markdown_report(pipeline) -> str:
    """Generate pure markdown report (human-readable, easy PDF conversion)."""
    dataset_hash = hashlib.md5(pipeline.current_dataset.encode()).hexdigest()[:8]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines: List[str] = []
    lines.append("# EquiAudit Fairness Report")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **Dataset:** {pipeline.current_dataset}")
    lines.append(f"- **Timestamp:** {ts}")
    lines.append(f"- **Dataset Hash:** {dataset_hash}")
    if hasattr(pipeline, "target_column") and pipeline.target_column:
        lines.append(f"- **Target Column:** {pipeline.target_column}")
    lines.append(f"- **Objective:** {pipeline.user_objective or 'Dataset auditing'}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Inject Executive Summary if available
    executive_summary = pipeline.evaluation_results.get("executive_summary")
    if executive_summary:
        lines.append(executive_summary)
        lines.append("")
        lines.append("---")
        lines.append("")
    
    stage_titles = {
        "0_loading": "Stage 0: Dataset Loading",
        "1_objective": "Stage 1: Objective Validation",
        "2_quality": "Stage 2: Data Quality Inspection",
        "3_sensitive": "Stage 3: Sensitive Attribute Identification",
        "3_5_discretization": "Stage 3.5: Sensitive Attribute Discretization",
        "4_imbalance": "Stage 4: Imbalance Analysis",
        "4_5_target_fairness": "Stage 4.5: Target Fairness Analysis",
        "5_recommendations": "Stage 5: Recommendation Synthesis",
        "6_bias_mitigation": "Stage 6: Bias Mitigation",
    }
    
    for stage_key, stage_data in pipeline.evaluation_results["stages"].items():
        title = stage_titles.get(stage_key, stage_key.replace("_", " ").title())
        lines.append(f"## {title}")
        lines.append("")
        
        if not isinstance(stage_data, dict):
            lines.append(str(stage_data))
            lines.append("")
            lines.append("---")
            lines.append("")
            continue
        
        if "tool_used" in stage_data:
            lines.append(f"**Tool Used:** `{stage_data['tool_used']}`")
            lines.append("")
        
        if stage_key == "6_bias_mitigation" and "methods" in stage_data:
            format_mitigation_markdown(lines, stage_data)
        elif stage_key == "3_5_discretization":
            # Structured discretization report
            _format_discretization_markdown(lines, stage_data)
        elif "agent_analysis" in stage_data:
            lines.append("### Analysis")
            lines.append("")
            
            if stage_key == "2_data_quality" and "tool_result" in stage_data:
                _format_stage_2_tool_markdown(lines, stage_data["tool_result"])

            if stage_key == "3_sensitive":
                _format_stage_3_tool_markdown(lines, stage_data)

            if stage_key == "4_imbalance" and "tool_result" in stage_data:
                _format_stage_4_tool_markdown(lines, stage_data["tool_result"])
                
            # Add Pair Selection before Target Fairness
            if stage_key == "4_5_target_fairness":
                sensitive_stage = pipeline.evaluation_results["stages"].get("3_sensitive", {})
                pair_sel = sensitive_stage.get("pair_selection")
                if pair_sel:
                    format_pair_selection_markdown(lines, pair_sel)
                
                if "tool_result" in stage_data:
                    _format_stage_4_5_tool_markdown(lines, stage_data["tool_result"])
                    
            # Add ML Model info before analysis
            if "ml_model_results" in stage_data:
                format_ml_model_markdown(lines, stage_data["ml_model_results"], title="Base Fairness ML Model")
            if "single_attribute_ml_results" in stage_data:
                format_ml_model_markdown(lines, stage_data["single_attribute_ml_results"], title="Per-Attribute Fairness ML Model")
            if "intersectional_ml_results" in stage_data:
                format_ml_model_markdown(lines, stage_data["intersectional_ml_results"], title="Intersectional Fairness ML Model")
                
            lines.append(stage_data["agent_analysis"])
            lines.append("")
                
        elif "recommendations" in stage_data:
            lines.append("### Recommendations")
            lines.append("")
            lines.append(stage_data["recommendations"])
            lines.append("")
        elif "agent_response" in stage_data:
            lines.append("### Response")
            lines.append("")
            lines.append(str(stage_data["agent_response"]))
            lines.append("")
        else:
            if "objective" in stage_data:
                lines.append(f"**Objective:** {stage_data.get('objective', 'N/A')}")
                lines.append("")
            if "validation" in stage_data:
                lines.append(f"**Validation:** {stage_data.get('validation', 'N/A')}")
                lines.append("")
        
        lines.append("---")
        lines.append("")
    
    lines.append("*Report generated by EquiAudit*")
    return "\n".join(lines)


def generate_json_data(pipeline) -> Dict[str, Any]:
    """Generate JSON file with all tool results organized by stage."""
    dataset_hash = hashlib.md5(pipeline.current_dataset.encode()).hexdigest()[:8]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    json_data = {
        "metadata": {
            "dataset": pipeline.current_dataset,
            "timestamp": ts,
            "dataset_hash": dataset_hash,
            "target_column": getattr(pipeline, "target_column", None),
            "objective": pipeline.user_objective or "Dataset auditing",
            "report_directory": pipeline.report_dir,
        },
        "stages": {}
    }
    
    for stage_key, stage_data in pipeline.evaluation_results["stages"].items():
        stage_json = {}
        
        if isinstance(stage_data, dict):
            if "tool_used" in stage_data:
                stage_json["tool_used"] = stage_data["tool_used"]
            if "tool_result" in stage_data:
                stage_json["tool_result"] = stage_data["tool_result"]
            if "pair_selection" in stage_data:
                stage_json["pair_selection"] = stage_data["pair_selection"]
            if "ml_model_results" in stage_data:
                stage_json["ml_model_results"] = stage_data["ml_model_results"]
            if "single_attribute_ml_results" in stage_data:
                stage_json["single_attribute_ml_results"] = stage_data["single_attribute_ml_results"]
            if "intersectional_ml_results" in stage_data:
                stage_json["intersectional_ml_results"] = stage_data["intersectional_ml_results"]
            if "methods" in stage_data:
                stage_json["methods"] = stage_data["methods"]
                stage_json["applied_methods"] = stage_data.get("applied_methods", [])
                stage_json["status"] = stage_data.get("status", "unknown")
        else:
            stage_json["data"] = stage_data
        
        json_data["stages"][stage_key] = stage_json
    
    return json_data


def save_fairness_comparison_files(pipeline):
    """Save individual fairness comparison JSON files for each method."""
    stage_data = pipeline.evaluation_results["stages"].get("6_bias_mitigation", {})
    methods_results = stage_data.get("methods", {})
    
    for method, mr in methods_results.items():
        mitigation_result = mr.get("mitigation_result", {})
        fairness_comparison = mr.get("fairness_comparison") or mitigation_result.get("fairness_comparison")
        
        if fairness_comparison and fairness_comparison.get("status") != "error":
            try:
                fn = f"fairness_comparison_{method.lower().replace(' ', '_')}.json"
                fp = os.path.join(pipeline.report_dir, fn)
                with open(fp, "w", encoding="utf-8") as f:
                    f.write(safe_json_dumps(fairness_comparison))
            except Exception as exc:
                print(f"Warning: Could not save fairness comparison JSON: {exc}")

    save_fairness_csv_files(pipeline)


def save_fairness_csv_files(pipeline):
    """Save fairness detailed group statistics as CSV files in the report directory."""
    import csv

    def extract_and_save(ml_results: Dict[str, Any], title: str):
        if not ml_results or ml_results.get("status") != "success":
            return
        fairness = ml_results.get("fairness_analysis", {})
        if not fairness:
            return
            
        folder_name, prefix = _get_csv_folder_and_prefix(title)
        
        target_dir = os.path.join(pipeline.report_dir, folder_name)
        os.makedirs(target_dir, exist_ok=True)

        for col_name, data in fairness.items():
            groups_data = data.get("groups", {})
            if not groups_data:
                continue
            
            sanitized_col = col_name.replace("_combined", "").replace(" + ", "_").replace(" ", "")
            file_prefix = f"{prefix}_" if prefix else ""
            csv_filename = f"{file_prefix}fairness_stats_{sanitized_col}.csv"
            csv_path = os.path.join(target_dir, csv_filename)
            
            try:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Group", "Count", "Accuracy", "F1 Score", "Selection Rate", "Base Rate", "FNR", "FPR", "TPR", "TNR"])
                    
                    for group, metrics in groups_data.items():
                        g_name = group.replace("_", " + ")
                        
                        def fmt(v):
                            if isinstance(v, (int, float)) and not isinstance(v, bool):
                                if isinstance(v, float):
                                    return f"{v:.4f}"
                                return str(v)
                            return str(v)

                        writer.writerow([
                            g_name,
                            fmt(metrics.get("count", "N/A")),
                            fmt(metrics.get("accuracy", "N/A")),
                            fmt(metrics.get("f1_macro", "N/A")),
                            fmt(metrics.get("positive_rate", "N/A")),
                            fmt(metrics.get("base_rate", "N/A")),
                            fmt(metrics.get("fnr", "N/A")),
                            fmt(metrics.get("fpr", "N/A")),
                            fmt(metrics.get("tpr", "N/A")),
                            fmt(metrics.get("tnr", "N/A"))
                        ])
            except Exception as exc:
                print(f"Warning: Could not save fairness CSV file {csv_filename}: {exc}")

    stages = pipeline.evaluation_results.get("stages", {})
    
    # Check stage 4 Base Model
    if "4_imbalance" in stages and "ml_model_results" in stages["4_imbalance"]:
        extract_and_save(stages["4_imbalance"]["ml_model_results"], "Base Fairness ML Model")
        
    # Check stage 4.5 Target Fairness
    if "4_5_target_fairness" in stages:
        s45 = stages["4_5_target_fairness"]
        if "single_attribute_ml_results" in s45:
            extract_and_save(s45["single_attribute_ml_results"], "Per-Attribute Fairness ML Model")
        if "intersectional_ml_results" in s45:
            extract_and_save(s45["intersectional_ml_results"], "Intersectional Fairness ML Model")
        
    # Check stage 6 Bias Mitigation
    if "6_bias_mitigation" in stages:
        mr_dict = stages["6_bias_mitigation"].get("methods", {})
        for method, mr in mr_dict.items():
            mitigation_result = mr.get("mitigation_result", {})
            fc = mr.get("fairness_comparison") or mitigation_result.get("fairness_comparison")
            if fc:
                mitigated_metrics = fc.get("mitigated_metrics", {})
                if mitigated_metrics and mitigated_metrics.get("status") == "success":
                    extract_and_save(mitigated_metrics, f"Evaluation ML Model ({method})")

def generate_detailed_markdown_report(pipeline: Any) -> str:
    """Generates a separate markdown report containing detailed group-level fairness tables."""
    lines = []
    lines.append("# Detailed Group Metrics Report")
    lines.append("")
    
    stages = pipeline.evaluation_results.get("stages", {})
    
    def append_detailed_tables(ml_results: Dict[str, Any], title: str) -> None:
        if not ml_results or ml_results.get("status") != "success":
            return
        fairness = ml_results.get("fairness_analysis", {})
        if not fairness:
            return
            
        lines.append(f"## {title}")
        lines.append("")
        
        for col_name, data in fairness.items():
            groups_data = data.get("groups", {})
            if not groups_data:
                continue
                
            attr_display = col_name.replace("_combined", "").replace("_", " + ")
            lines.append(f"### Detailed Group Metrics: {attr_display}")
            lines.append("")
            lines.append("| Group | Count | Accuracy | F1 Score | Base Rate | Selection Rate | FNR | FPR |")
            lines.append("|-------|-------|----------|----------|-----------|----------------|-----|-----|")
            
            for group, metrics in groups_data.items():
                g_name = group.replace("_", " + ")
                
                def fmt(v):
                    return f"{v:.4f}" if isinstance(v, float) else str(v)
                
                acc = fmt(metrics.get("accuracy", "N/A"))
                f1 = fmt(metrics.get("f1_macro", "N/A"))
                br = fmt(metrics.get("base_rate", "N/A"))
                sr = fmt(metrics.get("positive_rate", "N/A"))
                fnr = fmt(metrics.get("fnr", "N/A"))
                fpr = fmt(metrics.get("fpr", "N/A"))
                count = fmt(metrics.get("count", "N/A"))
                
                lines.append(f"| {g_name} | {count} | {acc} | {f1} | {br} | {sr} | {fnr} | {fpr} |")
                
            lines.append("")

    if "4_imbalance" in stages and "ml_model_results" in stages["4_imbalance"]:
        append_detailed_tables(stages["4_imbalance"]["ml_model_results"], "Stage 4: Base Fairness ML Model")
        
    if "4_5_target_fairness" in stages:
        s45 = stages["4_5_target_fairness"]
        append_detailed_tables(s45.get("single_attribute_ml_results") or {}, "Stage 4.5: Per-Attribute Fairness ML Model")
        append_detailed_tables(s45.get("intersectional_ml_results") or {}, "Stage 4.5: Intersectional Fairness ML Model")
        
    if "6_bias_mitigation" in stages:
        base_ml_results = (stages.get("4_imbalance") or {}).get("ml_model_results") or {}
        base_fairness = base_ml_results.get("fairness_analysis") or {}
        
        mr_dict = stages["6_bias_mitigation"].get("methods", {})
        for method, mr in mr_dict.items():
            mitigation_result = mr.get("mitigation_result", {})
            fc = mr.get("fairness_comparison") or mitigation_result.get("fairness_comparison")
            if fc:
                mitigated_metrics = fc.get("mitigated_metrics", {})
                if mitigated_metrics and mitigated_metrics.get("status") == "success":
                    append_detailed_tables(mitigated_metrics, f"Stage 6: Post-Mitigation ML Model ({method})")
                    
                    # Generate side-by-side comparative table for all metrics per group
                    lines.append(f"### Comparative Group Metrics: Before vs After ({method})")
                    lines.append("")
                    
                    mitigated_fairness = mitigated_metrics.get("fairness_analysis", {})
                    
                    for col_name, base_data in base_fairness.items():
                        base_groups = base_data.get("groups", {})
                        mitigated_groups = mitigated_fairness.get(col_name, {}).get("groups", {})
                        
                        if not base_groups or not mitigated_groups:
                            continue
                            
                        attr_display = col_name.replace("_combined", "").replace("_", " + ")
                        lines.append(f"#### {attr_display}")
                        lines.append("")
                        lines.append("| Group | F1 (Before) | F1 (After) | FNR (Before) | FNR (After) | FPR (Before) | FPR (After) | Sel. Rate (Before) | Sel. Rate (After) |")
                        lines.append("|-------|-------------|------------|--------------|-------------|--------------|-------------|--------------------|-------------------|")
                        
                        for group, b_metrics in base_groups.items():
                            m_metrics = mitigated_groups.get(group, {})
                            g_name = group.replace("_", " + ")
                            
                            def fmt(v):
                                return f"{v:.4f}" if isinstance(v, float) else str(v)
                            
                            b_f1 = fmt(b_metrics.get("f1_macro", "N/A"))
                            m_f1 = fmt(m_metrics.get("f1_macro", "N/A"))
                            b_fnr = fmt(b_metrics.get("fnr", "N/A"))
                            m_fnr = fmt(m_metrics.get("fnr", "N/A"))
                            b_fpr = fmt(b_metrics.get("fpr", "N/A"))
                            m_fpr = fmt(m_metrics.get("fpr", "N/A"))
                            b_sr = fmt(b_metrics.get("positive_rate", "N/A"))
                            m_sr = fmt(m_metrics.get("positive_rate", "N/A"))
                            
                            lines.append(f"| {g_name} | {b_f1} | {m_f1} | {b_fnr} | {m_fnr} | {b_fpr} | {m_fpr} | {b_sr} | {m_sr} |")
                        
                        lines.append("")

    return "\n".join(lines)
