from __future__ import annotations

import os
from typing import Any, Dict

from equiaudit.pipeline.stages.base import BaseStageExecutor, safe_json_dumps


"""
Stage 6 – Bias Mitigation.
Apply bias-mitigation techniques and compare fairness metrics.
"""


class MitigationStage(BaseStageExecutor):
    METHOD_MAP = {
        "Reweighting": "reweighting",
        "SMOTE": "smote",
        "Random Oversampling": "oversampling",
        "Random Undersampling": "undersampling",
        "AIF360 Reweighing": "aif360_reweighing",
    }

    def __call__(self, stage, ctx: Dict[str, Any]) -> Dict[str, Any]:
        mitigation_cfg = ctx.get("mitigation_config")
        if not mitigation_cfg or not mitigation_cfg.get("methods"):
            return {"status": "skipped", "message": "No mitigation methods selected."}

        sensitive_cols: list = (
            ctx["results"].get("3_sensitive", {}).get("sensitive_columns", [])
        )

        all_results: Dict[str, Any] = {}
        selected_methods = mitigation_cfg["methods"]

        for method_name, method_params in selected_methods.items():
            try:
                method_key = self.METHOD_MAP.get(method_name, method_name.lower())
                sens = method_params.get("sensitive_columns", sensitive_cols)
                extra = {k: v for k, v in method_params.items() if k != "sensitive_columns"}

                mitigation_result = _apply_single_mitigation(
                    method=method_key,
                    ctx=ctx,
                    sensitive_columns=sens,
                    extra_params=extra,
                )

                if mitigation_result and mitigation_result.get("status") == "success":
                    comparison_result = _compare_datasets(
                        ctx=ctx,
                        mitigated_dataset=mitigation_result.get("output_file"),
                        sensitive_columns=sensitive_cols,
                        agent=ctx["pipeline"].recommendation_agent,
                        fairness_comparison=mitigation_result.get("fairness_comparison"),
                    )
                    all_results[method_name] = {
                        "status": "success",
                        "method_params": method_params,
                        "mitigation_result": mitigation_result,
                        "comparison_result": comparison_result,
                        "fairness_comparison": mitigation_result.get("fairness_comparison"),
                    }
                else:
                    all_results[method_name] = {
                        "status": "error",
                        "error": (mitigation_result or {}).get("message", "Unknown error"),
                    }
            except Exception as exc:
                all_results[method_name] = {"status": "error", "error": str(exc)}

        return {
            "status": "success",
            "methods": all_results,
            "applied_methods": list(selected_methods.keys()),
        }




def _apply_single_mitigation(
    method: str,
    ctx: Dict[str, Any],
    sensitive_columns: list,
    extra_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply one bias-mitigation technique and optionally run a fairness comparison."""
    
    bias_tools = ctx["bias_mitigation_tools"]
    fairness_tools = ctx["fairness_tools"]
    dataset_name = ctx["dataset_name"]
    target_column = ctx["target_column"]

    output_dir = os.path.join(ctx["report_dir"], "mitigation")
    os.makedirs(output_dir, exist_ok=True)

    shared = dict(dataset_name=dataset_name, target_column=target_column, output_dir=output_dir)

    if method == "reweighting":
        if not sensitive_columns:
            return {"status": "error", "message": "Sensitive columns required for reweighting"}
        result = bias_tools.apply_reweighting(sensitive_columns=sensitive_columns, **shared)
    elif method == "smote":
        result = bias_tools.apply_smote(
            k_neighbors=extra_params.get("k_neighbors", 5),
            sampling_strategy=extra_params.get("sampling_strategy", "auto"),
            **shared,
        )
    elif method == "oversampling":
        result = bias_tools.apply_oversampling(
            sampling_strategy=extra_params.get("sampling_strategy", "auto"),
            **shared,
        )
    elif method == "undersampling":
        result = bias_tools.apply_undersampling(
            sampling_strategy=extra_params.get("sampling_strategy", "auto"),
            **shared,
        )
    elif method == "aif360_reweighing":
        if not sensitive_columns:
            return {"status": "error", "message": "Sensitive columns required for AIF360 Reweighing"}
        result = bias_tools.apply_aif360_reweighing(sensitive_columns=sensitive_columns, **shared)
    else:
        return {"status": "error", "message": f"Unknown method: {method}"}

    # Fairness comparison against baseline
    if result.get("status") == "success" and result.get("output_file"):
        import pandas as pd

        # Use stage 4.5 (Target Fairness) as the baseline — it is the authoritative
        # pre-mitigation fairness measurement.
        target_fairness = (ctx.get("results") or {}).get("4_5_target_fairness", {})
        baseline_raw = target_fairness.get("single_attribute_ml_results") or {}
        baseline = dict(baseline_raw)

        analyzed_cols = list(target_fairness.get("analyzed_sensitive_columns") or [])

        intersectional_ml = target_fairness.get("intersectional_ml_results") or {}
        selected_pairs = ctx.get("selected_pairs", [])
        
        # If we have intersectional pairs, prepare them in the mitigated dataset
        if selected_pairs and intersectional_ml.get("status") == "success":
            try:
                df_mitig = pd.read_csv(result["output_file"])
                for pair in selected_pairs:
                    col1, col2 = pair[0], pair[1]
                    combined = f"{col1}_{col2}_combined"
                    df_mitig[combined] = df_mitig[col1].astype(str) + "_" + df_mitig[col2].astype(str)
                    if combined not in analyzed_cols:
                        analyzed_cols.append(combined)
                df_mitig.to_csv(result["output_file"], index=False)
                
                # Merge the baseline metrics for the pairs
                paired_baseline = intersectional_ml.get("fairness_analysis", {})
                if paired_baseline:
                    if "fairness_analysis" not in baseline:
                        baseline["fairness_analysis"] = {}
                    baseline["fairness_analysis"].update(paired_baseline)
            except Exception as e:
                print(f"Warning: Could not create intersectional columns in mitigated dataset: {e}")

        if (
            baseline
            and baseline.get("status") == "success"
            and analyzed_cols
            and target_column
        ):
            import tempfile, shutil
            import pandas as _pd
            from sklearn.model_selection import train_test_split as _tts

            ml_config = ctx.get("ml_config") or {}
            _test_size = ml_config.get("test_size", 0.25)
            _model_type = ml_config.get("model_type", "Random Forest")
            _model_params = ml_config.get("model_params", {})

            mitigated_metrics = None
            eval_tmp_dir = None
            try:
                # Split-first protocol: apply mitigation to training partition only,
                # evaluate on the untouched original test set.
                orig_path = bias_tools._resolve_path(dataset_name)
                orig_df = _pd.read_csv(orig_path).dropna(subset=[target_column])
                try:
                    train_orig, test_orig = _tts(
                        orig_df, test_size=_test_size, random_state=42,
                        stratify=orig_df[target_column]
                    )
                except ValueError:
                    train_orig, test_orig = _tts(
                        orig_df, test_size=_test_size, random_state=42
                    )

                eval_tmp_dir = tempfile.mkdtemp()
                train_tmp_path = os.path.join(eval_tmp_dir, "train_tmp.csv")
                train_orig.to_csv(train_tmp_path, index=False)

                shared_eval = dict(
                    dataset_name=train_tmp_path,
                    target_column=target_column,
                    output_dir=eval_tmp_dir,
                )
                if method == "reweighting":
                    eval_r = bias_tools.apply_reweighting(
                        sensitive_columns=sensitive_columns, **shared_eval
                    )
                elif method == "smote":
                    eval_r = bias_tools.apply_smote(
                        k_neighbors=extra_params.get("k_neighbors", 5),
                        sampling_strategy=extra_params.get("sampling_strategy", "auto"),
                        **shared_eval,
                    )
                elif method == "oversampling":
                    eval_r = bias_tools.apply_oversampling(
                        sampling_strategy=extra_params.get("sampling_strategy", "auto"),
                        **shared_eval,
                    )
                elif method == "undersampling":
                    eval_r = bias_tools.apply_undersampling(
                        sampling_strategy=extra_params.get("sampling_strategy", "auto"),
                        **shared_eval,
                    )
                elif method == "aif360_reweighing":
                    eval_r = bias_tools.apply_aif360_reweighing(
                        sensitive_columns=sensitive_columns, **shared_eval
                    )
                else:
                    eval_r = {"status": "error"}

                if eval_r.get("status") == "success":
                    mitig_train_df = _pd.read_csv(eval_r["output_file"])
                    test_df_eval = test_orig.copy()
                    for pair in selected_pairs:
                        c1, c2 = pair[0], pair[1]
                        comb = f"{c1}_{c2}_combined"
                        if c1 in mitig_train_df.columns and c2 in mitig_train_df.columns:
                            mitig_train_df[comb] = (
                                mitig_train_df[c1].astype(str) + "_" + mitig_train_df[c2].astype(str)
                            )
                        if c1 in test_df_eval.columns and c2 in test_df_eval.columns:
                            test_df_eval[comb] = (
                                test_df_eval[c1].astype(str) + "_" + test_df_eval[c2].astype(str)
                            )
                    # Drop artifact columns added by the mitigation tool that are
                    # absent from the original test split (e.g. combined_group from
                    # reweighting). Mismatched feature counts crash model.predict().
                    test_cols = set(test_df_eval.columns)
                    extra_train_cols = [
                        c for c in mitig_train_df.columns
                        if c not in test_cols and c != target_column and c != "sample_weight"
                    ]
                    if extra_train_cols:
                        mitig_train_df = mitig_train_df.drop(columns=extra_train_cols)
                    mitigated_metrics = fairness_tools.train_and_evaluate_ml_model(
                        target_column=target_column,
                        sensitive_columns=analyzed_cols,
                        test_size=_test_size,
                        model_type=_model_type,
                        model_params=_model_params,
                        train_df=mitig_train_df,
                        test_df=test_df_eval,
                    )
            except Exception as _e:
                print(f"[mitigation] Split-first eval failed ({_e}); falling back.")
            finally:
                if eval_tmp_dir:
                    shutil.rmtree(eval_tmp_dir, ignore_errors=True)

            if mitigated_metrics is None:
                mitigated_metrics = fairness_tools.train_and_evaluate_ml_model(
                    dataset_name=result["output_file"],
                    target_column=target_column,
                    sensitive_columns=analyzed_cols,
                    test_size=_test_size,
                    model_type=_model_type,
                    model_params=_model_params,
                )

            if mitigated_metrics.get("status") == "success":
                result["fairness_comparison"] = _compare_fairness_metrics(
                    baseline, mitigated_metrics, method,
                )

    return result


def _compare_datasets(
    ctx: Dict[str, Any],
    mitigated_dataset: str,
    sensitive_columns: list,
    agent,
    fairness_comparison: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Compare original vs mitigated dataset and have the agent analyse."""

    result = ctx["bias_mitigation_tools"].compare_datasets(
        original_dataset=ctx["dataset_name"],
        mitigated_dataset=mitigated_dataset,
        target_column=ctx["target_column"],
        sensitive_columns=sensitive_columns,
    )

    fairness_section = ""
    if fairness_comparison and fairness_comparison.get("per_attribute_comparison"):
        fairness_section = (
            "\n\n## Fairness Metric Comparison (ML model trained on mitigated data)\n"
            "IMPORTANT: For weight-based techniques the dataset row counts are unchanged, "
            "but the ML model below was trained WITH the sample weights applied — these "
            "metrics reflect the real fairness impact of the mitigation.\n\n"
            f"{safe_json_dumps(fairness_comparison)}"
        )

    prompt = (
        "Analyze the comparison between original and mitigated datasets:\n\n"
        f"{safe_json_dumps(result)}"
        f"{fairness_section}\n\n"
        "Provide a detailed analysis:\n"
        "1. Was the bias mitigation effective? (Yes/No and why)\n"
        "2. What improved? (specific metrics and percentages)\n"
        "3. What remained problematic? (if any)\n"
        "4. Recommendations for further improvements\n\n"
        "IMPORTANT: If this is a weight-based technique (uses_weights=true), focus your "
        "analysis on the Fairness Metric Comparison section above — the raw dataset "
        "distribution will appear unchanged, but the weighted model results show the "
        "real effect. Do NOT conclude 'nothing changed' based on row counts alone."
    )
    result["agent_analysis"] = agent.run(prompt)
    return result


def _compare_fairness_metrics(
    baseline: Dict[str, Any],
    mitigated: Dict[str, Any],
    method_name: str,
) -> Dict[str, Any]:
    """Compute per-attribute fairness improvement between baseline and mitigated."""
    
    if not baseline or baseline.get("status") != "success":
        return {"status": "error", "message": "Invalid baseline metrics"}
    if not mitigated or mitigated.get("status") != "success":
        return {"status": "error", "message": "Invalid mitigated metrics"}

    comparison: Dict[str, Any] = {
        "method": method_name,
        "baseline_metrics": baseline,
        "mitigated_metrics": mitigated,
        "improvements": {},
        "per_attribute_comparison": {},
    }

    baseline_fairness = baseline.get("fairness_analysis", {})
    mitigated_fairness = mitigated.get("fairness_analysis", {})

    for attr in baseline_fairness:
        if attr not in mitigated_fairness:
            continue

        b_metrics = baseline_fairness[attr].get("metrics", {})
        m_metrics = mitigated_fairness[attr].get("metrics", {})

        spd_b = b_metrics.get("statistical_parity_difference", 0)
        spd_m = m_metrics.get("statistical_parity_difference", 0)

        di_b = b_metrics.get("disparate_impact", 0)
        di_m = m_metrics.get("disparate_impact", 0)

        # Per-group breakdown (for table comparison)
        b_groups = baseline_fairness[attr].get("groups", {})
        m_groups = mitigated_fairness[attr].get("groups", {})
        all_groups = sorted(set(b_groups) | set(m_groups))

        group_comparison = []
        _GROUP_METRICS = [
            ("accuracy", "Accuracy"),
            ("f1_macro", "F1 Score"),
            ("positive_rate", "Selection Rate"),
            ("base_rate", "Base Rate"),
            ("fnr", "FNR"),
            ("fpr", "FPR"),
            ("tpr", "TPR"),
            ("tnr", "TNR"),
        ]
        for group in all_groups:
            bg = b_groups.get(group, {})
            mg = m_groups.get(group, {})
            row = {"group": group}
            for key, label in _GROUP_METRICS:
                bv = bg.get(key)
                mv = mg.get(key)
                row[f"baseline_{key}"] = bv
                row[f"mitigated_{key}"] = mv
                if bv is not None and mv is not None:
                    row[f"delta_{key}"] = round(float(mv) - float(bv), 4)
                else:
                    row[f"delta_{key}"] = None
            group_comparison.append(row)

        comparison["per_attribute_comparison"][attr] = {
            "statistical_parity_difference": {
                "baseline": float(spd_b),
                "mitigated": float(spd_m),
                "change": float(spd_b - spd_m),
                "improved": bool(abs(spd_m) < abs(spd_b)),
            },
            "disparate_impact": {
                "baseline": float(di_b),
                "mitigated": float(di_m),
                "change": float(di_m - di_b),
                "improved": bool(abs(1.0 - di_m) < abs(1.0 - di_b)),
            },
            "group_comparison": group_comparison,
        }

    improvements_count = sum(
        1
        for ac in comparison["per_attribute_comparison"].values()
        if ac["statistical_parity_difference"]["improved"]
        or ac["disparate_impact"]["improved"]
    )
    total_metrics = len(comparison["per_attribute_comparison"]) * 2

    if total_metrics == 0:
        comparison["overall_improvement"] = "Unknown"
    elif improvements_count > total_metrics * 0.6:
        comparison["overall_improvement"] = "Significant"
    elif improvements_count > total_metrics * 0.3:
        comparison["overall_improvement"] = "Moderate"
    elif improvements_count > 0:
        comparison["overall_improvement"] = "Minor"
    else:
        comparison["overall_improvement"] = "None or Negative"

    return comparison

