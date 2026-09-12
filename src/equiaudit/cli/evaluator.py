from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import shutil
import traceback
from equiaudit.models.agents.base_agent import APIError

import pandas as pd
from equiaudit.pipeline.pipeline import DatasetEvaluationPipeline
from equiaudit.pipeline.utils import TECHNIQUE_DISPLAY

# Root of the equiaudit package (parent of cli/)
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class EvaluationResult:
    """Result of a fairness evaluation run."""
    
    success: bool
    dataset: str
    target_column: Optional[str]
    report_dir: str
    pdf_path: Optional[str]
    markdown_path: Optional[str]
    json_path: Optional[str]
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    stages_completed: List[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"EvaluationResult({status}, dataset={self.dataset}, report_dir={self.report_dir})"


@dataclass
class VerificationResult:
    """Result of configuration verification."""
    
    success: bool
    config_path: str
    model_name: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        status = "OK" if self.success else "FAILED"
        return f"VerificationResult({status}, model={self.model_name})"


class FairnessEvaluator:
    """
    Main class for running fairness evaluations on datasets.
    
    This class wraps the DatasetEvaluationPipeline and provides a simplified
    interface for headless (non-GUI) evaluation runs.
    
    Example:
        >>> evaluator = FairnessEvaluator(config_path="models/config.yml")
        >>> result = evaluator.verify()
        >>> if result.success:
        ...     eval_result = evaluator.evaluate("adult-all.csv", target="Income")
        ...     print(f"Report saved to: {eval_result.pdf_path}")
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize the FairnessEvaluator.
        
        Args:
            config_path: Path to the YAML configuration file. If None, uses default.
            output_dir: Base directory for reports. If None, uses project's reports/ folder.
            model: Override the default model from config. If None, uses config default.
            api_key: API key for the model provider (OpenRouter/Gemini). Overrides env variable.
            verbose: Whether to print progress messages.
        """
        self.verbose = verbose
        self.model_override = model
        self.api_key = api_key
        
        # Resolve config path
        if config_path is None:
            config_path = os.path.join(_SRC_DIR, "models", "config.yml")
        self.config_path = os.path.abspath(config_path)

        # Resolve output directory
        if output_dir is None:
            # Check config file for report_dir key
            config_report_dir = None
            try:
                with open(self.config_path, encoding="utf-8") as _f:
                    _cfg = yaml.safe_load(_f) or {}
                config_report_dir = _cfg.get("report_dir") or None
            except Exception:
                pass

            if config_report_dir:
                # Relative paths are resolved relative to the config file's directory
                config_dir = os.path.dirname(self.config_path)
                output_dir = os.path.join(config_dir, config_report_dir) if not os.path.isabs(config_report_dir) else config_report_dir
            else:
                # Default: reports/ next to the config file
                output_dir = os.path.join(os.path.dirname(self.config_path), "reports")
        self.output_dir = os.path.abspath(output_dir)
        
        # Pipeline will be initialized lazily
        self._pipeline = None
        self._initialized = False
        
        # Verbose initialization happens when evaluation starts
    
    def _ensure_initialized(self) -> None:
        """Lazily initialize the pipeline."""
        if self._initialized:
            return
                
        self._pipeline = DatasetEvaluationPipeline(
            config_path=self.config_path,
            default_model=self.model_override,
            api_key=self.api_key,
        )
        self._initialized = True
    
    def verify(self) -> VerificationResult:
        """
        Verify that the configuration is valid and the system is ready.
        
        Returns:
            VerificationResult with status and any errors/warnings.
        """
        errors: List[str] = []
        warnings: List[str] = []
        model_name = "unknown"
        
        # Check config file exists
        if not os.path.exists(self.config_path):
            errors.append(f"Configuration file not found: {self.config_path}")
            return VerificationResult(
                success=False,
                config_path=self.config_path,
                model_name=model_name,
                errors=errors,
                warnings=warnings,
            )
        
        # Try to initialize pipeline
        try:
            self._ensure_initialized()
            model_name = self._pipeline.agent_manager.config.get("default_model", "unknown")
            
            if self.verbose:
                print(f"Configuration loaded successfully")
                print(f"  Default model: {model_name}")
        except Exception as e:
            errors.append(f"Failed to initialize pipeline: {str(e)}")
            return VerificationResult(
                success=False,
                config_path=self.config_path,
                model_name=model_name,
                errors=errors,
                warnings=warnings,
            )
        
        # Check output directory
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            if self.verbose:
                print(f"Output directory ready: {self.output_dir}")
        except Exception as e:
            errors.append(f"Cannot create output directory: {str(e)}")
        
        # Check API key if using OpenRouter
        model_config = self._pipeline.agent_manager.config.get("models", {}).get(model_name, {})
        provider = model_config.get("provider", "")
        
        if provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                warnings.append("OPENROUTER_API_KEY environment variable not set")
        elif provider == "gemini":
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                warnings.append("GOOGLE_API_KEY environment variable not set")
        
        success = len(errors) == 0
        
        if self.verbose:
            if success:
                print("\nVerification: PASSED")
                if warnings:
                    print(f"  Warnings: {len(warnings)}")
                    for w in warnings:
                        print(f"    - {w}")
            else:
                print("\nVerification: FAILED")
                for e in errors:
                    print(f"  ERROR: {e}")
        
        return VerificationResult(
            success=success,
            config_path=self.config_path,
            model_name=model_name,
            errors=errors,
            warnings=warnings,
        )
    
    def evaluate(
        self,
        data: Union[str, Path, pd.DataFrame],
        target: Optional[str] = None,
        objective: Optional[str] = None,
        sensitive_columns: Optional[List[str]] = None,
        sensitive_pairs: Optional[List[List[str]]] = None,
        mitigation_techniques: Optional[List[str]] = None,
        ml_config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        generate_pdf: bool = True,
        max_pairs: Optional[int] = None,
    ) -> EvaluationResult:
        """
        Run a fairness evaluation on a dataset.
        
        Args:
            data: Path to CSV file, or a pandas DataFrame.
            target: Target column name for classification. Auto-detected if None.
            objective: Custom objective/prompt for the evaluation.
            sensitive_columns: List of sensitive attribute columns to analyze.
                              If None, uses config settings (auto-detect or restricted list).
            sensitive_pairs: List of attribute pairs to analyze for intersectionality.
                            Each pair is a list of two column names, e.g. [["Sex", "Race"]].
                            If None, uses config settings (auto-select or restricted list).
            mitigation_techniques: List of bias mitigation techniques to apply.
                                  Options: "reweighting", "smote", "resampling",
                                  "oversampling", "undersampling".
                                  If None, reads from config. If config also has none,
                                  mitigation stage is skipped entirely.
            ml_config: ML model configuration for fairness metrics.
            output_dir: Override output directory for this run.
            generate_pdf: Whether to generate PDF report.
            max_pairs: Maximum number of sensitive attribute pairs to analyze.
                      Only used when sensitive_pairs is None and config type is "auto".
                      If None, uses value from config.
        
        Returns:
            EvaluationResult with paths to generated reports.
        """
        self._ensure_initialized()
        
        warnings: List[str] = []
        stages_completed: List[str] = []
        
        # Resolve data input
        if isinstance(data, pd.DataFrame):
            # Save DataFrame to temporary CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_csv_name = f"temp_dataset_{timestamp}.csv"
            data_dir = os.path.join(_SRC_DIR, "data")
            os.makedirs(data_dir, exist_ok=True)
            temp_csv_path = os.path.join(data_dir, temp_csv_name)
            data.to_csv(temp_csv_path, index=False)
            dataset_name = temp_csv_name
            if self.verbose:
                print(f"DataFrame saved to: {temp_csv_path}")
        else:
            data_path = Path(data)
            if not data_path.exists():
                # Try looking in the bundled data directory
                data_path = Path(_SRC_DIR) / "data" / data_path.name
            
            if not data_path.exists():
                return EvaluationResult(
                    success=False,
                    dataset=str(data),
                    target_column=target,
                    report_dir="",
                    pdf_path=None,
                    markdown_path=None,
                    json_path=None,
                    error=f"Dataset not found: {data}",
                )
            
            dataset_name = data_path.name
            
            # Copy to data/ if not already there
            expected_path = Path(_SRC_DIR) / "data" / dataset_name
            if data_path.resolve() != expected_path.resolve():
                os.makedirs(expected_path.parent, exist_ok=True)
                shutil.copy2(data_path, expected_path)
        
        try:
            # Load evaluation config directly from YAML file
            cfg: Dict[str, Any] = {}
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as _f:
                    cfg = yaml.safe_load(_f) or {}
            
            # Fall back to config target_column if not explicitly provided
            if target is None:
                target = cfg.get("target_column") or None
                if self.verbose and target:
                    print(f"Using target column from config: {target}")
            
            # Build objective prompt AFTER target fallback
            if objective is None:
                target_str = f"Target: {target}. " if target else ""
                objective = (
                    f"Evaluate the dataset '{dataset_name}' for data quality and fairness issues. "
                    f"{target_str}Provide a detailed report highlighting any problems found and suggestions for improvement."
                )
        
            if self.verbose:
                print(f"\nEvaluating '{dataset_name}' (target: {target or 'auto-detect'})")
            
            # Handle sensitive attribute analysis configuration
            sens_attr_config = cfg.get("sensitive_attribute_analysis", {})
            if sensitive_columns is None and sens_attr_config.get("type") == "restricted":
                # Use restricted list from config
                sensitive_columns = sens_attr_config.get("attributes", [])
                if self.verbose and sensitive_columns:
                    print(f"Using restricted sensitive attributes: {sensitive_columns}")
            
            # Handle pair evaluation configuration
            pair_config = cfg.get("pair_evaluation", {})
            final_sensitive_pairs = sensitive_pairs  # Start with explicit parameter
            
            if final_sensitive_pairs is None and pair_config.get("type") == "restricted":
                # Use restricted pairs from config
                final_sensitive_pairs = pair_config.get("pairs", [])
                if self.verbose and final_sensitive_pairs:
                    print(f"Using restricted pairs: {final_sensitive_pairs}")
            
            # Load max_pairs from config if not provided
            if max_pairs is None:
                max_pairs = pair_config.get("max_pairs")
            
            # Handle mitigation techniques configuration
            final_mitigation_config = None
            techniques = mitigation_techniques
            if techniques is None:
                mitigation_cfg = cfg.get("mitigation_techniques", {})
                techniques = mitigation_cfg.get("techniques", [])
            if techniques:
                methods = {}
                for t in techniques:
                    display = TECHNIQUE_DISPLAY.get(t.lower())
                    if display:
                        methods[display] = {}
                    else:
                        warnings.append(f"Unknown mitigation technique '{t}', skipped.")
                if methods:
                    final_mitigation_config = {"methods": methods}
                    if self.verbose:
                        print(f"Applying mitigation: {list(methods.keys())}")
            
            final_ml_config = ml_config
            if final_ml_config is None:
                ml_eval_cfg = cfg.get("ml_evaluation", {})
                if ml_eval_cfg:
                    m_type = ml_eval_cfg.get("model_type", "Random Forest")
                    m_params = ml_eval_cfg.get("model_params", {}).get(m_type, {})
                    final_ml_config = {
                        "enabled": True,
                        "model_type": m_type,
                        "model_params": m_params,
                        "test_size": ml_eval_cfg.get("test_size", 0.25)
                    }
                    
            # Build discretization config from YAML
            disc_setting = sens_attr_config.get("discretization", "enable")
            disc_enabled = disc_setting in ("enable", True, "true", "yes")
            disc_method = sens_attr_config.get("method", "auto")
            disc_bins = int(sens_attr_config.get("number_of_bins", 5))
            disc_threshold = int(sens_attr_config.get("continuous_threshold", 10))
            discretization_config = {
                "discretization_enabled": disc_enabled,
                "discretization_method": disc_method,
                "discretization_bins": disc_bins,
                "discretization_threshold": disc_threshold,
            }
            if self.verbose and disc_enabled:
                print(f"Discretization: method={disc_method}, bins={disc_bins}, threshold={disc_threshold}")

            # Run the pipeline
            self._pipeline.evaluate_dataset(
                user_prompt=objective,
                confirmed_sensitive=sensitive_columns,
                sensitive_pairs=final_sensitive_pairs,
                ml_config=final_ml_config,
                max_pairs=max_pairs,
                mitigation_config=final_mitigation_config,
                discretization_config=discretization_config,
                report_base_dir=self.output_dir,
            )
            
            # Generate report
            self._pipeline.generate_report()
            
            report_dir = self._pipeline.report_dir
            md_path = os.path.join(report_dir, "evaluation_report.md")
            json_path = os.path.join(report_dir, "stage_data.json")
            
            # Determine PDF path
            pdf_path = None
            if generate_pdf:
                pdf_path = os.path.join(report_dir, "evaluation_report.pdf")
                if not os.path.exists(pdf_path):
                    pdf_path = None
                    warnings.append("PDF generation may have failed")
            
            # Get completed stages
            stages_completed = list(self._pipeline.evaluation_results.get("stages", {}).keys())
            
            if self.verbose:
                print(f"\nEvaluation completed successfully!")
                print(f"  Report directory: {report_dir}")
                print(f"  Stages completed: {len(stages_completed)}")
                if pdf_path:
                    print(f"  PDF report: {pdf_path}")
            
            return EvaluationResult(
                success=True,
                dataset=dataset_name,
                target_column=target or self._pipeline.target_column,
                report_dir=report_dir,
                pdf_path=pdf_path,
                markdown_path=md_path,
                json_path=json_path,
                warnings=warnings,
                stages_completed=stages_completed,
            )
        
        except Exception as e:
            
            # Handle API errors with clean messages
            if isinstance(e, APIError):
                error_msg = str(e)
                if self.verbose:
                    print(f"\nError: {error_msg}")
            else:
                error_msg = f"{type(e).__name__}: {str(e)}"
                if self.verbose:
                    print(f"\nEvaluation failed: {error_msg}")
                    traceback.print_exc()
            
            return EvaluationResult(
                success=False,
                dataset=dataset_name,
                target_column=target,
                report_dir=getattr(self._pipeline, "report_dir", ""),
                pdf_path=None,
                markdown_path=None,
                json_path=None,
                error=error_msg,
                warnings=warnings,
                stages_completed=stages_completed,
            )

