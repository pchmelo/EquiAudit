from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from itertools import combinations as iter_combinations

# Root directory of the project (parent of src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equiaudit.models.agents.function_caller_agent import FunctionCallerAgent
from equiaudit.models.agents.data_analyst_agent import DataAnalystAgent
from equiaudit.models.agents.conversational_agent import ConversationalAgent
from equiaudit.models.agents.humanizer_agent import HumanizerAgent
from equiaudit.models.agents.summary_agent import SummaryAgent
from equiaudit.models.agent_manager import AgentManager
from equiaudit.tools.fairness_tools import FairnessTools
from equiaudit.tools.bias_mitigation_tools import BiasMitigationTools
from equiaudit.tools.discretization_tools import DiscretizationTools

from equiaudit.pipeline.stage import Stage, NavigationAction
from equiaudit.pipeline.config import EVALUATION_STAGES, load_pipeline_config
from equiaudit.pipeline.stages.base import safe_json_dumps
from equiaudit.pipeline.utils import (
    format_mitigation_markdown, 
    format_pair_selection_markdown,
    generate_markdown_report,
    generate_json_data,
    save_fairness_comparison_files,
    generate_detailed_markdown_report
)
from equiaudit.pipeline.stages.pair_selection import build_pair_selection_prompt, parse_pair_selection_response

from equiaudit.reporting.pdf_generator import generate_pdf_bytes


class DatasetEvaluationPipeline:
    """
    Pipeline for evaluating datasets for quality and fairness issues.
    """
    def __init__(
        self,
        config_path: str = None,
        default_model: str = None,
        pipeline_config_path: str = None,
        api_key: str = None,
    ):
        self.fairness_tools = FairnessTools()
        self.bias_mitigation_tools = BiasMitigationTools()
        self.discretization_tools = DiscretizationTools()
        self.agent_manager: Optional[AgentManager] = None
        self.api_key = api_key

        self._stage_definitions = (
            load_pipeline_config(pipeline_config_path)
            if pipeline_config_path
            else EVALUATION_STAGES
        )

        # Determine config path
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "config.yml",
            )
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )
        
        self._init_from_config(config_path, default_model=default_model, api_key=api_key)

        self.current_dataset: Optional[str] = None
        self.user_objective: Optional[str] = None
        self.evaluation_results: Dict[str, Any] = {}

        # Dynamic pipeline state
        self._stages: List[Stage] = []
        self._current_stage_index: int = 0
        self._pipeline_ctx: Dict[str, Any] = {}

        # Pipeline ready


    def _init_from_config(self, config_path: str, default_model: str = None, api_key: str = None):
        self.agent_manager = AgentManager.from_yaml(config_path, api_key=api_key)
        if default_model:
            self.agent_manager.config["default_model"] = default_model
        self.model_client = self.agent_manager.get_client()
        self._initialize_agents()
        # Config loaded

    def _initialize_agents(self):
        self.file_parser_agent = self.agent_manager.get_primary_agent_for_stage("parsing")
        self.inspector_agent = self.agent_manager.get_primary_agent_for_stage("inspection")
        self.bias_mitigation_agent = self.agent_manager.get_primary_agent_for_stage("mitigation")
        self.quality_agent = self.agent_manager.get_primary_agent_for_stage("quality_analysis")
        self.fairness_agent = self.agent_manager.get_primary_agent_for_stage("fairness_analysis")
        self.recommendation_agent = self.agent_manager.get_primary_agent_for_stage("recommendation")

        if self.file_parser_agent is None:
            self.file_parser_agent = FunctionCallerAgent(
                tool_manager=self.fairness_tools, model_client=self.model_client,
                reflect_on_tool_use=True)
        if self.inspector_agent is None:
            self.inspector_agent = FunctionCallerAgent(
                tool_manager=self.fairness_tools, model_client=self.model_client,
                reflect_on_tool_use=True)
        if self.bias_mitigation_agent is None:
            self.bias_mitigation_agent = FunctionCallerAgent(
                tool_manager=self.bias_mitigation_tools, model_client=self.model_client,
                reflect_on_tool_use=True)
        if self.quality_agent is None:
            self.quality_agent = DataAnalystAgent(
                tool_manager=self.fairness_tools, model_client=self.model_client)
        if self.fairness_agent is None:
            self.fairness_agent = DataAnalystAgent(
                tool_manager=self.fairness_tools, model_client=self.model_client)
        if self.recommendation_agent is None:
            self.recommendation_agent = ConversationalAgent(model_client=self.model_client)

        self.humanizer_agent = None
        if self.agent_manager.config.get("use_humanizer", False):
            self.humanizer_agent = HumanizerAgent(model_client=self.model_client)

        # Agents ready


    def build_stages(
        self,
        dataset_name: str,
        target_column: Optional[str] = None,
        user_prompt: str = "",
        report_base_dir: Optional[str] = None,
    ) -> List[Stage]:
        """
        Build the ordered list of stages for an evaluation run.
        """
        self.current_dataset = dataset_name
        self.target_column = target_column
        self.user_objective = user_prompt

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _base = report_base_dir if report_base_dir else os.path.join(BASE_DIR, "reports")
        self.report_dir = os.path.join(_base, f"{dataset_name}_{timestamp}")
        self.images_dir = os.path.join(self.report_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        self.evaluation_results = {
            "dataset": dataset_name,
            "target_column": target_column,
            "user_objective": user_prompt,
            "report_directory": self.report_dir,
            "stages": {},
        }

        # Shared context dict that every stage can read/write
        self._pipeline_ctx = {
            "pipeline": self,
            "dataset_name": dataset_name,
            "target_column": target_column,
            "user_prompt": user_prompt,
            "report_dir": self.report_dir,
            "images_dir": self.images_dir,
            "confirmed_sensitive_columns": None,
            "ml_config": {"enabled": True},
            "selected_pairs": None,
            "mitigation_config": None,
            # Discretization config
            "discretization_enabled": True,
            "discretization_method": "auto",
            "discretization_bins": 5,
            "discretization_threshold": 10,
            # Tool managers / helpers
            "fairness_tools": self.fairness_tools,
            "bias_mitigation_tools": self.bias_mitigation_tools,
            "discretization_tools": self.discretization_tools,
            # Reference to the running results dict
            "results": self.evaluation_results["stages"],
        }

        stages: List[Stage] = []
        for defn in self._stage_definitions:
            if defn.requires_target and not target_column:
                continue
            agent = getattr(self, defn.agent_attr, None)
            stages.append(
                Stage(
                    key=defn.key,
                    name=defn.name,
                    execute_fn=defn.executor,
                    agent=agent,
                    description=defn.description,
                    optional=defn.optional,
                    requires_confirmation=defn.requires_confirmation,
                )
            )

        self._stages = stages
        self._current_stage_index = 0
        return stages


    @property
    def stages(self) -> List[Stage]:
        return self._stages

    @property
    def current_stage_index(self) -> int:
        return self._current_stage_index

    @current_stage_index.setter
    def current_stage_index(self, value: int):
        self._current_stage_index = max(0, min(value, len(self._stages) - 1))

    @property
    def current_stage(self) -> Optional[Stage]:
        if 0 <= self._current_stage_index < len(self._stages):
            return self._stages[self._current_stage_index]
        return None

    @property
    def pipeline_ctx(self) -> Dict[str, Any]:
        return self._pipeline_ctx

    @property
    def is_finished(self) -> bool:
        return self._current_stage_index >= len(self._stages)

    def navigate(self, action: NavigationAction, user_context: str = "") -> Dict[str, Any]:
        """Perform a navigation action and return the stage result."""
        if action == NavigationAction.BACKWARD:
            return self._go_backward(user_context)
        if action == NavigationAction.REPEAT:
            return self._go_repeat(user_context)
        return self._go_forward(user_context)

    def humanize_text(self, text: str) -> str:
        """Process text through the HumanizerAgent if configured."""
        if self.humanizer_agent and text:
            print(f"  Humanizing intermediate output ({len(text)} chars)...")
            try:
                humanized = self.humanizer_agent.run(text)
                if humanized:
                    return humanized
            except Exception as e:
                print(f"  Warning: Humanizer failed ({e}), using original text.")
        return text

    def _humanize_dict(self, data: Any) -> Any:
        if isinstance(data, dict):
            for k, v in data.items():
                if k in ["agent_analysis", "agent_response", "recommendations"] and isinstance(v, str):
                    data[k] = self.humanize_text(v)
                elif isinstance(v, (dict, list)):
                    self._humanize_dict(v)
        elif isinstance(data, list):
            for item in data:
                self._humanize_dict(item)
        return data

    def _go_forward(self, user_context: str = "") -> Dict[str, Any]:
        if self._current_stage_index >= len(self._stages):
            return {"status": "finished", "message": "All stages completed."}
        stage = self._stages[self._current_stage_index]
        stage.user_context = user_context or None
        result = stage.execute(self._pipeline_ctx)

        if self.humanizer_agent:
            self._humanize_dict(result)

        self.evaluation_results["stages"][stage.key] = result
        self._current_stage_index += 1
        
        # After sensitive attribute detection, handle pair selection (GUI mode)
        if stage.key == "3_sensitive":
            user_pairs = self._pipeline_ctx.get("user_specified_pairs")
            max_pairs = self._pipeline_ctx.get("max_pairs")
            # Only run if not already set (avoid overwriting user's chat-based overrides)
            if "selected_pairs" not in self._pipeline_ctx or self._pipeline_ctx["selected_pairs"] is None:
                self._handle_pair_selection(user_pairs, max_pairs)
        
        return result

    def _go_backward(self, user_context: str = "") -> Dict[str, Any]:
        if self._current_stage_index <= 0:
            return {"status": "info", "message": "Already at the first stage."}
        self._current_stage_index -= 1
        stage = self._stages[self._current_stage_index]
        stage.reset()
        self.evaluation_results["stages"].pop(stage.key, None)
        return {
            "status": "rewound",
            "message": f"Returned to **{stage.name}**. It will re-run on the next forward.",
        }

    def _go_repeat(self, user_context: str = "") -> Dict[str, Any]:
        idx = max(0, self._current_stage_index - 1)
        stage = self._stages[idx]
        stage.reset()
        stage.user_context = user_context or None
        self.evaluation_results["stages"].pop(stage.key, None)
        result = stage.execute(self._pipeline_ctx)

        if self.humanizer_agent:
            self._humanize_dict(result)

        self.evaluation_results["stages"][stage.key] = result
        return result

    # ==================================================================
    # Terminal-mode entry point
    # ==================================================================

    def evaluate_dataset(
        self,
        user_prompt: str,
        confirmed_sensitive: list = None,
        sensitive_pairs: list = None,
        ml_config: dict = None,
        max_pairs: int = None,
        mitigation_config: dict = None,
        discretization_config: dict = None,
        report_base_dir: str = None,
    ) -> Dict[str, Any]:
        """Run the full pipeline in one shot (used by terminal mode).
        
        Args:
            user_prompt: The evaluation objective/prompt.
            confirmed_sensitive: Pre-confirmed sensitive columns (skip detection).
            sensitive_pairs: Pre-defined pairs for intersectional analysis.
                            Each pair is a tuple/list of two column names.
                            If provided, uses these pairs directly (restricted mode).
                            If None, pairs are auto-selected based on max_pairs.
            ml_config: ML model configuration for fairness metrics.
            max_pairs: Maximum number of sensitive attribute pairs to analyze.
                      Only used when sensitive_pairs is None.
                      If set, the agent selects the most important pairs.
            mitigation_config: Bias mitigation configuration dict with format
                              {"methods": {"Reweighting": {}, "SMOTE": {}}}.
                              If None, the mitigation stage is skipped.
            discretization_config: Discretization configuration dict with keys:
                              enabled (bool), method (str), bins (int), threshold (int).
        """
        dataset_name = self._extract_dataset_name(user_prompt)
        target_column = self._extract_target_column(user_prompt)
        self.build_stages(dataset_name, target_column, user_prompt, report_base_dir=report_base_dir)

        if confirmed_sensitive:
            self._pipeline_ctx["confirmed_sensitive_columns"] = confirmed_sensitive
        if ml_config:
            self._pipeline_ctx["ml_config"] = ml_config
        if mitigation_config:
            self._pipeline_ctx["mitigation_config"] = mitigation_config
        
        # Apply discretization config
        if discretization_config:
            for key, value in discretization_config.items():
                self._pipeline_ctx[key] = value

        # Store pair configuration in context
        self._pipeline_ctx["max_pairs"] = max_pairs
        self._pipeline_ctx["user_specified_pairs"] = sensitive_pairs

        # Count executable stages (skip mitigation if not configured)
        run_mitigation = bool(mitigation_config)
        executable_stages = [
            s for s in self._stages
            if s.key != "6_bias_mitigation" or run_mitigation
        ]
        total_stages = len(executable_stages)
        current_num = 0

        while not self.is_finished:
            stage = self.current_stage
            if stage.key == "6_bias_mitigation" and not run_mitigation:
                self._current_stage_index += 1
                continue
            current_num += 1
            print(f"[{current_num}/{total_stages}] {stage.name}")
            self.navigate(NavigationAction.FORWARD)

        print("Done.")

        return self.evaluation_results
    
    def _handle_pair_selection(self, sensitive_pairs: list = None, max_pairs: int = None) -> None:
        """
        Handle pair selection after sensitive attribute detection.
        
        Args:
            sensitive_pairs: User-specified pairs for restricted mode. If provided,
                           these pairs are used directly without agent selection.
            max_pairs: Maximum pairs for auto mode. If set, agent selects best pairs.
        """
        
        results = self.evaluation_results.get("stages", {})
        sensitive_cols = list(
            results.get("3_sensitive", {}).get("sensitive_columns", [])
        )
        
        # Exclude target column if present
        target = self._pipeline_ctx.get("target_column")
        if target and target in sensitive_cols:
            sensitive_cols = [c for c in sensitive_cols if c != target]
        
        if len(sensitive_cols) < 2:
            return  # No pairs possible
        
        all_pairs = list(iter_combinations(sensitive_cols, 2))
        
        if sensitive_pairs is not None:
            # Restricted mode: use user-specified pairs directly.
            # Pairs may reference any dataset column, not just the detected
            # sensitive columns. Validation against actual column existence
            # happens later in the fairness tool itself.
            valid_pairs = [(pair[0], pair[1]) for pair in sensitive_pairs if len(pair) == 2]
            
            if valid_pairs:
                self._pipeline_ctx["selected_pairs"] = valid_pairs
                self._pipeline_ctx["pair_selection_reasoning"] = (
                    f"User-specified pairs (restricted mode): {len(valid_pairs)} pair(s) selected."
                )
                self._save_pair_selection_to_results(
                    valid_pairs,
                    self._pipeline_ctx["pair_selection_reasoning"],
                    all_pairs,
                    len(valid_pairs),
                    mode="restricted"
                )
                print(f"  Using user-specified pairs: {[f'{p[0]}+{p[1]}' for p in valid_pairs]}")
            else:
                print("  Warning: No valid pairs from user specification, using all pairs")
                self._pipeline_ctx["selected_pairs"] = all_pairs
        
        elif max_pairs is not None:
            # Auto mode with limit: use agent to select best pairs
            self._select_best_pairs(max_pairs)
        
        else:
            # Auto mode without limit: use all pairs
            self._pipeline_ctx["selected_pairs"] = all_pairs
            self._pipeline_ctx["pair_selection_reasoning"] = (
                f"All {len(all_pairs)} pairs selected (no limit specified)."
            )
    
    def _select_best_pairs(self, max_pairs: int) -> None:
        """
        Use the agent to intelligently select the most important pairs
        for intersectional fairness analysis.
        """
        
        results = self.evaluation_results.get("stages", {})
        sensitive_cols = list(
            results.get("3_sensitive", {}).get("sensitive_columns", [])
        )
        
        # Exclude target column if present
        target = self._pipeline_ctx.get("target_column")
        if target and target in sensitive_cols:
            sensitive_cols = [c for c in sensitive_cols if c != target]
        
        if len(sensitive_cols) < 2:
            return  # No pairs possible
        
        all_pairs = list(iter_combinations(sensitive_cols, 2))
        
        if len(all_pairs) <= max_pairs:
            # No need to select, use all pairs
            self._pipeline_ctx["selected_pairs"] = all_pairs
            self._pipeline_ctx["pair_selection_reasoning"] = (
                f"All {len(all_pairs)} pairs selected (within max_pairs={max_pairs} limit)."
            )
            return
        
        # Ask agent to select best pairs
        print(f"  Selecting {max_pairs} most important pairs from {len(all_pairs)} possible...")
        
        prompt = build_pair_selection_prompt(sensitive_cols, all_pairs, max_pairs)
        
        try:
            response = self.recommendation_agent.run(prompt)
            selected, reasoning = parse_pair_selection_response(response, all_pairs, max_pairs)
            
            if not reasoning:
                reasoning = "Agent selected these pairs based on fairness analysis criteria."
            
            self._pipeline_ctx["selected_pairs"] = selected
            self._pipeline_ctx["pair_selection_reasoning"] = reasoning
            
            # Save to evaluation results for the report
            self._save_pair_selection_to_results(selected, reasoning, all_pairs, max_pairs)
            
            print(f"  Selected pairs: {[f'{p[0]}+{p[1]}' for p in selected]}")
            
        except Exception as e:
            # Fallback: use first N pairs
            print(f"  Warning: Pair selection failed ({e}), using first {max_pairs} pairs")
            selected = list(all_pairs[:max_pairs])
            self._pipeline_ctx["selected_pairs"] = selected
            self._pipeline_ctx["pair_selection_reasoning"] = f"Automatic selection: first {max_pairs} pairs (fallback)."
            self._save_pair_selection_to_results(selected, self._pipeline_ctx["pair_selection_reasoning"], all_pairs, max_pairs)
    
    def _save_pair_selection_to_results(self, selected: list, reasoning: str, all_pairs: list, max_pairs: int, mode: str = "auto") -> None:
        """Save pair selection info to evaluation results for the report.
        
        Args:
            selected: List of selected pairs.
            reasoning: Explanation for why these pairs were selected.
            all_pairs: All possible pairs from detected sensitive columns.
            max_pairs: The max_pairs limit that was applied.
            mode: Selection mode - "auto" (agent-selected) or "restricted" (user-specified).
        """
        # Update the sensitive stage results with pair selection info
        if "3_sensitive" in self.evaluation_results.get("stages", {}):
            self.evaluation_results["stages"]["3_sensitive"]["pair_selection"] = {
                "mode": mode,
                "max_pairs_limit": max_pairs,
                "total_possible_pairs": len(all_pairs),
                "selected_pairs": [f"{p[0]} + {p[1]}" for p in selected],
                "reasoning": reasoning,
            }

        return self.evaluation_results

    # ---- prompt parsing helpers (terminal mode) ----------------------

    @staticmethod
    def _extract_dataset_name(user_prompt: str) -> str:
        prompt_lower = user_prompt.lower()
        words = user_prompt.split()

        for word in words:
            if ".csv" in word:
                return word.strip("'\"")

        common = ["adult-all.csv", "adult-all", "adult", "census", "credit", "compas", "german", "bank"]
        for ds in common:
            if ds in prompt_lower:
                return ds if ".csv" in ds else f"{ds}.csv"

        skip = {"audit", "analyze", "evaluate", "check", "inspect", "dataset", "the", "a", "an", "target"}
        remaining = [w.strip("'\"") for w in words if w.lower() not in skip and len(w) > 2]
        if remaining:
            return remaining[0]
        return words[0].strip("'\"") if words else "dataset"

    @staticmethod
    def _extract_target_column(user_prompt: str) -> Optional[str]:
        prompt_lower = user_prompt.lower()
        if "target=" in prompt_lower or "target:" in prompt_lower:
            m = re.search(r"target[=:]\s*([a-zA-Z_-]+)", user_prompt, re.IGNORECASE)
            if m:
                return m.group(1)
        m = re.search(r"target\s+(?:is|as)\s+([a-zA-Z_-]+)", user_prompt, re.IGNORECASE)
        if m:
            return m.group(1)
        for t in ("income", "salary", "class", "label", "outcome", "result", "prediction"):
            if t in prompt_lower:
                return t
        return None

    # ==================================================================
    # Report generation
    # ==================================================================

    def generate_report(self, output_path: str = None) -> str:
        """Generate both markdown report and JSON data file."""        
        
        # 1) Generate Executive Summary if configured
        generate_exec_summary = True
        if hasattr(self, "agent_manager") and self.agent_manager and hasattr(self.agent_manager, "config"):
            generate_exec_summary = self.agent_manager.config.get("generate_executive_summary", True)
            
        if generate_exec_summary:
            print("Generating Executive Summary...")
            try:
                client = self.agent_manager.get_client() if self.agent_manager else None
                summary_agent = SummaryAgent(model_client=client)
                
                # We can feed it a slimmed-down JSON, or the whole evaluation_results
                # Since the model might choke on the massive token count of the raw dataframe dumps,
                # we only send the relevant JSON bits (Stages 4, 4.5, 6).
                relevant_data = {
                    "Stage_4": self.evaluation_results.get("stages", {}).get("4_imbalance", {}),
                    "Stage_4_5": self.evaluation_results.get("stages", {}).get("4_5_target_fairness", {}),
                    "Stage_6": self.evaluation_results.get("stages", {}).get("6_bias_mitigation", {})
                }
                
                raw_payload = safe_json_dumps(relevant_data)
                executive_summary = summary_agent.run(raw_payload)
                self.evaluation_results["executive_summary"] = executive_summary
            except Exception as e:
                print(f"Warning: Could not generate Executive Summary: {e}")
                
        md_path = os.path.join(self.report_dir, "evaluation_report.md")
        json_path = os.path.join(self.report_dir, "stage_data.json")
        
        md_content = generate_markdown_report(self)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Markdown report saved: {md_path}")
        
        json_content = generate_json_data(self)
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(safe_json_dumps(json_content))
        print(f"JSON data saved: {json_path}")
        
        save_fairness_comparison_files(self)
        
        # Generate and save PDF inside the report folder
        try:
            pdf_path = os.path.join(self.report_dir, "evaluation_report.pdf")
            pdf_bytes = generate_pdf_bytes(md_path)
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            print(f"PDF report saved: {pdf_path}")
        except Exception as e:
            print(f"Warning: Could not generate PDF: {e}")
            
        # Check config to see if detailed report should be generated
        generate_detailed = True
        if hasattr(self, "agent_manager") and self.agent_manager and hasattr(self.agent_manager, "config"):
            generate_detailed = self.agent_manager.config.get("generate_detailed_report", True)
            
        if generate_detailed:
            try:
                detailed_md_content = generate_detailed_markdown_report(self)
                detailed_md_path = os.path.join(self.report_dir, "detailed_metrics_report.md")
                with open(detailed_md_path, "w", encoding="utf-8") as f:
                    f.write(detailed_md_content)
                print(f"Detailed Markdown report saved: {detailed_md_path}")
                
                detailed_pdf_path = os.path.join(self.report_dir, "detailed_metrics_report.pdf")
                detailed_pdf_bytes = generate_pdf_bytes(detailed_md_path)
                with open(detailed_pdf_path, "wb") as f:
                    f.write(detailed_pdf_bytes)
                print(f"Detailed PDF report saved: {detailed_pdf_path}")
            except Exception as e:
                print(f"Warning: Could not generate Detailed PDF: {e}")
        
        return md_content


