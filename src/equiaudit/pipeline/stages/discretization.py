from __future__ import annotations

import re
from typing import Any, Dict, List

from equiaudit.models.agents.base_agent import APIError
from equiaudit.pipeline.stages.base import BaseStageExecutor, safe_json_dumps


"""
Stage 3.5 – Sensitive Attribute Discretization.
Identify continuous sensitive columns and discretize them into categorical bins.
"""


class DiscretizationStage(BaseStageExecutor):

    def __call__(self, stage, ctx: Dict[str, Any]) -> Dict[str, Any]:
        results = ctx["results"]
        discretization_tools = ctx["discretization_tools"]

        # ── Get the confirmed sensitive columns ──────────────────────
        # Priority:
        # 1. discretization_sensitive_columns: GUI-selected subset for Stage 3.5
        # 2. confirmed_sensitive_columns: manually chosen in Stage 3 (manual mode)
        # 3. auto-detected columns from Stage 3 results
        user_disc_attrs = (
            ctx.get("discretization_sensitive_columns")
            or ctx.get("confirmed_sensitive_columns")
        )
        if user_disc_attrs:
            sensitive_cols = list(user_disc_attrs)
        else:
            sensitive_cols = list(
                results.get("3_sensitive", {}).get("sensitive_columns", [])
            )
        if not sensitive_cols:
            return {
                "status": "skipped",
                "message": "No sensitive columns identified — discretization skipped.",
            }

        # ── Check if discretization is enabled ────────────────────────
        disc_enabled = ctx.get("discretization_enabled", True)
        if not disc_enabled:
            return {
                "status": "skipped",
                "message": "Discretization is disabled in the configuration.",
            }

        # ── Identify continuous columns ───────────────────────────────
        unique_threshold = ctx.get("discretization_threshold", 10)
        identification = discretization_tools.identify_continuous_sensitive(
            dataset_name=ctx["dataset_name"],
            sensitive_columns=sensitive_cols,
            unique_threshold=unique_threshold,
        )

        if identification.get("status") != "success":
            return {
                "status": "error",
                "message": f"Failed to identify continuous columns: {identification.get('message')}",
            }

        continuous_cols = identification.get("continuous_columns", [])
        if not continuous_cols:
            return {
                "status": "success",
                "message": "No continuous sensitive columns found — no discretization needed.",
                "identification": identification,
                "discretized_columns": [],
            }

        # ── Read discretization config ────────────────────────────────
        method = ctx.get("discretization_method", "auto")
        n_bins = ctx.get("discretization_bins", 5)

        # ── Discretize each continuous column ─────────────────────────
        discretized_info: List[Dict[str, Any]] = []
        agent_reasoning = ""
        current_dataset = ctx["dataset_name"]

        for col_stats in continuous_cols:
            col_name = col_stats["column_name"]

            if method == "auto":
                # Ask the agent to decide binning strategy
                prompt = self._build_auto_prompt(col_stats)
                prompt = self._append_user_context(prompt, stage.user_context)

                try:
                    agent_response = stage.agent.run(prompt, max_tokens=4096)
                except (APIError, Exception) as exc:
                    # Model failed to respond; fall back to equal-width so the
                    # pipeline continues rather than crashing.
                    print(
                        f"[Stage 3.5] WARNING: Agent call failed for '{col_name}' "
                        f"({exc}). Falling back to equal_width with {n_bins} bins."
                    )
                    result = discretization_tools.discretize_column_manual(
                        dataset_name=current_dataset,
                        column_name=col_name,
                        method="equal_width",
                        number_of_bins=n_bins,
                    )
                    result["method"] = "equal_width (fallback — agent unavailable)"
                    agent_reasoning += f"\n### {col_name}\nAgent unavailable; applied equal-width binning with {n_bins} bins.\n"
                    if result.get("status") == "success":
                        discretized_info.append(result)
                        current_dataset = result["output_dataset"]
                    else:
                        discretized_info.append({
                            "column": col_name,
                            "status": "error",
                            "message": result.get("message", "Unknown error"),
                        })
                    continue

                # Parse the agent's bin edges from the response
                bin_edges = self._parse_bin_edges(agent_response, col_stats)
                labels = None

                if bin_edges:
                    labels = self._parse_labels(agent_response, len(bin_edges) - 1)
                    result = discretization_tools.discretize_column_auto(
                        dataset_name=current_dataset,
                        column_name=col_name,
                        bin_edges=bin_edges,
                        labels=labels,
                    )
                else:
                    # Fallback: agent didn't produce parseable BIN_EDGES.
                    # Log clearly and fall back to equal-width.
                    n = self._parse_bin_count(agent_response, default=n_bins)
                    print(
                        f"[Stage 3.5] WARNING: Agent did not return parseable BIN_EDGES "
                        f"for '{col_name}'. Falling back to equal_width with {n} bins."
                    )
                    result = discretization_tools.discretize_column_manual(
                        dataset_name=current_dataset,
                        column_name=col_name,
                        method="equal_width",
                        number_of_bins=n,
                    )
                    # Mark result so the report clearly shows this was a fallback
                    result["method"] = "equal_width (fallback — agent failed to return BIN_EDGES)"

                # Build a clean reasoning summary for the report
                reasoning_summary = self._extract_reasoning_summary(
                    agent_response, col_name, bin_edges, labels, result
                )
                agent_reasoning += f"\n### {col_name}\n{reasoning_summary}\n"

            else:
                # equal_width or equal_frequency
                result = discretization_tools.discretize_column_manual(
                    dataset_name=current_dataset,
                    column_name=col_name,
                    method=method,
                    number_of_bins=n_bins,
                )

            if result.get("status") == "success":
                discretized_info.append(result)
                # Update dataset_name to the discretized file for the next column
                current_dataset = result["output_dataset"]
            else:
                discretized_info.append({
                    "column": col_name,
                    "status": "error",
                    "message": result.get("message", "Unknown error"),
                })

        # ── Update pipeline context ───────────────────────────────────
        # Point downstream stages to the discretized dataset
        if current_dataset != ctx["dataset_name"]:
            ctx["dataset_name"] = current_dataset

        # ── Build agent analysis for the report ───────────────────────
        if method != "auto":
            # For non-auto modes, produce a summary via the agent
            summary_prompt = self._build_summary_prompt(
                method, n_bins, continuous_cols, discretized_info,
            )
            summary_prompt = self._append_user_context(summary_prompt, stage.user_context)
            agent_analysis = stage.agent.run(summary_prompt, max_tokens=2048)
        else:
            agent_analysis = agent_reasoning

        return {
            "status": "success",
            "identification": identification,
            "discretized_columns": discretized_info,
            "method": method,
            "agent_analysis": agent_analysis,
            "updated_dataset": current_dataset,
        }

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_auto_prompt(col_stats: Dict[str, Any]) -> str:
        return (
            "You are a fairness analysis expert. Discretize this continuous sensitive "
            "attribute into meaningful categorical bins for fairness analysis.\n\n"
            "Column: {name} | Range: [{min}, {max}] | Mean: {mean} | "
            "Median: {median} | Std: {std} | Unique: {unique} | Rows: {total}\n"
            "Sample values: {samples}\n\n"
            "Choose 3-5 bins with human-readable labels based on:\n"
            "- The column's real-world meaning (e.g. age → Young/Middle-Aged/Senior)\n"
            "- The data distribution (avoid very imbalanced bins)\n"
            "- Domain conventions for this attribute\n\n"
            "CRITICAL FORMATTING REQUIREMENT:\n"
            "Your response MUST end with EXACTLY these two lines and nothing after them:\n\n"
            "BIN_EDGES: [v1, v2, v3, ...]\n"
            "LABELS: [label1, label2, ...]\n\n"
            "Rules:\n"
            "- BIN_EDGES: sorted numeric boundaries, first ≤ min, last ≥ max\n"
            "- LABELS: one fewer than BIN_EDGES, short human-readable names\n"
            "- LABELS must be simple words or hyphenated words ONLY — NO commas, NO parentheses, NO numbers inside label names\n"
            "- Do NOT put any text AFTER the LABELS line\n"
            "- The BIN_EDGES and LABELS lines are MANDATORY — omitting them causes a pipeline failure\n\n"
            "Example for an age column (range 18-75):\n"
            "Brief reasoning here (2-3 sentences max).\n"
            "BIN_EDGES: [18, 30, 45, 60, 75]\n"
            "LABELS: [Young, Middle-Aged, Senior, Elderly]\n\n"
            "Now discretize the column '{name}' and end your response with the BIN_EDGES and LABELS lines."
        ).format(
            name=col_stats['column_name'],
            min=col_stats['min'],
            max=col_stats['max'],
            mean=col_stats['mean'],
            median=col_stats['median'],
            std=col_stats['std'],
            unique=col_stats['unique_count'],
            total=col_stats['total_count'],
            samples=col_stats['sample_values'],
        )

    @staticmethod
    def _build_summary_prompt(
        method: str,
        n_bins: int,
        continuous_cols: List[Dict],
        discretized_info: List[Dict],
    ) -> str:
        # Build per-column detail
        col_details = []
        for col_stat, disc in zip(continuous_cols, discretized_info):
            detail = (
                f"- Column: {col_stat['column_name']}\n"
                f"  Range: [{col_stat['min']}, {col_stat['max']}], "
                f"Mean: {col_stat['mean']}, Std: {col_stat['std']}\n"
            )
            if disc.get("status") == "success" or disc.get("labels"):
                detail += f"  Bins: {disc.get('labels', [])}\n"
                detail += f"  Distribution: {disc.get('distribution', {})}\n"
            else:
                detail += f"  Error: {disc.get('message', 'Unknown')}\n"
            col_details.append(detail)

        return (
            "The following continuous sensitive columns were discretized for "
            "fairness analysis using a user-specified method.\n\n"
            f"## Configuration\n"
            f"- Method: {method}\n"
            f"- Number of Bins: {n_bins}\n\n"
            f"## Column Details\n\n"
            + "\n".join(col_details)
            + "\n\n"
            "## Your Task\n\n"
            "Provide a brief analysis covering:\n"
            "1. Whether the discretization method was appropriate for each column, "
            "given its name and value range\n"
            "2. The resulting distribution across bins — are there any very "
            "imbalanced bins that could affect fairness analysis?\n"
            "3. Any recommendations for alternative binning strategies\n\n"
            "FORMATTING RULES:\n"
            "- Use ## for main headers, ### for subsections\n"
            "- Use numbered lists (1. 2. 3.) for ordered items\n"
            "- Do NOT use ** bold markers ** around headers\n"
            "- Do NOT use emojis, icons, or special symbols\n"
        )

    # ------------------------------------------------------------------
    # Response parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_bin_edges(response: str, col_stats: Dict) -> list:
        """Extract BIN_EDGES from the agent response."""
        # Try exact format first
        match = re.search(r"BIN_EDGES\s*:\s*\[([^\]]+)\]", response)
        if not match:
            # Try alternative formats: "bin edges: [...]" or "edges: [...]"
            match = re.search(r"(?:bin[_ ]?edges|edges)\s*[:=]\s*\[([^\]]+)\]", response, re.IGNORECASE)
        if not match:
            return []
        try:
            edges = [float(x.strip()) for x in match.group(1).split(",")]
            if len(edges) >= 2:
                return sorted(edges)
        except (ValueError, TypeError):
            pass
        return []

    @staticmethod
    def _parse_labels(response: str, expected_count: int) -> list:
        """Extract LABELS from the agent response.

        Handles the common failure mode where the model includes commas or
        parenthetical annotations inside label names, causing a naive comma-split
        to produce more tokens than expected.  Recovery strategy: strip
        parenthetical suffixes from each raw token and re-join tokens that look
        like they belong together (e.g. "Middle-Aged (28" and "40)" merge back
        to "Middle-Aged").
        """
        match = re.search(r"LABELS\s*:\s*\[([^\]]+)\]", response)
        if not match:
            match = re.search(r"(?:labels|bin[_ ]?labels)\s*[:=]\s*\[([^\]]+)\]", response, re.IGNORECASE)
        if not match:
            return None

        raw = match.group(1)
        try:
            tokens = [x.strip().strip("'\"") for x in raw.split(",")]

            # Happy path: count matches immediately
            if len(tokens) == expected_count:
                return [t for t in tokens if t]

            # Recovery: strip parenthetical annotations like "(19-28)" or "(prime age)"
            # and then re-split.  Pattern: remove anything in ( ... ) from each token.
            cleaned = [re.sub(r"\s*\([^)]*\)", "", t).strip() for t in tokens]
            cleaned = [t for t in cleaned if t]
            if len(cleaned) == expected_count:
                return cleaned

            # Last resort: if we have too many tokens because commas appeared inside
            # annotations, reconstruct by merging tokens until we hit expected_count.
            # Strategy: greedily take tokens that look like label words (non-numeric
            # leading character) and merge short fragments into the previous label.
            labels: list = []
            for tok in cleaned:
                if labels and (not tok or tok[0].isdigit() or tok.startswith("-")):
                    # numeric fragment — belongs to the previous token's annotation
                    continue
                labels.append(tok)
            if len(labels) == expected_count:
                return labels

        except (ValueError, TypeError):
            pass
        return None

    @staticmethod
    def _parse_bin_count(response: str, default: int = 5) -> int:
        """Try to extract a number of bins from the agent response as fallback."""
        match = re.search(r"(\d+)\s*bins?", response, re.IGNORECASE)
        if match:
            n = int(match.group(1))
            if 2 <= n <= 50:
                return n
        return default

    @staticmethod
    def _extract_reasoning_summary(
        agent_response: str,
        col_name: str,
        bin_edges: list,
        labels: list,
        result: Dict[str, Any],
    ) -> str:
        """Extract a clean, concise reasoning summary from the agent response.

        Instead of dumping the entire agent output (which can be very long),
        extract just the REASONING section text, or produce a short fallback
        summary describing what was done.
        """
        # Try to extract explicit REASONING section
        reasoning_match = re.search(
            r"REASONING\s*:\s*\n?(.*?)(?=\nBIN_EDGES|\nLABELS|$)",
            agent_response,
            re.DOTALL | re.IGNORECASE,
        )
        if reasoning_match:
            reasoning_text = reasoning_match.group(1).strip()
            # Limit to a reasonable length (first ~500 chars)
            if len(reasoning_text) > 500:
                # Cut at sentence boundary
                cut = reasoning_text[:500].rfind(".")
                if cut > 100:
                    reasoning_text = reasoning_text[: cut + 1]
                else:
                    reasoning_text = reasoning_text[:500] + "..."
            return reasoning_text

        # No explicit REASONING block — build a summary from the result
        actual_labels = result.get("labels", labels or [])
        actual_edges = result.get("bin_edges", bin_edges or [])
        actual_method = result.get("method", "auto")

        if actual_method == "auto" and actual_labels:
            return (
                f"The agent analysed the distribution and semantics of `{col_name}` "
                f"and chose {len(actual_labels)} bins: {', '.join(str(l) for l in actual_labels)}. "
                f"Bin edges: {actual_edges}."
            )
        elif actual_edges:
            return (
                f"Discretized `{col_name}` using {actual_method} method into "
                f"{len(actual_edges) - 1} bins (edges: {actual_edges})."
            )
        else:
            return f"Discretized `{col_name}` using {actual_method} method."
