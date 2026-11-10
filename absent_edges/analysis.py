"""Paired issue-family analysis from strict, evaluator-safe episode summaries.

This module never opens trajectories, patches, tests or benchmark records. It
does not infer missing episodes to be successful or silently drop invalid blocks.
Bootstrap intervals are not p-values. Statistical methods must be frozen before
confirmation and external p-values need an explicit method/provenance record.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from statistics import fmean


@dataclass(frozen=True)
class Contrast:
    id: str
    left: str
    right: str
    panel: str
    role: str


CONTRASTS = (
    Contrast("answer_semantics", "C3_NONEXHAUSTIVE", "C2_GENERIC", "P_mechanism", "primary"),
    Contrast("deterministic_verification", "C4_VERIFY", "C1_STANDARD", "P_natural", "secondary"),
    Contrast("verification_training", "T3_VERIFY_TRAINED", "T2_NOISE_TRAINED", "P_training", "secondary"),
)
SECONDARY_FAMILY = tuple(c.id for c in CONTRASTS if c.role == "secondary")
PILOT_CONDITIONS = ("C0_SOURCE", "C1_STANDARD", "C2_GENERIC", "C3_NONEXHAUSTIVE", "C4_VERIFY")
AGENT_FAILURES = frozenset({"agent_timeout", "tool_budget_exhausted", "model_budget_exhausted",
                            "malformed_patch", "empty_patch", "missing_patch", "agent_error",
                            "unsupported_candidate", "candidate_test_timeout", "candidate_test_failure"})
OUTCOMES = AGENT_FAILURES | {"resolved", "unresolved", "infrastructure_invalid"}
SAFE_METRICS = frozenset({"duration_s", "input_tokens", "output_tokens", "source_reads", "graph_calls",
                          "verification_calls", "tool_calls", "peak_memory_bytes"})
CELL_FIELDS = frozenset({"panel", "condition", "task_id", "variant_id", "inference_seed", "training_seed",
                         "repository_family", "issue_family", "config_hash", "model_hash"})
SUMMARY_FIELDS = CELL_FIELDS | {"episode_id", "study_type", "split", "protocol_hash", "outcome", "resolved", "infra_reason", "metrics"}


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _hash(value, field_name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field_name} must be a pinned SHA256")


@dataclass(frozen=True)
class CellSpec:
    panel: str
    condition: str
    task_id: str
    variant_id: str
    inference_seed: int
    training_seed: int | None
    repository_family: str
    issue_family: str
    config_hash: str
    model_hash: str

    def validate(self):
        for name in ("panel", "condition", "task_id", "variant_id", "repository_family", "issue_family"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"missing safe identity: {name}")
        if type(self.inference_seed) is not int or (self.training_seed is not None and type(self.training_seed) is not int):
            raise ValueError("seeds must be exact integers or a null training seed")
        _hash(self.config_hash, "config_hash")
        _hash(self.model_hash, "model_hash")
        return self

    @property
    def cell_key(self):
        return (self.panel, self.condition, self.task_id, self.variant_id, self.inference_seed, self.training_seed)

    @property
    def pair_key(self):
        return (self.panel, self.task_id, self.variant_id, self.inference_seed, self.training_seed)


@dataclass(frozen=True)
class AnalysisPlan:
    protocol_hash: str
    study_type: str
    split: str
    expected_cells: tuple[CellSpec, ...]
    bootstrap_seed: int = 4711
    bootstrap_replicates: int = 2000
    confidence: float = .95

    def validate(self):
        _hash(self.protocol_hash, "protocol_hash")
        if self.study_type not in ("pilot", "confirmation", "replication", "synthetic_fixture"):
            raise ValueError("engineering_smoke and unregistered study types cannot enter analysis")
        if self.split not in ("D", "P", "Q", "synthetic"):
            raise ValueError("invalid analysis split")
        required_split = {"pilot": "D", "confirmation": "P", "replication": "Q", "synthetic_fixture": "synthetic"}[self.study_type]
        if self.split != required_split:
            raise ValueError(f"{self.study_type} analysis must use the registered {required_split} split")
        if type(self.bootstrap_seed) is not int or type(self.bootstrap_replicates) is not int or self.bootstrap_replicates < 100:
            raise ValueError("deterministic bootstrap needs an integer seed and at least 100 replicates")
        if type(self.confidence) not in (int, float) or not 0 < self.confidence < 1:
            raise ValueError("confidence must be between zero and one")
        if not self.expected_cells:
            raise ValueError("a registered expected-cell manifest is required")
        keys, tasks, issues, arm_models = set(), {}, {}, {}
        for cell in self.expected_cells:
            cell.validate()
            if cell.cell_key in keys:
                raise ValueError("duplicate logical cell in expected manifest")
            keys.add(cell.cell_key)
            identity = (cell.repository_family, cell.issue_family)
            if cell.task_id in tasks and tasks[cell.task_id] != identity:
                raise ValueError("task variants changed issue/repository family")
            tasks[cell.task_id] = identity
            if cell.issue_family in issues and issues[cell.issue_family] != cell.repository_family:
                raise ValueError("one issue family spans inconsistent repository families")
            issues[cell.issue_family] = cell.repository_family
            arm = (cell.panel, cell.condition, cell.training_seed)
            # Configuration may differ by task (e.g. a task manifest). The model
            # checkpoint within one registered arm/training seed may not drift.
            if arm in arm_models and arm_models[arm] != cell.model_hash:
                raise ValueError("model checkpoint drift within an arm/training seed")
            arm_models[arm] = cell.model_hash
        return self

    @property
    def plan_hash(self):
        data = asdict(self)
        data["expected_cells"] = sorted(data["expected_cells"], key=canonical_hash)
        return canonical_hash(data)


def validate_summary(record: dict) -> dict:
    if not isinstance(record, dict) or set(record) - SUMMARY_FIELDS:
        raise ValueError("only allowlisted per-episode summary fields are accepted")
    required = SUMMARY_FIELDS - {"infra_reason", "metrics"}
    if required - set(record):
        raise ValueError("safe summary is missing required identity/outcome fields")
    if record["study_type"] == "engineering_smoke":
        raise ValueError("engineering_smoke is not a research observation and cannot be pooled")
    if not isinstance(record["episode_id"], str) or not record["episode_id"]:
        raise ValueError("episode_id required")
    CellSpec(**{k: record[k] for k in CELL_FIELDS}).validate()
    _hash(record["protocol_hash"], "protocol_hash")
    outcome = record["outcome"]
    if not isinstance(outcome, str) or outcome not in OUTCOMES or record["resolved"] is not None and type(record["resolved"]) is not bool:
        raise ValueError("invalid outcome contract")
    result = dict(record)
    if outcome == "infrastructure_invalid":
        if record["resolved"] is not None or not isinstance(record.get("infra_reason"), str) or not record["infra_reason"]:
            raise ValueError("invalid infrastructure requires an explicit reason and unknown resolved status")
    elif outcome in AGENT_FAILURES:
        if record["resolved"] is True:
            raise ValueError("agent-failure outcome contradicts success")
        # Explicit agent failures remain measured zeros, including missing patch
        # artifacts. A completely absent episode record is NOT inferred to fail.
        result["resolved"] = False
    elif record["resolved"] is not (outcome == "resolved"):
        raise ValueError("resolved/unresolved label and binary outcome disagree")
    if outcome != "infrastructure_invalid" and record.get("infra_reason") is not None:
        raise ValueError("infrastructure reasons cannot silently relabel agent failures")
    metrics = record.get("metrics", {})
    if not isinstance(metrics, dict) or set(metrics) - SAFE_METRICS:
        raise ValueError("only declared safe aggregate metrics are accepted")
    for value in metrics.values():
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("metrics must be finite nonnegative numbers or explicit unknowns")
    return result


def load_safe_summaries(path: str | Path, max_bytes=32*1024**2) -> list[dict]:
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError("summary input exceeds byte budget")
    text = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(text)
    if not isinstance(rows, list):
        raise ValueError("expected a JSON list or JSONL of safe episode summaries")
    return [validate_summary(row) for row in rows]


def validate_records(plan: AnalysisPlan, summaries: list[dict]):
    plan.validate()
    expected = {cell.cell_key: cell for cell in plan.expected_cells}
    seen, episode_ids = {}, set()
    for record in summaries:
        row = validate_summary(record)
        if (row["protocol_hash"], row["study_type"], row["split"]) != (plan.protocol_hash, plan.study_type, plan.split):
            raise ValueError("mixed protocol, study type or split cannot be pooled")
        cell = CellSpec(**{k: row[k] for k in CELL_FIELDS})
        if cell.cell_key not in expected:
            raise ValueError("episode cell was not registered in the analysis manifest")
        if cell != expected[cell.cell_key]:
            raise ValueError("registered model/config/task-family identity mismatch")
        if cell.cell_key in seen or row["episode_id"] in episode_ids:
            raise ValueError("duplicate episode or logical config/model/task/seed cell")
        seen[cell.cell_key] = row
        episode_ids.add(row["episode_id"])
    return seen


def paired_units(plan: AnalysisPlan, summaries: list[dict], contrast_id: str):
    contrast = next((c for c in CONTRASTS if c.id == contrast_id), None)
    if contrast is None:
        raise ValueError("contrast is not in the prespecified family")
    rows = validate_records(plan, summaries)
    expected = [c for c in plan.expected_cells if c.panel == contrast.panel and c.condition in (contrast.left, contrast.right)]
    if not expected:
        raise ValueError("registered manifest has no cells for this contrast panel")
    blocks = defaultdict(dict)
    for cell in expected:
        blocks[cell.pair_key][cell.condition] = cell
    if any(set(block) != {contrast.left, contrast.right} for block in blocks.values()):
        raise ValueError("expected design itself is unpaired; no complete-case subset is inferred")
    missing, invalid, family_blocks = [], [], defaultdict(list)
    outcomes = Counter()
    for pair_key in sorted(blocks, key=repr):
        block = blocks[pair_key]
        left_cell, right_cell = block[contrast.left], block[contrast.right]
        if (left_cell.repository_family, left_cell.issue_family) != (right_cell.repository_family, right_cell.issue_family):
            raise ValueError("paired conditions differ in issue family")
        if contrast.panel != "P_training" and left_cell.model_hash != right_cell.model_hash:
            raise ValueError("untrained paired conditions require the same model checkpoint")
        for cell in block.values():
            if cell.cell_key not in rows:
                missing.append(asdict(cell))
            else:
                outcomes[rows[cell.cell_key]["outcome"]] += 1
        if any(cell.cell_key not in rows for cell in block.values()):
            continue
        pair_rows = [rows[left_cell.cell_key], rows[right_cell.cell_key]]
        infra = [r for r in pair_rows if r["outcome"] == "infrastructure_invalid"]
        if infra:
            invalid.append({"pair_key": list(pair_key), "affected_episode_ids": [r["episode_id"] for r in pair_rows],
                            "reasons": [r["infra_reason"] for r in infra]})
            continue
        family_blocks[left_cell.issue_family].append({"pair_key": pair_key, "repository_family": left_cell.repository_family,
                                                     "left": int(pair_rows[0]["resolved"]), "right": int(pair_rows[1]["resolved"]),
                                                     "training_seed": left_cell.training_seed})
    audit = {"expected_episode_cells": len(expected), "expected_paired_cells": len(blocks),
             "expected_issue_families": len({c.issue_family for c in expected}),
             "expected_repository_families": len({c.repository_family for c in expected}),
             "observed_outcomes": dict(outcomes), "missing_episode_cells": missing, "infrastructure_invalid_blocks": invalid,
             "automatic_exclusions": 0, "complete_case_analysis_performed": False}
    if missing or invalid:
        return contrast, [], audit
    units = []
    for issue in sorted(family_blocks):
        pairs = family_blocks[issue]
        units.append({"issue_family": issue, "repository_family": pairs[0]["repository_family"],
                      "paired_cells": len(pairs), "left": fmean(p["left"] for p in pairs), "right": fmean(p["right"] for p in pairs),
                      "difference": fmean(p["left"]-p["right"] for p in pairs), "repetitions": pairs})
    return contrast, units, audit


def _validate_units(units):
    if not units:
        raise ValueError("no independent issue-family units")
    seen = set()
    for unit in units:
        if not isinstance(unit.get("issue_family"), str) or not unit["issue_family"] or not isinstance(unit.get("repository_family"), str) or not unit["repository_family"]:
            raise ValueError("issue/repository family identity is required")
        if unit["issue_family"] in seen:
            raise ValueError("duplicate independent issue family")
        seen.add(unit["issue_family"])
        if type(unit.get("paired_cells")) is not int or unit["paired_cells"] < 1:
            raise ValueError("issue family requires a positive paired-cell count")
        for name in ("left", "right", "difference"):
            if type(unit.get(name)) not in (int, float) or not math.isfinite(unit[name]):
                raise ValueError("unit means and differences must be finite")
        if not (0 <= unit["left"] <= 1 and 0 <= unit["right"] <= 1) or not math.isclose(unit["difference"], unit["left"]-unit["right"], abs_tol=1e-12):
            raise ValueError("unit outcomes and paired difference disagree")


def estimates(units):
    _validate_units(units)
    repos = defaultdict(list)
    for unit in units:
        repos[unit["repository_family"]].append(unit)
    micro = {key: fmean(u[key] for u in units) for key in ("left", "right", "difference")}
    macro = {key: fmean(fmean(u[key] for u in group) for group in repos.values()) for key in micro}
    return {"micro_issue": micro, "macro_repository": macro,
            "issue_families": len(units), "repository_families": len(repos),
            "weighting": {"micro_issue": "equal weight per independent issue family",
                          "macro_repository": "equal weight per repository family, then equal issue weight within repository",
                          "within_issue": "equal weight per registered paired task/variant/inference/training-seed cell"}}


def _quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered)-1)*probability
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(position-lo)


def bootstrap_intervals(units, *, seed=4711, replicates=2000, confidence=.95, hierarchical=False):
    """Resample complete issue-family vectors, never individual seeds/variants.

    Primary CI conditions on the observed repositories and resamples issues
    within each repository. Sensitivity first resamples repositories, then their
    issue clusters. All repeated checkpoints/seeds remain together in each unit.
    """
    _validate_units(units)
    if type(seed) is not int or type(replicates) is not int or replicates < 100 or type(confidence) not in (int, float) or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap inputs")
    repos = defaultdict(list)
    for unit in sorted(units, key=lambda u: (u["repository_family"], u["issue_family"])):
        repos[unit["repository_family"]].append(unit)
    names, rng = sorted(repos), random.Random(seed)
    micro_draws, macro_draws = [], []
    for _ in range(replicates):
        selected_repos = [rng.choice(names) for _ in names] if hierarchical else names
        groups = [[rng.choice(repos[name]) for _ in repos[name]] for name in selected_repos]
        # Duplicate selected repositories are separate bootstrap cluster copies.
        # Do not merge them by original repo name and lose sampled multiplicity.
        micro_draws.append(fmean(u["difference"] for group in groups for u in group))
        macro_draws.append(fmean(fmean(u["difference"] for u in group) for group in groups))
    alpha = (1-confidence)/2
    result = {"method": "hierarchical_repository_then_issue_cluster_percentile" if hierarchical else "repository_stratified_issue_cluster_percentile",
              "seed": seed, "replicates": replicates, "confidence": confidence,
              "micro_issue": [_quantile(micro_draws, alpha), _quantile(micro_draws, 1-alpha)],
              "macro_repository": [_quantile(macro_draws, alpha), _quantile(macro_draws, 1-alpha)],
              "resampling_unit": "whole_issue_family_including_all_variants_and_seeds",
              "conditional_on": "registered model checkpoints and training seeds; not a training-method population interval"}
    if len(names) < 2:
        result["warning"] = "one repository cannot identify between-repository uncertainty"
    if len(units) < 2:
        result["warning"] = "one issue-family interval is degenerate and cannot support generalization"
    return result


def exact_mcnemar(units):
    """Exact two-sided binomial McNemar, ONLY one binary pair per issue family."""
    _validate_units(units)
    if any(u["paired_cells"] != 1 or u["left"] not in (0, 1) or u["right"] not in (0, 1) for u in units):
        raise ValueError("exact McNemar requires one prespecified binary result per independent issue; averaged seeds are ineligible")
    b = sum(u["left"] == 1 and u["right"] == 0 for u in units)
    c = sum(u["left"] == 0 and u["right"] == 1 for u in units)
    n = b+c
    p = min(1.0, 2*sum(math.comb(n, k) for k in range(min(b, c)+1))/2**n) if n else 1.0
    return {"p_value": p, "method": "exact_two_sided_McNemar_binomial",
            "discordant_left_only": b, "discordant_right_only": c,
            "assumptions": "independent issue families; one prespecified binary pair per issue; conditional binomial null",
            "input_sha256": canonical_hash(units)}


def holm_adjust(p_values: dict[str, float], *, family=SECONDARY_FAMILY):
    if set(p_values) != set(family) or len(family) != len(set(family)):
        raise ValueError("the complete prespecified hypothesis family is required; do not drop unavailable tests")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in p_values.values()):
        raise ValueError("p-values must be genuine finite values in [0,1]")
    ranked = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted, previous = {}, 0.0
    for i, name in enumerate(ranked):
        previous = max(previous, min(1.0, (len(ranked)-i)*p_values[name]))
        adjusted[name] = previous
    return {name: adjusted[name] for name in family}


def adjust_secondary_evidence(evidence: dict[str, dict], alpha=.05):
    if type(alpha) not in (int, float) or not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if set(evidence) != set(SECONDARY_FAMILY):
        raise ValueError("both prespecified secondary tests need evidence")
    for record in evidence.values():
        required = {"p_value", "method", "input_sha256", "assumptions"}
        counts = {"discordant_left_only", "discordant_right_only"}
        if not isinstance(record, dict) or required - set(record) or set(record) - required - counts:
            raise ValueError("each p-value needs method, input provenance and assumptions")
        if any(not isinstance(record[name], str) or not record[name] for name in ("method", "assumptions")):
            raise ValueError("each p-value needs method, input provenance and assumptions")
        if any(type(record[name]) is not int or record[name] < 0 for name in counts & set(record)):
            raise ValueError("discordant counts must be nonnegative integers")
        _hash(record["input_sha256"], "p-value input_sha256")
    adjusted = holm_adjust({name: record["p_value"] for name, record in evidence.items()})
    return {"method": "Holm_step_down", "family": list(SECONDARY_FAMILY), "alpha": alpha,
            "tests": {name: {**evidence[name], "adjusted_p_value": adjusted[name], "reject_at_alpha": adjusted[name] <= alpha}
                      for name in SECONDARY_FAMILY},
            "primary_included": False}


def analyze_contrast(plan: AnalysisPlan, summaries: list[dict], contrast_id: str):
    contrast, units, audit = paired_units(plan, summaries, contrast_id)
    base = {"contrast": asdict(contrast), "plan_sha256": plan.plan_hash, "protocol_hash": plan.protocol_hash,
            "study_type": plan.study_type, "split": plan.split, "audit": audit,
            "p_value": None, "p_value_status": "not_computed_from_bootstrap",
            "secondary_family": list(SECONDARY_FAMILY) if contrast.role == "secondary" else None}
    if audit["missing_episode_cells"] or audit["infrastructure_invalid_blocks"]:
        return {**base, "status": "blocked_incomplete_or_infrastructure_invalid", "estimate": None, "interval": None,
                "required_action": "preserve records; resolve the full affected paired block under the frozen infrastructure policy; never take an automatic complete-case subset"}
    return {**base, "status": "computed_on_registered_complete_pairs", "estimate": estimates(units),
            "interval": bootstrap_intervals(units, seed=plan.bootstrap_seed, replicates=plan.bootstrap_replicates, confidence=plan.confidence),
            "repository_sensitivity": bootstrap_intervals(units, seed=plan.bootstrap_seed+1, replicates=plan.bootstrap_replicates,
                                                            confidence=plan.confidence, hierarchical=True),
            "units": units, "p_value_note": "A separate predeclared valid paired test is needed. Exact McNemar is available only for one binary pair per independent issue.",
            "interpretation_limit": "A confidence interval containing zero is not proof of no effect, equivalence or noninferiority."}


def pilot_plan_from_manifest(manifest: dict) -> AnalysisPlan:
    """Read the safe enqueuer contract; ignore no outcomes and infer no cells."""
    if manifest.get("panel") != "D_full_agent":
        raise ValueError("expected the descriptive D_full_agent manifest")
    plan = AnalysisPlan(manifest["protocol_hash"], manifest["study_type"], manifest["split"],
                        tuple(CellSpec(**cell) for cell in manifest["expected_cells"]))
    _validate_pilot_plan(plan)
    return plan


def _validate_pilot_plan(plan: AnalysisPlan):
    plan.validate()
    if plan.study_type != "pilot" or plan.split != "D":
        raise ValueError("descriptive pilot report requires pilot/D")
    blocks = defaultdict(set)
    for cell in plan.expected_cells:
        if cell.panel != "D_full_agent" or cell.condition not in PILOT_CONDITIONS or cell.training_seed is not None:
            raise ValueError("pilot requires D_full_agent base-model C0-C4 cells")
        blocks[cell.pair_key].add(cell.condition)
    if any(conditions != set(PILOT_CONDITIONS) for conditions in blocks.values()):
        raise ValueError("pilot expected manifest must contain every C0-C4 arm in each registered block")
    # The pilot compares interfaces with the same checkpoint, never a model swap.
    if len({cell.model_hash for cell in plan.expected_cells}) != 1:
        raise ValueError("pilot base model checkpoint must be identical across all cells")


def describe_pilot(plan: AnalysisPlan, summaries: list[dict]) -> dict:
    """Describe every registered natural pilot cell without CIs or p-values.

    A missing observation remains unknown. A condition's rate is unavailable
    until its whole expected denominator is observed and infrastructure-valid.
    This natural full-agent panel is never relabeled as P_mechanism.
    """
    _validate_pilot_plan(plan)
    rows = validate_records(plan, summaries)
    conditions = {}
    for condition in PILOT_CONDITIONS:
        expected = [cell for cell in plan.expected_cells if cell.condition == condition]
        observed = [rows[cell.cell_key] for cell in expected if cell.cell_key in rows]
        missing = [asdict(cell) for cell in expected if cell.cell_key not in rows]
        invalid = [row for row in observed if row["outcome"] == "infrastructure_invalid"]
        measured = [row for row in observed if row["outcome"] != "infrastructure_invalid"]
        by_issue = defaultdict(list)
        for cell in expected:
            by_issue[cell.issue_family].append(cell)
        issue_descriptions = []
        for issue, cells in sorted(by_issue.items()):
            issue_rows = [rows[cell.cell_key] for cell in cells if cell.cell_key in rows]
            valid = [row for row in issue_rows if row["outcome"] != "infrastructure_invalid"]
            complete = len(valid) == len(cells)
            issue_descriptions.append({"issue_family": issue, "repository_family": cells[0].repository_family,
                "expected_cells": len(cells), "observed_cells": len(issue_rows), "measured_outcome_cells": len(valid),
                "resolved_count": sum(row["resolved"] is True for row in valid),
                "resolved_rate": fmean(int(row["resolved"]) for row in valid) if complete else None})
        rates = None
        if not missing and not invalid:
            repos = defaultdict(list)
            for issue in issue_descriptions:
                repos[issue["repository_family"]].append(issue["resolved_rate"])
            rates = {"micro_issue_resolved_rate": fmean(issue["resolved_rate"] for issue in issue_descriptions),
                     "macro_repository_resolved_rate": fmean(fmean(values) for values in repos.values())}
        conditions[condition] = {"expected_cells": len(expected), "observed_cells": len(observed),
            "measured_outcome_cells": len(measured), "resolved_count": sum(row["resolved"] is True for row in measured),
            "outcome_counts": dict(Counter(row["outcome"] for row in observed)),
            "missing_cells": missing, "infrastructure_invalid_cells": [
                {"episode_id": row["episode_id"], "reason": row["infra_reason"]} for row in invalid],
            "rates": rates, "issue_families": issue_descriptions}
    complete = all(item["rates"] is not None for item in conditions.values())
    return {"schema_version": 1, "scope": "descriptive_natural_full_agent_development_only", "panel": "D_full_agent",
        "study_type": "pilot", "split": "D", "protocol_hash": plan.protocol_hash, "plan_sha256": plan.plan_hash,
        "status": "complete_registered_block" if complete else "incomplete_or_infrastructure_invalid",
        "expected_episode_cells": len(plan.expected_cells), "observed_episode_cells": len(rows),
        "independent_issue_families": len({cell.issue_family for cell in plan.expected_cells}),
        "repository_families": len({cell.repository_family for cell in plan.expected_cells}),
        "conditions": conditions, "automatic_exclusions": 0, "inferential_statistics_generated": False,
        "interpretation_limit": "Descriptive D observations only. No fixed-evidence causal, confirmation, generalized improvement, equivalence or no-effect claim.",
        "denominator_policy": "Every expected cell is retained. Explicit agent failures are zeros; missing records and infrastructure-invalid observations remain unknown. Rates require complete valid condition denominators.",
        "weighting": "Average all registered repetitions within issue; micro weights issues equally; macro weights repositories equally then issues within repository."}


PILOT_EVIDENCE_REQUIRED = frozenset({"episode_job_id", "attempt_number", "episode_artifact_sha256", "outcome_source"})
PILOT_EVIDENCE_OPTIONAL = frozenset({"grader_run_id", "grader_artifact_sha256", "grader_fingerprint", "harness_revision",
    "wrapper_sha256", "candidate_policy_sha256", "serving_binding_sha256", "serving_stable", "image_audit_sha256",
    "task_sha256", "launch_profile_sha256", "source_tree_sha256"})


def validate_pilot_evidence(summaries: list[dict], provenance_by_episode: dict[str, dict]) -> dict:
    """Validate safe evidence references; caller must verify artifact content/hashes.

    Private tests, patches and grader output are forbidden. These provenance
    declarations are not proof that a grader ran or that an artifact is genuine.
    """
    rows = [validate_summary(row) for row in summaries]
    if len({row["episode_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate episode in pilot evidence")
    if set(provenance_by_episode) != {row["episode_id"] for row in rows}:
        raise ValueError("one evidence envelope is required for each observed episode, without extras")
    result, attempts, grader_runs = {}, set(), set()
    for row in rows:
        evidence = provenance_by_episode[row["episode_id"]]
        if not isinstance(evidence, dict) or PILOT_EVIDENCE_REQUIRED-set(evidence) or set(evidence)-PILOT_EVIDENCE_REQUIRED-PILOT_EVIDENCE_OPTIONAL:
            raise ValueError("only allowlisted pilot evidence identity fields are accepted")
        if not isinstance(evidence["episode_job_id"], str) or not evidence["episode_job_id"] or type(evidence["attempt_number"]) is not int or evidence["attempt_number"] < 1:
            raise ValueError("episode job and positive attempt number are required")
        attempt = (evidence["episode_job_id"], evidence["attempt_number"])
        if attempt in attempts:
            raise ValueError("one episode attempt cannot be reused across logical cells")
        attempts.add(attempt)
        for name, value in evidence.items():
            if name.endswith("_sha256") or name == "grader_fingerprint":
                _hash(value, name)
        if "serving_stable" in evidence and type(evidence["serving_stable"]) is not bool:
            raise ValueError("serving_stable must be an explicit boolean")
        if evidence.get("serving_stable") is False and row["outcome"] != "infrastructure_invalid":
            raise ValueError("serving drift must be explicitly infrastructure-invalid")
        source = evidence["outcome_source"]
        if source == "grader":
            required = {"grader_run_id", "grader_artifact_sha256", "grader_fingerprint", "harness_revision", "wrapper_sha256"}
            if required-set(evidence) or any(not isinstance(evidence[name], str) or not evidence[name] for name in required):
                raise ValueError("grader outcomes require run, artifact, fingerprint, harness and wrapper provenance")
            revision = evidence["harness_revision"]
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                raise ValueError("grader harness revision must be a pinned Git SHA")
            if evidence["grader_run_id"] in grader_runs:
                raise ValueError("one grader run cannot be reused across independent episode cells")
            grader_runs.add(evidence["grader_run_id"])
        elif source == "grader_policy":
            if row["outcome"] not in AGENT_FAILURES or {"grader_artifact_sha256", "candidate_policy_sha256"}-set(evidence):
                raise ValueError("policy rejection requires a failed agent outcome and pinned policy/artifact")
        elif source == "agent_controller":
            if row["outcome"] not in AGENT_FAILURES:
                raise ValueError("controller evidence alone cannot establish a resolved/unresolved grader result")
        elif source == "infrastructure":
            if row["outcome"] != "infrastructure_invalid":
                raise ValueError("infrastructure evidence requires an explicit invalid outcome")
        else:
            raise ValueError("unknown pilot outcome source")
        result[row["episode_id"]] = dict(evidence)
    return result


def export_pilot_report(path: str | Path, plan: AnalysisPlan, summaries: list[dict], provenance_by_episode: dict[str, dict]) -> dict:
    """Create a new descriptive JSON artifact; never overwrite prior evidence."""
    report = describe_pilot(plan, summaries)
    report["evidence_references"] = validate_pilot_evidence(summaries, provenance_by_episode)
    report["evidence_validation_scope"] = "Provided safe provenance schema only; artifact authenticity and grader evidence must be verified by the exporting controller."
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode()+b"\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(target.resolve()), "sha256": hashlib.sha256(encoded).hexdigest(),
            "status": report["status"], "scope": report["scope"]}
