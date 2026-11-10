"""Exp A power simulation for the primary contrasts H1-H3 (D9, D11.2, D11.7; prereg section 14).

    <figures env python> scripts/expA_power.py --params power_params.json --out OUT_DIR [--overwrite]
    <figures env python> scripts/expA_power.py --write-default-params power_params.json

Environment: needs numpy, which the controller env lacks. Use the figures env
(/mnt/kldata/absent-edges-20261018-core2/expA/dev/integrate/envs/fig/bin/python, numpy 2.4.6). Without
numpy the script exits with status 2 and names that env. CPU only.

Data-generating model (one simulated study = the frozen block P design)
-----------------------------------------------------------------------
For family f, item i, condition c in {C0, C1, C2, C3, C4a} and seed s, the latent value

    Z = sqrt(rho_F) * A_f + sqrt(rho_I) * B_fi + sqrt(1 - rho_F - rho_I) * E_fics,   A, B, E ~ N(0, 1)

is recovered (Y = 1) when Z < tau_{c,f}. rho_F is the intra-family and rho_I the extra intra-item
correlation on the latent scale (an item that is easy in one condition and seed tends to be easy in
the others). tau_{c,f} = Phi^-1(p_c) * sqrt(1 + h^2) + h * H_{f,c} for the treatment-side arms C3,
C4a and C0 (H ~ N(0, 1); h = effect_heterogeneity_sd), and Phi^-1(p_c) for C1 and C2, so every
arm's marginal rate is exactly p_c while the contrasts vary between families with SD of order h.
Effects are absolute rate differences: p_C3 = p_C2 + d_H1, p_C4a = p_C1 + d_H2, p_C0 = p_C1 - d_H3
(H3 = C1 - C0, expected negative).

Parameterisation (one rule; review finding on base_rates): ``base_rates`` give every arm's rate
(pilot-calibrated before the freeze). C1 and C2 are the anchors; the background effects are
derived from the given rates, H1 = C3 - C2, H2 = C4a - C1 and H3 = C1 - C0, and ``effects``, when
present, must equal them (a params file whose effects contradict its rates is refused). The
global null sets all three contrasts to 0 (C3 = C2, C4a = C1, C0 = C1 at the anchors); the power
of one contrast varies only that contrast while the others keep their background values. The arm
rates actually simulated are recorded in POWER.json and POWER.md.

Analysis per simulated study (the frozen D11.2 procedure): item mean over seeds, paired difference
within item, family mean, then a two-sided one-sample t test over the F family differences (df =
F - 1) with the zero-variance rules (all differences 0: p = 1; identical nonzero: the exact sign-flip
p = 2^(1-F)); Holm across H1-H3 at alpha. p-values come from an exact table of the Student t
two-sided tail (absent_edges.probe.analysis.student_t_two_sided_p) interpolated on a 0.0005 grid.

Outputs: POWER.json (parameters and their sha256, per scenario: the simulated arm rates, family-wise
error at the global null, each contrast's rejection rate at the base rates, power of each contrast at
the practically meaningful effect with the others at their background values, the minimal
detectable effect at the target power in the expected and the opposite direction, and every power
evaluation) and POWER.md. Common random numbers are used within a scenario, so power is evaluated on
the same simulated latents for every effect size and the bisection is stable.

Minimum data (D11.3): with fewer than min_families_confirmatory families every contrast is
exploratory and the analysis never rejects, so a configuration below that size reports
confirmatory power 0, no MDE and family-wise error 0, keeps the unrestricted Holm rates under
``*_ignoring_minimum_data`` keys, and is labelled exploratory in POWER.md.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from absent_edges.probe.analysis import student_t_two_sided_p  # noqa: E402

PARAMS_SCHEMA = "expA-power-params-v1"
OUTPUT_SCHEMA = "expA-power-v1"
FIGURES_ENV = "/mnt/kldata/absent-edges-20261018-core2/expA/dev/integrate/envs/fig/bin/python"
CONDITIONS = ("C0", "C1", "C2", "C3", "C4a")
CONTRASTS = (("H1", "C3", "C2"), ("H2", "C4a", "C1"), ("H3", "C1", "C0"))
HETEROGENEOUS = ("C0", "C3", "C4a")
EXPECTED_DIRECTION = {"H1": 1, "H2": 1, "H3": -1}
TOP_LEVEL = {"schema", "label", "families", "items_per_family", "seeds", "base_rates", "effects",
             "intra_family_correlation", "intra_item_correlation", "effect_heterogeneity_sd", "alpha", "power_target",
             "min_families_confirmatory", "practically_meaningful_effect", "simulations", "seed", "mde", "scenarios",
             "notes"}
SCENARIO_KEYS = {"name", "base_rate", "base_rates", "effects", "families", "items_per_family", "seeds",
                 "intra_family_correlation", "intra_item_correlation", "effect_heterogeneity_sd", "simulations"}
P_GRID_STEP, P_GRID_MAX = 0.0005, 60.0


def default_params():
    """Placeholder rates until the pilot: every arm at the scenario's base rate (prereg section 14)."""
    scenarios = []
    for corr_name, rho_f, rho_i in (("low", 0.02, 0.10), ("medium", 0.05, 0.30), ("high", 0.15, 0.50)):
        for het_name, het in (("none", 0.0), ("moderate", 0.3)):
            for base in (0.1, 0.5, 0.9):
                scenarios.append({"name": f"corr-{corr_name}_het-{het_name}_base-{base}", "base_rate": base,
                                  "intra_family_correlation": rho_f, "intra_item_correlation": rho_i,
                                  "effect_heterogeneity_sd": het})
    return {"schema": PARAMS_SCHEMA, "label": "PLACEHOLDER rates (replace base_rates with the pilot's before the "
                                              "freeze)",
            "families": 30, "items_per_family": 25, "seeds": 3,
            "base_rates": {"C0": 0.5, "C1": 0.5, "C2": 0.5, "C3": 0.5, "C4a": 0.5},
            "effects": {"H1": 0.0, "H2": 0.0, "H3": 0.0}, "intra_family_correlation": 0.05,
            "intra_item_correlation": 0.30, "effect_heterogeneity_sd": 0.0, "alpha": 0.05, "power_target": 0.8,
            "min_families_confirmatory": 20, "practically_meaningful_effect": 0.10, "simulations": 2000,
            "seed": 20261101, "mde": {"max_effect": 0.6, "tolerance": 0.005}, "scenarios": scenarios}


def _numpy():
    try:
        import numpy
    except ModuleNotFoundError:
        sys.stderr.write("expA_power.py needs numpy; run it with the figures env: " + FIGURES_ENV + "\n")
        raise SystemExit(2)
    return numpy


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def validate_params(params):
    _check(isinstance(params, dict) and params.get("schema") == PARAMS_SCHEMA, f"params schema must be {PARAMS_SCHEMA}")
    _check(not set(params) - TOP_LEVEL, f"unknown params fields {sorted(set(params) - TOP_LEVEL)}")
    for scenario in params.get("scenarios") or []:
        _check(isinstance(scenario, dict) and isinstance(scenario.get("name"), str) and scenario["name"]
               and not set(scenario) - SCENARIO_KEYS, f"bad scenario {scenario!r}")
    names = [s["name"] for s in params.get("scenarios") or []]
    _check(len(names) == len(set(names)), "scenario names must be unique")
    return params


def resolve(params, scenario=None):
    """One fully specified configuration (scenario overrides applied)."""
    config = {k: copy.deepcopy(v) for k, v in params.items() if k not in ("scenarios", "notes")}
    config["name"] = "base"
    for key, value in (scenario or {}).items():
        if key == "base_rate":
            config["base_rates"] = {c: value for c in CONDITIONS}
        else:
            config[key] = copy.deepcopy(value)
    f = config["families"]
    items = config["items_per_family"]
    items = [items] * f if isinstance(items, int) else list(items)
    _check(type(f) is int and f >= 1 and len(items) == f and all(type(m) is int and m >= 1 for m in items),
           "families must be >= 1 and items_per_family an int or a list with one count per family")
    config["items_per_family"] = items
    _check(type(config["seeds"]) is int and config["seeds"] >= 1, "seeds must be an integer >= 1")
    rates = config["base_rates"]
    _check(set(rates) == set(CONDITIONS) and all(0 < rates[c] < 1 for c in CONDITIONS),
           f"base_rates must give each of {CONDITIONS} a rate in (0, 1)")
    background = background_effects(rates)
    effects = config.get("effects") or {}
    _check(set(effects) <= {"H1", "H2", "H3"}, "effects are keyed by H1, H2, H3")
    conflicts = {h: (float(value), background[h]) for h, value in effects.items()
                 if abs(float(value) - background[h]) > 1e-9}
    _check(not conflicts, "effects are derived from base_rates (H1 = C3 - C2, H2 = C4a - C1, H3 = C1 - C0); "
                          f"these given effects contradict the rates (given, derived): {conflicts}. Give every arm's "
                          "rate in base_rates and omit effects or make them consistent")
    config["effects"] = background
    rho_f, rho_i = config["intra_family_correlation"], config["intra_item_correlation"]
    _check(0 <= rho_f and 0 <= rho_i and rho_f + rho_i < 1, "correlations must be >= 0 and sum to < 1")
    _check(config["effect_heterogeneity_sd"] >= 0, "effect_heterogeneity_sd must be >= 0")
    _check(0 < config["alpha"] < 1 and 0 < config["power_target"] < 1, "alpha and power_target must be in (0, 1)")
    _check(type(config["simulations"]) is int and config["simulations"] >= 100, "simulations must be >= 100")
    return config


def background_effects(rates):
    """The contrasts implied by every arm's rate: H1 = C3 - C2, H2 = C4a - C1, H3 = C1 - C0."""
    return {"H1": rates["C3"] - rates["C2"], "H2": rates["C4a"] - rates["C1"], "H3": rates["C1"] - rates["C0"]}


def arm_rates(config, effects):
    """Marginal rates per arm after applying the contrast effects (C1 and C2 are the anchors)."""
    rates = dict(config["base_rates"])
    rates["C3"] = rates["C2"] + effects["H1"]
    rates["C4a"] = rates["C1"] + effects["H2"]
    rates["C0"] = rates["C1"] - effects["H3"]
    return rates


_P_TABLES = {}


class PValueTable:
    """Two-sided Student t tail p(|t|) on a fine grid (exact values from analysis.student_t_two_sided_p)."""

    def __init__(self, np, df):
        self.np, self.df = np, df
        if df not in _P_TABLES:
            grid = np.arange(0.0, P_GRID_MAX + P_GRID_STEP / 2, P_GRID_STEP)
            _P_TABLES[df] = (grid, np.array([student_t_two_sided_p(float(t), df) for t in grid]))
        self.grid, self.p = _P_TABLES[df]

    def __call__(self, t):
        return self.np.interp(self.np.abs(t), self.grid, self.p, right=float(self.p[-1]))


class Simulator:
    """Common random numbers for one configuration: latents drawn once, thresholds varied per effect."""

    def __init__(self, np, config, seed):
        self.np, self.config = np, config
        rng = np.random.default_rng(seed)
        sims, f, s = config["simulations"], config["families"], config["seeds"]
        m = max(config["items_per_family"])
        self.mask = np.zeros((f, m), dtype=bool)
        for k, count in enumerate(config["items_per_family"]):
            self.mask[k, :count] = True
        self.counts = self.mask.sum(axis=1).astype(float)
        rho_f, rho_i = config["intra_family_correlation"], config["intra_item_correlation"]
        base = (math.sqrt(rho_f) * rng.standard_normal((sims, f, 1)).astype(np.float32)
                + math.sqrt(rho_i) * rng.standard_normal((sims, f, m)).astype(np.float32))
        noise = rng.standard_normal((sims, f, m, len(CONDITIONS), s), dtype=np.float32)
        self.z = base[..., None, None] + np.float32(math.sqrt(1.0 - rho_f - rho_i)) * noise
        self.h = rng.standard_normal((sims, f, len(CONDITIONS))).astype(np.float32)
        self.table = PValueTable(np, f - 1) if f >= 2 else None

    def thresholds(self, rates):
        np, het = self.np, self.config["effect_heterogeneity_sd"]
        nd = statistics.NormalDist()
        tau = np.empty(self.h.shape, dtype=np.float32)
        for c, condition in enumerate(CONDITIONS):
            p = min(max(rates[condition], 1e-9), 1 - 1e-9)
            if condition in HETEROGENEOUS and het > 0:
                tau[..., c] = nd.inv_cdf(p) * math.sqrt(1 + het * het) + het * self.h[..., c]
            else:
                tau[..., c] = nd.inv_cdf(p)
        return tau

    def family_differences(self, effects):
        """(simulations, families, 3) family mean paired differences for H1, H2, H3."""
        np = self.np
        tau = self.thresholds(arm_rates(self.config, effects))
        item_means = (self.z < tau[:, :, None, :, None]).mean(axis=4)          # (sims, F, m, C)
        index = {c: k for k, c in enumerate(CONDITIONS)}
        out = []
        for _, treatment, control in CONTRASTS:
            diff = item_means[..., index[treatment]] - item_means[..., index[control]]
            out.append((diff * self.mask).sum(axis=2) / self.counts)
        return np.stack(out, axis=2)

    def p_values(self, family_diffs):
        """D11.2 two-sided p per simulated study and contrast, with the zero-variance rules."""
        np = self.np
        f = family_diffs.shape[1]
        mean = family_diffs.mean(axis=1)
        if f < 2:
            return np.where(mean == 0, 1.0, 1.0)
        sd = family_diffs.std(axis=1, ddof=1)
        constant = np.isclose(sd, 0.0, atol=1e-12)
        safe = np.where(constant, 1.0, sd)
        t = mean / (safe / math.sqrt(f))
        p = self.table(t)
        p = np.where(constant & np.isclose(mean, 0.0, atol=1e-12), 1.0, p)
        p = np.where(constant & ~np.isclose(mean, 0.0, atol=1e-12), 2.0 ** (1 - f), p)
        return p

    def rejections(self, effects):
        """(simulations, 3) Holm rejections at alpha across H1-H3."""
        np = self.np
        p = self.p_values(self.family_differences(effects))
        order = np.argsort(p, axis=1, kind="stable")
        ordered = np.take_along_axis(p, order, axis=1)
        adjusted = np.maximum.accumulate(np.minimum(1.0, ordered * np.array([3.0, 2.0, 1.0])), axis=1)
        reject = np.empty_like(adjusted, dtype=bool)
        np.put_along_axis(reject, order, adjusted <= self.config["alpha"], axis=1)
        return reject


def _room(config, contrast, direction):
    """Largest |effect| that keeps every arm rate inside [0.001, 0.999] in this direction."""
    rates = config["base_rates"]
    anchor = {"H1": rates["C2"], "H2": rates["C1"], "H3": rates["C1"]}[contrast]
    moved = direction if contrast != "H3" else -direction
    return (0.999 - anchor) if moved > 0 else (anchor - 0.001)


def evaluate(sim, config, contrast, effect, cache):
    key = (contrast, round(effect, 10))
    if key not in cache:
        effects = dict(config["effects"])
        effects[contrast] = effect
        reject = sim.rejections(effects)
        k = [h for h, _, _ in CONTRASTS].index(contrast)
        cache[key] = float(reject[:, k].mean())
    return cache[key]


def minimal_detectable_effect(sim, config, contrast, direction, cache):
    """Smallest |effect| in this direction with power >= target (bisection under common random numbers)."""
    target, tolerance = config["power_target"], config["mde"]["tolerance"]
    high = min(config["mde"]["max_effect"], _room(config, contrast, direction))
    if high <= 0:
        return {"mde": None, "reason": "no room for an effect in this direction", "max_effect_tested": 0.0}
    power_high = evaluate(sim, config, contrast, direction * high, cache)
    if power_high < target:
        return {"mde": None, "reason": f"power {power_high:.3f} < target at the largest effect {high:.3f}",
                "max_effect_tested": high, "power_at_max": power_high}
    low = 0.0
    while high - low > tolerance:
        middle = (low + high) / 2
        if evaluate(sim, config, contrast, direction * middle, cache) >= target:
            high = middle
        else:
            low = middle
    return {"mde": high, "power_at_mde": evaluate(sim, config, contrast, direction * high, cache),
            "power_just_below": evaluate(sim, config, contrast, direction * low, cache) if low > 0 else None,
            "bracket": [low, high]}


def _rounded(rates):
    return {condition: round(float(rates[condition]), 12) for condition in CONDITIONS}


def run_configuration(np, config, seed):
    sim = Simulator(np, config, seed)
    cache = {}
    zero = {h: 0.0 for h, _, _ in CONTRASTS}
    null = sim.rejections(zero)
    at_base = sim.rejections(config["effects"])
    confirmatory = config["families"] >= config["min_families_confirmatory"]
    fwer = float(null.any(axis=1).mean())
    result = {"name": config["name"], "families": config["families"],
              "items_per_family": sorted(set(config["items_per_family"])), "seeds": config["seeds"],
              "base_rates": config["base_rates"], "background_effects": dict(config["effects"]),
              "simulated_arm_rates": {"base_rates": _rounded(arm_rates(config, config["effects"])),
                                      "global_null": _rounded(arm_rates(config, zero))},
              "intra_family_correlation": config["intra_family_correlation"],
              "intra_item_correlation": config["intra_item_correlation"],
              "effect_heterogeneity_sd": config["effect_heterogeneity_sd"], "simulations": config["simulations"],
              "seed": seed, "familywise_error_global_null": fwer if confirmatory else 0.0,
              "familywise_error_global_null_ignoring_minimum_data": fwer,
              "rejection_rate_global_null": {h: float(null[:, k].mean()) for k, (h, _, _) in enumerate(CONTRASTS)},
              "rejection_rate_at_base_rates": {h: float(at_base[:, k].mean()) if confirmatory else 0.0
                                               for k, (h, _, _) in enumerate(CONTRASTS)},
              "confirmatory_possible": confirmatory,
              "minimum_data_rule": (None if confirmatory else
                                    f"exploratory: {config['families']} families < min_families_confirmatory "
                                    f"{config['min_families_confirmatory']}; the analysis never rejects (D11.3), so "
                                    "confirmatory power is 0"),
              "contrasts": {}}
    meaningful = config["practically_meaningful_effect"]
    for contrast, treatment, control in CONTRASTS:
        expected = EXPECTED_DIRECTION[contrast]
        entry = {"treatment": treatment, "control": control, "expected_direction": "+" if expected > 0 else "-",
                 "power_at_meaningful_effect": None, "holm_rejection_rate_at_meaningful_effect": None,
                 "arm_rates_at_meaningful_effect": None}
        if _room(config, contrast, expected) >= meaningful:
            effects = dict(config["effects"], **{contrast: expected * meaningful})
            raw = evaluate(sim, config, contrast, expected * meaningful, cache)
            entry.update(power_at_meaningful_effect=raw if confirmatory else 0.0,
                         holm_rejection_rate_at_meaningful_effect=raw,
                         arm_rates_at_meaningful_effect=_rounded(arm_rates(config, effects)))
        mde = minimal_detectable_effect(sim, config, contrast, expected, cache)
        opposite = minimal_detectable_effect(sim, config, contrast, -expected, cache)
        if confirmatory:
            entry["mde_expected_direction"], entry["mde_opposite_direction"] = mde, opposite
        else:
            reason = {"mde": None, "reason": result["minimum_data_rule"]}
            entry["mde_expected_direction"], entry["mde_opposite_direction"] = dict(reason), dict(reason)
            entry["mde_expected_direction_ignoring_minimum_data"] = mde
            entry["mde_opposite_direction_ignoring_minimum_data"] = opposite
        entry["evaluations"] = sorted(([e, p] for (h, e), p in cache.items() if h == contrast), key=lambda x: x[0])
        result["contrasts"][contrast] = entry
    return result


def run(params, *, only=None):
    np = _numpy()
    validate_params(params)
    scenarios = params.get("scenarios") or [None]
    results = []
    for k, scenario in enumerate(scenarios):
        if only and scenario and scenario["name"] not in only:
            continue
        config = resolve(params, scenario)
        results.append(run_configuration(np, config, params["seed"] + 1000 * k))
    return results


def render_md(report):
    lines = [f"# Exp A power simulation ({report['schema_version']})", "", report["params"]["label"], "",
             f"Params sha256 `{report['params_sha256']}`; numpy {report['numpy']}. Holm across H1-H3, two-sided t "
             f"test on family differences, alpha {report['params']['alpha']}, target power "
             f"{report['params']['power_target']}. MDE = minimal detectable absolute recovery difference.", "",
             "| Scenario | Families | Confirmatory | Simulated rates C0/C1/C2/C3/C4a | FWER (null) | H1 power@0.10 "
             "| H1 MDE | H2 power@0.10 | H2 MDE | H3 power@0.10 | H3 MDE |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]

    def fmt(value):
        return "n/a" if value is None else f"{value:.3f}"
    for r in report["scenarios"]:
        confirmatory = r.get("confirmatory_possible", True)
        rates = (r.get("simulated_arm_rates") or {}).get("base_rates") or r["base_rates"]
        cells = [r["name"], str(r["families"]), "yes" if confirmatory else "no: exploratory (D11.3)",
                 "/".join(f"{rates[c]:.3f}" for c in CONDITIONS), fmt(r["familywise_error_global_null"])]
        for contrast in ("H1", "H2", "H3"):
            entry = r["contrasts"][contrast]
            if confirmatory:
                cells += [fmt(entry["power_at_meaningful_effect"]), fmt(entry["mde_expected_direction"].get("mde"))]
            else:
                cells += ["0 (exploratory)", "n/a (exploratory)"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "MDE in the expected direction (H1 > 0, H2 > 0, H3 < 0); n/a = not reached within the rate bounds "
                  "or max_effect. Power varies one contrast; the others stay at the background effects implied by "
                  "the base rates (H1 = C3 - C2, H2 = C4a - C1, H3 = C1 - C0). A configuration with fewer than "
                  "min_families_confirmatory families is exploratory: the analysis never rejects there, so its "
                  "confirmatory power is 0 (the unrestricted Holm rates are in POWER.json under "
                  "*_ignoring_minimum_data). Every evaluation is in POWER.json.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", help="power parameters JSON (expA-power-params-v1)")
    parser.add_argument("--out", help="output directory for POWER.json and POWER.md")
    parser.add_argument("--only", default=None, help="comma-separated scenario names to run")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-default-params", metavar="PATH", help="write the placeholder parameter grid")
    args = parser.parse_args(argv)
    if args.write_default_params:
        path = Path(args.write_default_params)
        if path.exists() and not args.overwrite:
            raise SystemExit(f"refusing to overwrite {path}")
        path.write_bytes((json.dumps(default_params(), indent=1, sort_keys=True) + "\n").encode("utf-8"))
        print(json.dumps({"written": str(path)}))
        return 0
    if not (args.params and args.out):
        parser.error("--params and --out are required")
    raw = Path(args.params).read_bytes()
    params = json.loads(raw)
    np = _numpy()
    results = run(params, only=set(args.only.split(",")) if args.only else None)
    report = {"schema_version": OUTPUT_SCHEMA, "params": params, "params_sha256": hashlib.sha256(raw).hexdigest(),
              "numpy": np.__version__, "model": "latent Gaussian threshold model with family, item and residual "
                                                 "components; see scripts/expA_power.py",
              "scenarios": results}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("POWER.json", (json.dumps(report, indent=1, sort_keys=True) + "\n").encode("utf-8")),
                       ("POWER.md", render_md(report).encode("utf-8"))):
        path = out / name
        if path.exists() and not args.overwrite:
            raise SystemExit(f"refusing to overwrite {path}")
        path.write_bytes(data)
    print(json.dumps({"out": str(out), "scenarios": len(results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
