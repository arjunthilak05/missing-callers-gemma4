"""SYNTHETIC tests for scripts/expA_followup_analysis.py (ANALYSIS RECORD).

Every score row, timeline row, item and run record below is SYNTHETIC and hand-made so that the right answer can be
computed by hand. None of it is run data. The last test class runs the `hard` path on the real main-run scores when
they are present locally and checks H1-H3 against the frozen main-run ANALYSIS.json (skipped otherwise).
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location("expA_followup_analysis", ROOT / "scripts" / "expA_followup_analysis.py")
F = importlib.util.module_from_spec(_SPEC)
sys.modules["expA_followup_analysis"] = F
_SPEC.loader.exec_module(F)
from expA_reanalyze_independent import t_statistics  # noqa: E402

VARIANT = {"C0": "none", "C1": "omit_node", "C2": "omit_node", "C3": "omit_node", "C4a": "omit_node"}


def row(item, family, cond, seed, rec, *, exit_status=None, state="completed", block="P", variant=None, **extra):
    """SYNTHETIC score row. rec: 1 recovered, 0 submitted without C*, None = NoAnswer (score 0)."""
    variant = variant or VARIANT[cond]
    if exit_status is None:
        exit_status = "NoAnswer" if rec is None else "Submitted"
    if state != "completed":
        exit_status = "InfraInterrupted"
    out = {"cell_id": f"{block}|{item}|{variant}|{cond}|s{seed}", "block": block, "item_id": item,
           "condition": cond, "variant": variant, "seed": seed, "family": family, "cell_state": state,
           "exit_status": exit_status,
           "site_recovered": None if state != "completed" else (0 if rec is None else rec),
           "n_submitted": 0 if exit_status != "Submitted" else 2, "n_unique_sites": 0 if exit_status != "Submitted"
           else 2, "precision_lower": None if exit_status != "Submitted" else 0.5,
           "precision_upper": None if exit_status != "Submitted" else 1.0}
    out.update(extra)
    return out


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    return str(path)


# SYNTHETIC A1 design. Neutral recovers nothing in C0/C1; "none" recovers C0 only in family A.
# Deadline per item (C0, C1):            item diff vs neutral   family mean
#   A: a1 (1, 1) -> 1, a2 (1, 1) -> 1                            1
#   B: b1 (1, 0) -> 1/2, b2 (0, 0) -> 0                          1/4
#   C: c1 (0, 1) -> 1/2, c2 (1, 1) -> 1                          3/4
# primary = (1 + 1/4 + 3/4) / 3 = 2/3
DEADLINE = {"a1": (1, 1), "a2": (1, 1), "b1": (1, 0), "b2": (0, 0), "c1": (0, 1), "c2": (1, 1)}
FAMILY = {"a1": "A", "a2": "A", "b1": "B", "b2": "B", "c1": "C", "c2": "C"}


def finalturn_rows():
    rows = {"neutral": [], "deadline": [], "none": []}
    for item, fam in FAMILY.items():
        d0, d1 = DEADLINE[item]
        rows["deadline"] += [row(item, fam, "C0", 11, d0 or None), row(item, fam, "C1", 11, d1 or None)]
        # neutral: C0 NoAnswer, C1 submitted without C*; a NoAnswer row carrying site_recovered 1 must score 0
        rows["neutral"] += [row(item, fam, "C0", 11, None), row(item, fam, "C1", 11, 0)]
        rows["none"] += [row(item, fam, "C0", 11, 1 if fam == "A" else None), row(item, fam, "C1", 11, 0)]
        for pol in rows:
            rows[pol] += [row(item, fam, c, 11, 1) for c in ("C2", "C3", "C4a")]
            rows[pol].append(row(item, fam, "C0", 23, 1))   # another seed: never used by A1
    rows["neutral"][0]["site_recovered"] = 1                # a1 C0 neutral: NoAnswer with a stray 1 -> scores 0
    return rows


class StatisticsTests(unittest.TestCase):
    def test_degenerate_rules(self):
        zero = F.summarize([Fraction(0)] * 4, test=True)
        self.assertEqual((zero["status"], zero["p_two_sided"], zero["ci_low"], zero["ci_high"]), ("all_zero", 1.0, 0, 0))
        const = F.summarize([Fraction(1, 4)] * 3, test=True)
        self.assertEqual(const["status"], "constant_nonzero")
        self.assertEqual(const["p_two_sided"], 0.25)          # 2 / 2**3
        self.assertIsNone(const["t"])
        one = F.summarize([Fraction(1, 2)], test=True)
        self.assertEqual(one["p_two_sided"], 1.0)              # 2 / 2**1
        self.assertEqual(F.summarize([], test=True)["status"], "not_estimable")

    def test_constant_nonzero_matches_frozen_sign_flip(self):
        sys.path.insert(0, str(ROOT))
        from absent_edges.probe import analysis
        for n in (1, 2, 3, 5):
            frozen = analysis.sign_flip_test([Fraction(1, 3)] * n)
            ours = F.summarize([Fraction(1, 3)] * n, test=True)["p_two_sided"]
            self.assertEqual(Fraction(frozen["p_exact"]["two-sided"]), Fraction(ours).limit_denominator(1 << 20))

    def test_tested_case_uses_t_statistics(self):
        vals = [Fraction(1), Fraction(1, 4), Fraction(3, 4)]
        ours = F.summarize(vals, test=True)
        ref = t_statistics(vals, 0.95)
        self.assertEqual(ours["estimate_exact"], "2/3")
        self.assertEqual((ours["t"], ours["p_two_sided"], ours["ci_low"], ours["ci_high"], ours["df"]),
                         (ref["t"], ref["p_two_sided"], ref["lower"], ref["upper"], 2))

    def test_weighted_cluster_family_weights_reproduce_primary(self):
        diffs = {"a1": Fraction(1), "a2": Fraction(0), "a3": Fraction(1), "b1": Fraction(-1), "c1": Fraction(1, 2),
                 "c2": Fraction(1)}
        fam = {i: i[0] for i in diffs}
        n_f = {f: sum(1 for i in fam if fam[i] == f) for f in set(fam.values())}
        w = F.weighted_cluster(diffs, {i: Fraction(1, n_f[fam[i]]) for i in diffs}, fam)
        primary = F.summarize(F.family_means(diffs, fam).values())
        self.assertEqual(w["estimate_exact"], primary["estimate_exact"])   # (2/3 - 1 + 3/4) / 3 = 5/36
        self.assertEqual(primary["estimate_exact"], "5/36")
        self.assertAlmostEqual(w["ci_low"], primary["ci_low"], places=12)
        self.assertAlmostEqual(w["ci_high"], primary["ci_high"], places=12)

    def test_weighted_cluster_item_weights(self):
        diffs = {"a1": Fraction(1), "a2": Fraction(0), "a3": Fraction(1), "b1": Fraction(-1), "c1": Fraction(1, 2),
                 "c2": Fraction(1)}
        fam = {i: i[0] for i in diffs}
        w = F.weighted_cluster(diffs, {i: Fraction(1) for i in diffs}, fam)
        self.assertEqual(w["estimate_exact"], "5/12")           # (1 + 0 + 1 - 1 + 1/2 + 1) / 6 = 5/12
        # pseudo-values: est + F (S_f - est n_f) / N = 5/12 + 3 (S_f - 5 n_f / 12) / 6
        est = Fraction(5, 12)
        pseudo = [est + Fraction(3, 6) * (s - est * n) for s, n in ((2, 3), (-1, 1), (Fraction(3, 2), 2))]
        ref = t_statistics(pseudo, 0.95)
        self.assertAlmostEqual(w["ci_low"], ref["lower"], places=12)
        self.assertEqual(w["n_families"], 3)


class FinalTurnTests(unittest.TestCase):
    def run_ft(self, rows, extra=()):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["finalturn", "--out", str(Path(tmp) / "out")]
            for pol in F.POLICIES:
                argv += [f"--scores-{pol}", write_jsonl(Path(tmp) / f"{pol}.jsonl", rows[pol])]
            for i, (flag, data) in enumerate(extra):
                argv += [flag, write_jsonl(Path(tmp) / f"extra{i}.jsonl", data)]
            F.main(argv)
            res = json.loads((Path(tmp) / "out" / "RESULTS.json").read_text(encoding="utf-8"))
            md = (Path(tmp) / "out" / "RESULTS.md").read_text(encoding="utf-8")
            with self.assertRaises(SystemExit):            # refuses to overwrite
                F.main(argv)
        return res, md

    def test_primary_known_answer(self):
        res, md = self.run_ft(finalturn_rows())
        p = res["primary"]
        self.assertEqual(p["estimate_exact"], "2/3")
        self.assertEqual(p["family_values"], {"A": 1.0, "B": 0.25, "C": 0.75})
        ref = t_statistics([Fraction(1), Fraction(1, 4), Fraction(3, 4)], 0.95)
        self.assertEqual((p["t"], p["df"], p["p_two_sided"]), (ref["t"], 2, ref["p_two_sided"]))
        self.assertEqual(p["n_items"], 6)
        self.assertEqual(p["reject_at_alpha"], ref["p_two_sided"] <= 0.05)
        self.assertIn("deadline - neutral", md)
        # components: C0 deadline - neutral = (1, 1/2, 1/2) by family; C1 = (1, 0, 1)
        self.assertEqual(res["primary_components_by_condition"]["C0"]["estimate_exact"], "2/3")
        self.assertEqual(res["primary_components_by_condition"]["C1"]["estimate_exact"], "2/3")
        self.assertEqual(res["population"]["rows_excluded"]["deadline"], {"seed_not_selected": 6})

    def test_secondaries_known_answers(self):
        res, _ = self.run_ft(finalturn_rows())
        s = res["secondary"]
        # S1: none recovers C0 in A only -> A item diffs (0 + 1)/2 = 1/2; B, C as primary: 1/2, 1/4, 3/4 -> 1/2
        self.assertEqual(s["S1_deadline_minus_none"]["estimate_exact"], "1/2")
        # S2 vs neutral: neutral list effect 0; deadline C1 - C0 per item: a 0,0; b -1,0; c 1,0 -> A 0, B -1/2, C 1/2
        self.assertEqual(s["S2_list_effect_change_vs_neutral"]["estimate_exact"], "0/1")
        self.assertEqual(s["S2_list_effect_change_vs_neutral"]["family_values"], {"A": 0.0, "B": -0.5, "C": 0.5})
        # S2 vs none: none list effect A -1, B 0, C 0 -> A 0-(-1) = 1, B -1/2, C 1/2 -> 1/3
        self.assertEqual(s["S2_list_effect_change_vs_none"]["estimate_exact"], "1/3")
        self.assertNotIn("p_two_sided", s["S1_deadline_minus_none"])      # no significance claims
        self.assertEqual(s["S2_list_effect_by_policy"]["none"]["estimate_exact"], "-1/3")
        s4 = s["S4_by_policy"]["neutral"]["C0"]
        self.assertEqual(s4["submission_rate"]["micro"]["mean"], 0.0)
        self.assertEqual(s4["recovery_given_submission"]["micro"]["n"], 0)
        s4d = s["S4_by_policy"]["deadline"]["C1"]
        self.assertEqual(s4d["submission_rate"]["micro"]["mean"], 4 / 6)
        self.assertEqual(s4d["recovery_given_submission"]["micro"]["mean"], 1.0)
        self.assertEqual(s4d["submitted_sites"]["micro"]["mean"], 2.0)
        self.assertEqual(s4d["precision_lower"]["family_weighted"]["estimate"], 0.5)
        self.assertIn("frozen scripts/expA_analyze.py", s["S3"])
        self.assertEqual(res["mechanism_subgroup"]["status"], "not_available")

    def test_precision_not_available_is_reported(self):
        rows = finalturn_rows()
        for pol in rows:
            for r in rows[pol]:
                r.pop("precision_lower")
        res, _ = self.run_ft(rows)
        self.assertEqual(res["secondary"]["S4_by_policy"]["none"]["C0"]["precision_lower"]["status"], "not_available")

    def test_unpaired_and_missing_cells_are_counted(self):
        rows = finalturn_rows()
        rows["deadline"] = [r for r in rows["deadline"] if not (r["item_id"] == "b2" and r["condition"] == "C1"
                                                                 and r["seed"] == 11)]
        for r in rows["neutral"]:
            if r["item_id"] == "c1" and r["condition"] == "C0" and r["seed"] == 11:
                r.update(cell_state="missing_infrastructure", exit_status="InfraInterrupted", site_recovered=None)
        res, _ = self.run_ft(rows)
        p = res["primary"]
        # b2 and c1 drop out: A 1, B 1/2 (b1 only), C 1 (c2 only) -> 5/6
        self.assertEqual(p["n_items"], 4)
        self.assertEqual(p["estimate_exact"], "5/6")
        self.assertEqual(p["dropped"]["items_missing_a_needed_cell"], 2)
        self.assertEqual(p["dropped"]["missing:deadline:C1"], 1)
        self.assertEqual(p["dropped"]["missing:neutral:C0"], 1)
        cells = res["population"]["cells"]
        self.assertEqual(cells["deadline"]["C1"], {"absent": 1, "completed": 5})
        self.assertEqual(cells["neutral"]["C0"], {"completed": 5, "not_completed:missing_infrastructure": 1})
        pair = res["population"]["pairing_deadline_vs_neutral"]
        self.assertEqual(pair["C1"], {"only_neutral": 1, "paired": 5})
        self.assertEqual(pair["C0"], {"only_deadline": 1, "paired": 5})
        # S1 (deadline vs none) loses only b2
        self.assertEqual(res["secondary"]["S1_deadline_minus_none"]["n_items"], 5)

    def test_mechanism_subgroup_logic(self):
        rows = finalturn_rows()

        def trow(r, n_prompts, n_responses, strict, problems=()):
            return {"cell_id": r["cell_id"], "exit_status": r["exit_status"], "site_recovered": r["site_recovered"],
                    "prompt_tokens_by_response": [100] * n_prompts, "n_responses": n_responses,
                    "first_cstar_visible_response": {"strict": strict, "loose": strict}, "problems": list(problems),
                    "last_action_kind": "submit_accepted" if r["exit_status"] == "Submitted" else "tool_call"}

        timelines = {}
        for pol in F.POLICIES:
            sel = {(r["item_id"], r["condition"]): r for r in rows[pol] if r["seed"] == 11}
            timelines[pol] = [
                trow(sel[("a1", "C0")], 8, 8, 7),      # open at 8, strict at 7 -> IN
                trow(sel[("a1", "C1")], 8, 8, 8),      # strict reveal 8 is not before response 8 -> out
                trow(sel[("a2", "C0")], 8, 8, None),   # never displayed -> out
                trow(sel[("a2", "C1")], 6, 6, 1),      # finished at response 6 -> out
                trow(sel[("b1", "C0")], 8, 7, 0),      # ContextLimit at the 8th request: open -> IN
                trow(sel[("b1", "C1")], 8, 8, 3, problems=["calls"]),   # timeline problem -> left out
            ]
        # deadline's b1 C0 row is Submitted+recovered at response 8 in this SYNTHETIC case; neutral/none are NoAnswer
        res, md = self.run_ft(rows, extra=[(f"--timeline-{p}", timelines[p]) for p in F.POLICIES])
        m = res["mechanism_subgroup"]
        for pol in F.POLICIES:
            self.assertEqual(m["by_policy"][pol]["subgroup_episodes"], 2)
            self.assertEqual(m["by_policy"][pol]["open_at_8"], 4)
            self.assertEqual(m["timeline_inputs"][pol]["problems"],
                             {"completed_cells_without_timeline_row": 24, "timeline_row_has_problems": 1})
        # deadline: a1 C0 recovered (1), b1 C0 recovered (1); neutral: a1 C0 NoAnswer (stray 1 scores 0), b1 0
        self.assertEqual(m["by_policy"]["deadline"]["by_condition"]["C0"]["micro_recovery"]["mean"], 1.0)
        self.assertEqual(m["by_policy"]["neutral"]["by_condition"]["C0"]["micro_recovery"]["mean"], 0.0)
        self.assertEqual(m["by_policy"]["none"]["by_condition"]["C0+C1"]["micro_recovery"]["mean"], 0.5)
        self.assertEqual(m["in_subgroup_under_all_three_policies"]["cells"], 2)
        kinds = res["secondary"]["S4_final_response_kind"]["by_policy"]["deadline"]["C0"]
        self.assertEqual(kinds["episodes_with_8_responses"], 2)
        self.assertIn("Mechanism subgroup", md)

    def test_timeline_disagreeing_with_scores_is_excluded(self):
        rows = finalturn_rows()
        sel = {(r["item_id"], r["condition"]): r for r in rows["deadline"] if r["seed"] == 11}
        bad = {"cell_id": sel[("a1", "C0")]["cell_id"], "exit_status": "NoAnswer", "site_recovered": 0,
               "prompt_tokens_by_response": [1] * 8, "n_responses": 8, "problems": [],
               "first_cstar_visible_response": {"strict": 2}}
        res, _ = self.run_ft(rows, extra=[("--timeline-deadline", [bad])])
        info = res["mechanism_subgroup"]["timeline_inputs"]["deadline"]
        self.assertEqual(info["problems"]["timeline_row_disagrees_with_score_row"], 1)
        self.assertEqual(res["mechanism_subgroup"]["not_available_for"], ["neutral", "none"])


def _ev(kind, value):
    return {"sequence": 0, "wall_time": 0.0, "kind": kind, "value": value}


SYN_ITEM = {"item_id": "a1", "family": "A", "omitted": {"identity": "pkg/mod.py::Holder.caller_fn", "stratum": "direct"},
            "V": [{"identity": "pkg/mod.py::Holder.caller_fn",
                   "sites": [{"path": "pkg/mod.py", "lineno": 12, "end_lineno": 13}]}],
            "target": {"name": "target_fn", "qualname": "target_fn"}, "names": {"lexical": []}, "covariates": {}}


def _synthetic_episode(n_responses, reveal_at, submit):
    """SYNTHETIC events: bash calls; the result of response `reveal_at` prints pkg/mod.py:12; optional submit."""
    events = [_ev("episode_setup", {"user": "SYNTHETIC task", "system": "SYNTHETIC"})]
    for i in range(n_responses):
        events.append(_ev("context_accounting", {"call_index": i, "count": 100}))
        events.append(_ev("model_exchange", {"call_index": i, "response": {"usage": {"completion_tokens": 5}}}))
        name = "submit_answer" if submit and i == n_responses - 1 else "bash"
        action = {"id": f"c{i}", "name": name, "arguments": {"command": "ls"} if name == "bash" else {"sites": []}}
        events.append(_ev("parsed_actions", {"call_index": i, "actions": [action]}))
        events.append(_ev("tool_action", action))
        if name == "submit_answer":
            events.append(_ev("submission_accepted", {"tool_call_id": action["id"], "call_index": i}))
            break
        events.append(_ev("tool_output", {"tool_call_id": action["id"], "name": "bash",
                                          "record": {"status": "executed", "argv": [["ls"]]}}))
        out = "pkg/mod.py:12:    target_fn()\n" if i + 1 == reveal_at else "nothing\n"
        events.append(_ev("model_visible_tool_result", {"tool_call_id": action["id"], "name": "bash", "content":
                                                        json.dumps({"exit_code": 0, "output": out,
                                                                    "truncated": False})}))
    events.append(_ev("episode_finished", {"exit_status": "Submitted" if submit else "NoAnswer"}))
    return events


class RunDirTests(unittest.TestCase):
    def make_run(self, root, policy, episodes):
        """SYNTHETIC run dir: run.json + one result.json/events.jsonl per cell; returns the score rows."""
        root.mkdir(parents=True)
        (root / "run.json").write_text(json.dumps({"episode_config": {} if policy == "none" else
                                                   {"final_turn_policy": policy}}), encoding="utf-8")
        rows = []
        for cond, (n, reveal, submit, rec) in episodes.items():
            r = row("a1", "A", cond, 11, rec if submit else None)
            cell = root / "cells" / cond
            cell.mkdir(parents=True)
            events = _synthetic_episode(n, reveal, submit)
            raw_events = "".join(json.dumps(e) + "\n" for e in events).encode()
            (cell / "events.jsonl").write_bytes(raw_events)
            result = {"events": {"path": "events.jsonl", "sha256": hashlib.sha256(raw_events).hexdigest(),
                                 "records": len(events)}, "exit_status": r["exit_status"], "calls": n,
                      "tokens": {"prompt": 100 * n, "completion": 5 * n}}
            raw_result = json.dumps(result).encode()
            (cell / "result.json").write_bytes(raw_result)
            r.update(result_path=f"cells/{cond}/result.json", result_sha256=hashlib.sha256(raw_result).hexdigest(),
                     attempt_id=r["cell_id"] + ".a001")
            rows.append(r)
        return rows

    def test_run_dir_path_matches_timeline_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            argv = ["finalturn", "--out", str(tmp / "out"), "--items", write_jsonl(tmp / "items.jsonl", [SYN_ITEM])]
            # C0: open at 8, revealed by response 3's result -> IN; C1: revealed by response 8's result -> out;
            # C2: submitted at response 5 -> out
            episodes = {"C0": (8, 3, False, 0), "C1": (8, 8, False, 0), "C2": (5, 2, True, 1)}
            for pol in F.POLICIES:
                rows = self.make_run(tmp / pol, pol, episodes)
                argv += [f"--scores-{pol}", write_jsonl(tmp / f"{pol}.jsonl", rows), f"--run-dir-{pol}",
                         str(tmp / pol)]
            F.main(argv)
            res = json.loads((tmp / "out" / "RESULTS.json").read_text(encoding="utf-8"))
            m = res["mechanism_subgroup"]
            for pol in F.POLICIES:
                self.assertEqual(m["timeline_inputs"][pol]["problems"], {})
                self.assertEqual(m["by_policy"][pol]["subgroup_episodes"], 1)
                self.assertEqual(m["by_policy"][pol]["by_condition"]["C0"]["micro_recovery"]["n"], 1)
            # a run.json whose policy disagrees is refused
            bad = list(argv)
            bad[bad.index(str(tmp / "neutral"))] = str(tmp / "deadline")
            bad[bad.index("--out") + 1] = str(tmp / "out2")
            with self.assertRaises(SystemExit):
                F.main(bad)


# SYNTHETIC A2 panel: 3 families; seeds 11, 23 complete; seed 47 has an infra_pending cell -> primary seeds (11, 23)
# H3 item differences (C1 mean over seeds - C0 mean over seeds):
#   A: h1 C1 (1,1) C0 (0,0) -> 1;   h2 C1 (1,0) C0 (0,0) -> 1/2     family 3/4     C*: X, X
#   B: h3 C1 (1,1) C0 (1,0) -> 1/2                                  family 1/2     C*: Y
#   C: h4 C1 (0,0) C0 (0,0) -> 0;   h5 C1 (1,1) C0 (0,1) -> 1/2     family 1/4     C*: Z, W
# primary 1/2; item-weighted (1 + 1/2 + 1/2 + 0 + 1/2) / 5 = 1/2; caller-balanced: X (1+1/2)/2 = 3/4, Y 1/2, Z 0,
# W 1/2 -> (3/4 + 1/2 + 0 + 1/2) / 4 = 7/16
HARD = {"h1": ("A", "X", (1, 1), (0, 0)), "h2": ("A", "X", (1, 0), (0, 0)), "h3": ("B", "Y", (1, 1), (1, 0)),
        "h4": ("C", "Z", (0, 0), (0, 0)), "h5": ("C", "W", (1, 1), (0, 1))}


def hard_fixture():
    rows, items = [], []
    for item, (fam, cstar, c1, c0) in HARD.items():
        items.append({"item_id": item, "family": fam, "omitted": {"identity": cstar, "stratum": "dynamic"}})
        for k, seed in enumerate((11, 23)):
            rows += [row(item, fam, "C1", seed, c1[k] or None), row(item, fam, "C0", seed, c0[k] or None)]
            rows += [row(item, fam, "C2", seed, 0), row(item, fam, "C3", seed, 1 if fam == "A" else 0),
                     row(item, fam, "C4a", seed, c1[k] or None)]
        state = "infra_pending" if item == "h1" else "completed"
        rows += [row(item, fam, c, 47, 0, state=state) for c in F.CONDITIONS]
        rows += [row(item, fam, "C1", 11, 1, block="S-VC", variant="vcomplete"),
                 row(item, fam, "C3", 11, 1 if item != "h5" else None, block="S-VC", variant="vcomplete")]
    return rows, items


class HardTests(unittest.TestCase):
    def run_hard(self, rows, items, main_rows, items_main=None):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            argv = ["hard", "--scores", write_jsonl(tmp / "s.jsonl", rows), "--main-scores",
                    write_jsonl(tmp / "m.jsonl", main_rows), "--items-hard", write_jsonl(tmp / "i.jsonl", items),
                    "--out", str(tmp / "out")]
            if items_main is not None:
                argv += ["--items-main", write_jsonl(tmp / "im.jsonl", items_main)]
            F.main(argv)
            return json.loads((tmp / "out" / "RESULTS.json").read_text(encoding="utf-8"))

    def test_hard_known_answers(self):
        rows, items = hard_fixture()
        res = self.run_hard(rows, items, rows)
        self.assertEqual(res["population"]["primary_seeds"], [11, 23])
        h3 = res["hypotheses"]["H3"]
        self.assertEqual(h3["estimate_exact"], "1/2")
        self.assertEqual(h3["family_values"], {"A": 0.75, "B": 0.5, "C": 0.25})
        ref = t_statistics([Fraction(3, 4), Fraction(1, 2), Fraction(1, 4)], 0.95)
        self.assertEqual((h3["p_two_sided"], h3["t"], h3["df"]), (ref["p_two_sided"], ref["t"], 2))
        self.assertEqual(h3["reject_at_alpha"], ref["p_two_sided"] <= 0.05)
        self.assertIn("D-4", h3["decision_rule"])
        # H1 = C3 - C2: A 1, B 0, C 0 -> 1/3; H2 = C4a - C1 = 0; no p for either (D-4)
        self.assertEqual(res["hypotheses"]["H1"]["estimate_exact"], "1/3")
        self.assertEqual(res["hypotheses"]["H2"]["estimate_exact"], "0/1")
        self.assertNotIn("p_two_sided", res["hypotheses"]["H1"])
        sens = res["h3_sensitivity"]
        self.assertEqual(sens["item_weighted"]["estimate_exact"], "1/2")
        self.assertEqual(sens["caller_balanced"]["estimate_exact"], "7/16")
        self.assertEqual(sens["caller_balanced"]["distinct_cstar"], 4)
        svc = res["s_vc_positive_control"]["recovery"]
        self.assertEqual(svc["C1"]["micro_item_mean"], 1.0)
        self.assertEqual(svc["C3"]["micro_item_mean"], 0.8)
        self.assertEqual(svc["C3"]["family_weighted"]["estimate_exact"], "5/6")   # A 1, B 1, C 1/2
        sub = res["submission_rates"]["P"]["C0"]
        self.assertEqual(sub["micro_item_mean"], 0.2)    # only h3 s11 and h5 s23 submitted: (1/2 + 1/2) / 5
        self.assertEqual(res["within_target"]["status"], "not_available")

    def test_within_target_contrasts(self):
        rows, items = hard_fixture()
        main_items = [dict(i) for i in items]
        main_items[0] = {"item_id": "h1", "family": "A", "omitted": {"identity": "D1", "stratum": "direct"}}
        main_items[2] = {"item_id": "h3", "family": "B", "omitted": {"identity": "D3", "stratum": "callback"}}
        main_items[4] = {"item_id": "h5", "family": "C", "omitted": {"identity": "D5", "stratum": "direct"}}
        main_rows = []
        for item in ("h1", "h2", "h3", "h4", "h5"):
            fam = HARD[item][0]
            for seed in (11, 23, 47):
                main_rows += [row(item, fam, c, seed, 1, omitted="main") for c in F.CONDITIONS]
        res = self.run_hard(rows, items, main_rows, items_main=main_items)
        w = res["within_target"]
        self.assertEqual((w["shared_targets"], w["same_cstar"], w["different_cstar"], w["targets"]), (5, 2, 3, 2))
        self.assertEqual(w["different_cstar_main_strata"], {"callback": 1, "direct": 2})
        self.assertFalse(w["targets_match_expected"])
        self.assertEqual(sorted(w["per_target"]), ["h1", "h5"])
        # hard C1 (seeds 11, 23): h1 1, h5 1; main C1 1 -> 0; hard C0: h1 0, h5 1/2 -> -1, -1/2
        self.assertEqual(w["by_condition"]["C1"]["hard_minus_main_micro"]["mean"], 0.0)
        self.assertEqual(w["by_condition"]["C0"]["hard_minus_main_micro"]["mean"], -0.75)
        self.assertEqual(w["h3_on_these_targets"]["hard"]["micro"]["mean"], 0.75)
        self.assertEqual(w["h3_on_these_targets"]["main"]["micro"]["mean"], 0.0)
        self.assertEqual(w["seeds"], {"hard": [11, 23], "main": [11, 23, 47]})

    def test_scores_items_disagreement_is_refused(self):
        rows, items = hard_fixture()
        with self.assertRaises(SystemExit):
            self.run_hard(rows, items[:-1], rows)


def _main_run_dir():
    candidates = [os.environ.get("EXPA_MAIN_ANALYSIS_DIR"),
                  "/mnt/d/C_moved/Documents-work/2026-10-17/expA-backups/main-analysis-20261101b",
                  "D:/C_moved/Documents-work/2026-10-17/expA-backups/main-analysis-20261101b"]
    for c in candidates:
        if c and (Path(c) / "scores.jsonl").is_file() and (Path(c) / "items.jsonl").is_file():
            return Path(c)
    return None


class MainRunCrossCheck(unittest.TestCase):
    """The hard path on the real main-run scores must reproduce the frozen main-run H1-H3 (H3 = +0.156444)."""

    def test_hard_path_reproduces_frozen_main_h3(self):
        base = _main_run_dir()
        analysis = ROOT / "reports" / "expA-main-20261101b" / "analysis" / "ANALYSIS.json"
        if base is None or not analysis.is_file():
            self.skipTest("main-run scores.jsonl / items.jsonl not available locally")
        with tempfile.TemporaryDirectory() as tmp:
            F.main(["hard", "--scores", str(base / "scores.jsonl"), "--main-scores", str(base / "scores.jsonl"),
                    "--items-hard", str(base / "items.jsonl"), "--frozen-analysis", str(analysis),
                    "--out", str(Path(tmp) / "out")])
            res = json.loads((Path(tmp) / "out" / "RESULTS.json").read_text(encoding="utf-8"))
        self.assertEqual(res["hypotheses"]["H3"]["estimate_exact"], "176/1125")    # 0.156444...
        self.assertTrue(all(v["match"] for v in res["frozen_crosscheck"].values()))
        self.assertTrue(res["frozen_crosscheck"]["H3"]["p_match"])
        self.assertEqual((res["hypotheses"]["H3"]["n_families"], res["hypotheses"]["H3"]["n_items"]), (30, 750))


if __name__ == "__main__":
    unittest.main()
