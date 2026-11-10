"""Exp A ledger v2: contract cell ids, leases, fencing, infra re-attempts and their cap, segment order (D10.8),
recovery, NFS refusal and durable rebuild."""
import json
import os
from pathlib import Path
import socket
import stat
import tempfile
import unittest

from absent_edges.probe import ledger as L
from absent_edges.registry import FenceError

LOCAL = [("/", "xfs", "/dev/root")]


def no_magic(path):
    return None


def cell(index, block="P", seed=11, variant="omit_node", condition="C1"):
    return L.make_cell(block=block, item_id=f"item{index}", family="fam", target=f"pkg/m.py::f{index}",
                       variant=variant, condition=condition, seed=seed, wording_pair="A",
                       payload_sha256=None if condition in ("C0", "T-offered") else "a" * 64)


def cells(n=3):
    return [cell(i) for i in range(n)]


class FakeRun:
    """Minimal stand-in for the runner's claim -> durable start -> dispatch -> finish sequence."""

    def __init__(self, directory, n=3, identity=None, max_attempts=3, specs=None):
        self.directory = Path(directory)
        self.store = L.DurableStore(self.directory / "run")
        self.cells = specs if specs is not None else cells(n)
        self.store.initialize({"run_id": "run-test"}, self.cells)
        self.identity = identity or {"host": "fixture-host", "boot_id": "boot", "pid": 4242, "start_ticks": 7}
        self.ledger = self.open(max_attempts)

    def open(self, max_attempts=3):
        return L.CellLedger(self.directory / "local" / "ledger.sqlite3", run_id="run-test",
                            max_attempts=max_attempts, mounts=LOCAL, magic=no_magic)

    def claim(self, owner="o1", eligible=None):
        claim = self.ledger.claim(owner, self.identity, lease_seconds=60, eligible=eligible)
        if claim is None:
            return None
        record = {"run_id": "run-test", "cell_id": claim["cell_id"], "attempt_id": claim["attempt_id"],
                  "ordinal": claim["ordinal"], "token": claim["token"], "owner": owner,
                  "owner_identity": self.identity, "started_unix": 1.0}
        directory, sha = self.store.start_attempt(claim["cell_id"], claim["attempt_id"], record)
        self.ledger.mark_started(claim["cell_id"], owner, claim["token"], sha)
        claim["directory"] = directory
        return claim

    def result(self, claim, exit_status, owner="o1", finish=True):
        sha = L.write_immutable_json(claim["directory"] / "result.json",
                                     {"cell_id": claim["cell_id"], "attempt_id": claim["attempt_id"],
                                      "exit_status": exit_status})
        if finish:
            return self.ledger.finish(claim["cell_id"], owner, claim["token"], exit_status=exit_status,
                                      result_sha256=sha, error_code="server" if exit_status == L.INFRA_EXIT else None)
        return sha


class CellIdentityTests(unittest.TestCase):
    def test_contract_cell_ids(self):
        first = cell(0)
        self.assertEqual(first["cell_id"], "P|item0|omit_node|C1|s11")
        self.assertEqual(L.cell_identifier(first), first["cell_id"])
        c0 = cell(1, condition="C0", variant="none", seed=47)
        self.assertEqual((c0["cell_id"], c0["payload_sha256"]), ("P|item1|none|C0|s47", None))
        t = cell(2, block="T", variant="none", condition="T-offered")
        self.assertEqual(t["cell_id"], "T|item2|none|T-offered|s11")
        self.assertEqual(set(first) - {"cell_id"}, set(L.CELL_FIELDS))
        self.assertEqual(L.attempt_identifier(first["cell_id"], 2), "P|item0|omit_node|C1|s11.a002")

    def test_invalid_specs_are_refused(self):
        good = {key: value for key, value in cell(0).items() if key != "cell_id"}
        bad = [dict(good, seed="11"), dict(good, seed=-1), dict(good, wording_pair="C"),
               dict(good, payload_sha256="A" * 64), dict(good, payload_sha256="x"), dict(good, item_id="a|b"),
               dict(good, item_id="a/b"), dict(good, item_id=""), dict(good, block="P Q"), dict(good, family=""),
               dict(good, target=None), dict(good, split="A")]
        for spec in bad:
            with self.subTest(spec={k: spec[k] for k in spec if spec[k] != good.get(k)}), \
                    self.assertRaises(ValueError):
                L.make_cell(**spec)
        missing = dict(good)
        del missing["wording_pair"]
        with self.assertRaises(ValueError):
            L.make_cell(**missing)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run = FakeRun(self.temporary.name)
        self.run.ledger.enqueue(self.run.cells)

    def tearDown(self):
        self.temporary.cleanup()

    def test_completed_cells_are_never_dispatched_again(self):
        seen = []
        while True:
            claim = self.run.claim()
            if claim is None:
                break
            seen.append(claim["cell_id"])
            self.assertEqual(self.run.result(claim, "Submitted"), "completed")
        self.assertEqual(seen, [c["cell_id"] for c in self.run.cells])  # queue order is the frozen order
        self.assertIsNone(self.run.ledger.claim("o2", self.run.identity))
        self.assertEqual(self.run.ledger.counts()["cells"], {"completed": 3})
        self.assertEqual(self.run.ledger.enqueue(self.run.cells), 0)
        with self.assertRaises(ValueError):
            self.run.ledger.unblock(seen[0], "reopen")

    def test_every_model_exit_completes_the_cell(self):
        run = FakeRun(Path(self.temporary.name) / "exits", n=4)
        run.ledger.enqueue(run.cells)
        for status in ("Submitted", "NoAnswer", "ContextLimit", "FormatExhausted"):
            self.assertEqual(run.result(run.claim(), status), "completed")
        self.assertIsNone(run.claim())

    def test_infra_interruption_gets_a_new_attempt_id_and_keeps_the_old(self):
        claim = self.run.claim()
        self.assertTrue(claim["attempt_id"].endswith(".a001"))
        self.assertEqual(self.run.result(claim, "InfraInterrupted"), "infra_interrupted")
        again = self.run.claim(eligible={claim["cell_id"]})
        self.assertEqual(again["cell_id"], claim["cell_id"])
        self.assertTrue(again["attempt_id"].endswith(".a002"))
        self.assertGreater(again["token"], claim["token"])
        self.assertEqual(self.run.result(again, "NoAnswer"), "completed")
        attempts = self.run.ledger.attempts(claim["cell_id"])
        self.assertEqual([(a["ordinal"], a["status"], a["exit_status"]) for a in attempts],
                         [(1, "infra_interrupted", "InfraInterrupted"), (2, "completed", "NoAnswer")])
        self.assertTrue((claim["directory"] / "result.json").exists())
        self.assertTrue((again["directory"] / "attempt_start.json").exists())

    def test_three_interrupted_attempts_block_the_cell(self):
        run = FakeRun(Path(self.temporary.name) / "capped", n=1)  # D11.3: at most 3 InfraInterrupted attempts
        run.ledger.enqueue(run.cells)
        for _ in range(3):
            run.result(run.claim(), "InfraInterrupted")
        self.assertIsNone(run.ledger.claim("o1", run.identity))
        state = run.ledger.cell(run.cells[0]["cell_id"])
        self.assertEqual((state["status"], state["error_code"]), ("blocked", "infra_attempts_exhausted"))
        self.assertEqual(len(run.ledger.attempts(state["cell_id"])), 3)
        run.ledger.unblock(state["cell_id"], "coordinator decision recorded as a deviation")
        self.assertTrue(run.claim()["attempt_id"].endswith(".a004"))

    def test_fencing_rejects_stale_or_foreign_owners(self):
        claim = self.run.claim("o1")
        cell_id = claim["cell_id"]
        with self.assertRaises(FenceError):
            self.run.ledger.heartbeat(cell_id, "o2", claim["token"])
        with self.assertRaises(FenceError):
            self.run.ledger.finish(cell_id, "o1", claim["token"] + 1, exit_status="Submitted", result_sha256="x")
        with self.assertRaises(FenceError):
            self.run.ledger.mark_started(cell_id, "o1", claim["token"], "x")
        self.run.ledger.heartbeat(cell_id, "o1", claim["token"])
        self.run.result(claim, "Submitted")
        with self.assertRaises(FenceError):
            self.run.ledger.finish(cell_id, "o1", claim["token"], exit_status="Submitted", result_sha256="x")
        second = self.run.claim("o1")
        with self.assertRaises(ValueError):
            self.run.ledger.finish(second["cell_id"], "o1", second["token"], exit_status="Bogus", result_sha256="x")
        self.assertEqual(self.run.ledger.cell(second["cell_id"])["status"], "running")

    def test_unstarted_release_and_block(self):
        claim = self.run.ledger.claim("o1", self.run.identity)
        self.run.ledger.release_unstarted(claim["cell_id"], "o1", claim["token"], "start_record_failed")
        self.assertEqual(self.run.ledger.attempts(claim["cell_id"])[0]["status"], "not_dispatched")
        again = self.run.ledger.claim("o1", self.run.identity)
        self.run.ledger.block(again["cell_id"], "episode_input:KeyError", owner="o1", token=again["token"])
        self.assertEqual(self.run.ledger.cell(again["cell_id"])["status"], "blocked")
        other = [c for c in self.run.cells if c["cell_id"] != again["cell_id"]][0]["cell_id"]
        self.run.ledger.block(other, "coordinator_hold")
        with self.assertRaises(ValueError):
            self.run.ledger.block(other, "again")

    def test_recover_expired_respects_owner_state(self):
        claim = self.run.claim("o1")
        later = 10**12
        for state in ("alive", "unknown"):
            outcome = self.run.ledger.recover_expired(lambda identity: state, now=later)
            self.assertEqual(outcome[0]["recovery"], "retained")
        abandoned = []
        outcome = self.run.ledger.recover_expired(lambda identity: "gone", now=later,
                                                  on_abandon=lambda *args: abandoned.append(args))
        self.assertEqual(outcome[0]["recovery"], "abandoned")
        self.assertEqual(abandoned[0][:2], (claim["cell_id"], claim["attempt_id"]))
        with self.assertRaises(FenceError):
            self.run.ledger.finish(claim["cell_id"], "o1", claim["token"], exit_status="Submitted", result_sha256="x")
        self.run.store.close_attempt(claim["cell_id"], claim["attempt_id"], {"status": "abandoned"})
        self.assertTrue((claim["directory"] / "attempt_closed.json").exists())
        self.assertEqual(self.run.ledger.attempts(claim["cell_id"])[0]["status"], "abandoned")
        retry = self.run.claim("o2", eligible={claim["cell_id"]})
        self.assertEqual((retry["cell_id"], retry["ordinal"]), (claim["cell_id"], 2))

    def test_an_outage_is_spread_over_cells_not_one_cell(self):
        interrupted = []
        for _ in range(3):
            claim = self.run.claim()
            interrupted.append(claim["cell_id"])
            self.run.result(claim, "InfraInterrupted")
        self.assertEqual(len(set(interrupted)), 3)  # requeued at the back, so no cell repeats
        self.assertEqual(self.run.ledger.counts()["cells"], {"pending": 3})

    def test_run_id_binding(self):
        with self.assertRaises(L.UnsafeLedgerLocation):
            L.CellLedger(self.run.ledger.path, run_id="another-run", mounts=LOCAL, magic=no_magic)


class SegmentOrderTests(unittest.TestCase):
    """D10.8: a requeued cell goes to the back of its own (block, seed) segment, never behind a later seed."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        specs = ([cell(i, seed=11) for i in range(3)] + [cell(i, seed=23) for i in range(2)]
                 + [cell(i, block="S-VC", variant="vcomplete", seed=11) for i in range(2)])
        self.run = FakeRun(self.temporary.name, specs=specs)
        self.run.ledger.enqueue(self.run.cells)

    def tearDown(self):
        self.temporary.cleanup()

    def test_interrupted_cell_returns_to_the_back_of_its_segment(self):
        first = self.run.claim()
        self.assertEqual(first["cell_id"], "P|item0|omit_node|C1|s11")
        self.run.result(first, "InfraInterrupted")
        order = []
        while True:
            claim = self.run.claim()
            if claim is None:
                break
            order.append((claim["cell_id"], claim["ordinal"]))
            self.run.result(claim, "Submitted")
        self.assertEqual(order, [("P|item1|omit_node|C1|s11", 1), ("P|item2|omit_node|C1|s11", 1),
                                 ("P|item0|omit_node|C1|s11", 2),
                                 ("P|item0|omit_node|C1|s23", 1), ("P|item1|omit_node|C1|s23", 1),
                                 ("S-VC|item0|vcomplete|C1|s11", 1), ("S-VC|item1|vcomplete|C1|s11", 1)])

    def test_eligible_selection_keeps_segment_order(self):
        eligible = {c["cell_id"] for c in self.run.cells if c["block"] == "S-VC" or c["seed"] == 23}
        claim = self.run.claim(eligible=eligible)
        self.assertEqual(claim["cell_id"], "P|item0|omit_node|C1|s23")
        snapshot = self.run.ledger.snapshot()
        self.assertEqual([row["segment"] for row in snapshot["cells"]], [0, 0, 0, 1, 1, 2, 2])


class DurableTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_immutable_writes_never_overwrite(self):
        path = self.directory / "record.json"
        sha = L.write_immutable_json(path, {"a": 1})
        self.assertEqual(sha, L.sha256_path(path))
        with self.assertRaises(FileExistsError):
            L.write_immutable_json(path, {"a": 2})
        self.assertEqual(json.loads(path.read_bytes()), {"a": 1})
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
        self.assertEqual([p.name for p in self.directory.iterdir()], ["record.json"])

    def test_run_initialization_is_frozen(self):
        store = L.DurableStore(self.directory / "run")
        frozen = store.initialize({"run_id": "r"}, cells(2))
        self.assertEqual((frozen["cell_count"], frozen["schema"]), (2, L.RUN_SCHEMA))
        store.initialize({"run_id": "r"}, cells(2))
        with self.assertRaises(L.LedgerInconsistency):
            store.initialize({"run_id": "r"}, cells(3))
        with self.assertRaises(L.LedgerInconsistency):
            store.initialize({"run_id": "r2"}, cells(2))
        self.assertEqual(store.load_cells(), cells(2))
        tampered = dict(cells(1)[0], seed=23)
        with self.assertRaises(ValueError):
            L.DurableStore(self.directory / "other").initialize({}, [tampered])

    def test_attempt_directories_use_the_contract_cell_id(self):
        run = FakeRun(self.directory, n=1)
        run.ledger.enqueue(run.cells)
        claim = run.claim()
        self.assertEqual(claim["directory"], self.directory / "run" / "cells" / "P|item0|omit_node|C1|s11"
                         / "P|item0|omit_node|C1|s11.a001")
        self.assertEqual(sorted(run.store.scan()), ["P|item0|omit_node|C1|s11"])

    def test_rebuild_from_durable_records_after_losing_sqlite(self):
        run = FakeRun(self.directory, n=4)
        run.ledger.enqueue(run.cells)
        done, interrupted, crashed, unfinished = (run.claim() for _ in range(4))
        self.assertEqual(len({c["cell_id"] for c in (done, interrupted, crashed, unfinished)}), 4)
        run.result(done, "Submitted")
        run.result(interrupted, "InfraInterrupted")
        # crashed: start record, never a result, owner later gone.
        run.result(unfinished, "FormatExhausted", finish=False)  # result written, crash before finish
        for suffix in ("", "-wal", "-shm"):
            Path(str(run.ledger.path) + suffix).unlink(missing_ok=True)
        fresh = run.open()
        report = fresh.reconcile(run.store, inspect=lambda identity: "gone")
        self.assertEqual(report["completed_from_durable"], 2)
        self.assertEqual(report["closed_gone"], 1)
        self.assertEqual(fresh.cell(done["cell_id"])["status"], "completed")
        self.assertEqual(fresh.cell(unfinished["cell_id"])["exit_status"], "FormatExhausted")
        self.assertTrue((crashed["directory"] / "attempt_closed.json").exists())
        statuses = {a["attempt_id"]: a["status"] for a in fresh.attempts()}
        self.assertEqual(statuses[interrupted["attempt_id"]], "infra_interrupted")
        self.assertEqual(statuses[crashed["attempt_id"]], "abandoned")
        claimed = []
        while True:
            claim = fresh.claim("o9", run.identity)
            if claim is None:
                break
            claimed.append((claim["cell_id"], claim["ordinal"]))
            fresh.release_unstarted(claim["cell_id"], "o9", claim["token"], "test")
            fresh.block(claim["cell_id"], "test_hold")
        self.assertEqual(sorted(claimed), sorted([(interrupted["cell_id"], 2), (crashed["cell_id"], 2)]))
        again = fresh.reconcile(run.store, inspect=lambda identity: "gone")
        self.assertEqual(again["completed_from_durable"], 0)
        self.assertEqual(again["closed_gone"], 0)

    def test_reconcile_keeps_live_owners_and_never_reuses_ordinals(self):
        run = FakeRun(self.directory, n=2)
        run.ledger.enqueue(run.cells)
        live = run.claim()
        start_less = run.store.attempt_dir(run.cells[1]["cell_id"], L.attempt_identifier(run.cells[1]["cell_id"], 1))
        start_less.mkdir(parents=True)
        for suffix in ("", "-wal", "-shm"):
            Path(str(run.ledger.path) + suffix).unlink(missing_ok=True)
        fresh = run.open()
        report = fresh.reconcile(run.store, inspect=lambda identity: "alive")
        self.assertEqual(report["live_retained"], 1)
        self.assertEqual(fresh.cell(live["cell_id"])["status"], "running")
        claim = fresh.claim("o2", run.identity)
        self.assertEqual((claim["cell_id"], claim["ordinal"]), (run.cells[1]["cell_id"], 2))
        self.assertIsNone(fresh.claim("o3", run.identity))
        self.assertEqual(fresh.finish(live["cell_id"], "o1", live["token"], exit_status="Submitted",
                                      result_sha256="x"), "completed")

    def test_completed_result_lookup(self):
        run = FakeRun(self.directory, n=1)
        run.ledger.enqueue(run.cells)
        claim = run.claim()
        self.assertIsNone(run.store.completed_result(claim["cell_id"]))
        run.result(claim, "ContextLimit", finish=False)
        found = run.store.completed_result(claim["cell_id"])
        self.assertEqual((found["attempt_id"], found["result"]["exit_status"]), (claim["attempt_id"], "ContextLimit"))

    def test_old_protocol_ledger_is_refused(self):
        run = FakeRun(self.directory, n=1)
        import sqlite3
        with sqlite3.connect(run.ledger.path) as db:
            db.execute("UPDATE metadata SET value='expA-ledger-v1' WHERE key='protocol'")
        with self.assertRaises(L.UnsafeLedgerLocation):
            run.open()


class FilesystemGuardTests(unittest.TestCase):
    MOUNTINFO = "\n".join([
        "22 1 259:1 / / rw,relatime shared:1 - ext4 /dev/root rw",
        "30 22 0:25 / /dev/shm rw,nosuid,nodev shared:2 - tmpfs tmpfs rw",
        "40 22 0:40 / /mnt/kldata rw,relatime shared:3 - nfs4 core2:/mnt/kldata rw,vers=4.2",
        "41 22 0:41 / /mnt/sshfs rw - fuse.sshfs user@host:/x rw",
        "42 22 0:42 / /mnt/with\\040space rw - xfs /dev/sdb rw",
        "43 40 0:43 / /mnt/kldata/local rw - xfs /dev/sdc rw",
    ])

    def test_mountinfo_parsing_with_optional_fields_and_escapes(self):
        mounts = L.parse_mountinfo(self.MOUNTINFO)
        self.assertIn(("/mnt/with space", "xfs", "/dev/sdb"), mounts)
        self.assertEqual(mounts[2], ("/mnt/kldata", "nfs4", "core2:/mnt/kldata"))

    def test_network_and_fuse_paths_are_refused(self):
        mounts = L.parse_mountinfo(self.MOUNTINFO)
        for path in ("/mnt/kldata/absent-edges/expA/ledger.sqlite3", "/mnt/sshfs/x.sqlite3"):
            with self.subTest(path=path), self.assertRaises(L.UnsafeLedgerLocation):
                L.require_local_filesystem(path, mounts=mounts, magic=no_magic)
        for path, fstype in (("/dev/shm/ae-expA-1/ledger.sqlite3", "tmpfs"), ("/var/lib/x.sqlite3", "ext4"),
                             ("/mnt/kldata/local/x.sqlite3", "xfs"), ("/mnt/with space/x", "xfs")):
            with self.subTest(path=path):
                self.assertEqual(L.require_local_filesystem(path, mounts=mounts, magic=no_magic)["fstype"], fstype)
        with self.assertRaises(L.UnsafeLedgerLocation):
            L.require_local_filesystem("/dev/shm/x", mounts=mounts, magic=lambda path: 0x6969)
        with self.assertRaises(L.UnsafeLedgerLocation):
            L.CellLedger("/mnt/kldata/absent-edges/ledger.sqlite3", run_id="r", mounts=mounts, magic=no_magic)
        shadowed = mounts + [("/dev/shm", "nfs", "server:/shadow")]
        with self.assertRaises(L.UnsafeLedgerLocation):
            L.require_local_filesystem("/dev/shm/x", mounts=shadowed, magic=no_magic)

    @unittest.skipUnless(os.path.exists("/proc/self/mountinfo"), "Linux mount table required")
    def test_real_mount_table_accepts_local_temporary_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            location = L.require_local_filesystem(Path(directory) / "ledger.sqlite3")
            self.assertFalse(L.is_network_fstype(location["fstype"]))
            ledger = L.CellLedger(Path(directory) / "ledger.sqlite3", run_id="real")
            self.assertEqual(ledger.counts(), {"cells": {}, "attempts": {}})

    @unittest.skipUnless(os.path.exists("/proc/self/stat"), "Linux /proc required")
    def test_process_identity_inspection(self):
        identity = L.process_identity()
        self.assertEqual(L.inspect_identity(identity), "alive")
        self.assertEqual(L.inspect_identity(dict(identity, start_ticks=identity["start_ticks"] + 1)), "gone")
        self.assertEqual(L.inspect_identity(dict(identity, boot_id="another-boot")), "gone")
        self.assertEqual(L.inspect_identity(dict(identity, host=socket.gethostname() + "-other")), "unknown")
        self.assertEqual(L.inspect_identity(None), "gone")


if __name__ == "__main__":
    unittest.main()
