# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0

"""benchmarks/scripts: chain.py runs the steps in order and logs them; monitor.py attributes
its samples to the records of a work folder. No benchmark step runs; no GPU."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "benchmarks" / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"bench_script_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


chain = _load("chain")
monitor = _load("monitor")


def test_descendants_follow_the_tree_from_the_root():
    parents = {1: 0, 10: 1, 11: 10, 12: 10, 20: 1, 30: 30}
    assert monitor._descendants(parents, 10) == {10, 11, 12}
    assert monitor._descendants(parents, 30) == {30}  # a self-parented PID is no cycle
    assert monitor._descendants(parents, 99) == set()
    assert monitor._descendants(parents, None) == set()


def _samples(path: Path, rows: list[dict]) -> None:
    with open(path / "samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=monitor.FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in monitor.FIELDS})


def _record(path: Path, started: str, finished: str, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"started": started, "finished": finished, **extra}))


def test_records_attribute_samples_and_name_the_other_load(tmp_path):
    work, mon = tmp_path / "work", tmp_path / "monitor"
    mon.mkdir()
    base = {"ncores": "8", "load1": "1.0", "cpu_total_pct": "50"}
    _samples(
        mon,
        [
            # quiet during the FastWarp run
            {**base, "utc": "2026-11-17T10:00:00Z", "other_cpu_cores": "0.1", "ac_online": "1"},
            {**base, "utc": "2026-11-17T10:00:05Z", "other_cpu_cores": "0.1", "ac_online": "1"},
            # 2 other cores busy and on battery during the ANTs run
            {**base, "utc": "2026-11-17T10:01:00Z", "other_cpu_cores": "2.0", "ac_online": "0"},
            {**base, "utc": "2026-11-17T10:01:05Z", "other_cpu_cores": "2.0", "ac_online": "0"},
            # another process on the GPU during the post-processing
            {
                **base,
                "utc": "2026-11-17T10:02:00Z",
                "other_cpu_cores": "0",
                "gpu_util_pct": "60",
                "gpu_other_sm_pct": "40",
                "ac_online": "1",
            },
        ],
    )
    pair = work / "snsx" / "cohort" / "a__b"
    _record(
        pair / "fastwarp" / "run.json",
        "2026-11-17T10:00:00Z",
        "2026-11-17T10:00:06Z",
        engine="fastwarp",
        status="ok",
    )
    _record(
        pair / "ants" / "run.json",
        "2026-11-17T10:01:00Z",
        "2026-11-17T10:01:06Z",
        engine="ants",
        status="ok",
    )
    _record(pair / "post.json", "2026-11-17T10:02:00Z", "2026-11-17T10:02:01Z")
    _record(
        work / "snsx" / "linear" / "a__b" / "fastwarp" / "run.json",
        "2026-11-17T11:00:00Z",
        "2026-11-17T11:00:01Z",
    )
    (work / "snsx" / "cohort" / "other.json").write_text("{}")  # not a record

    assert monitor.main(["records", "--work", str(work), "--monitor", str(mon)]) == 0
    with open(mon / "per_record.csv", newline="") as f:
        rows = {r["record"]: r for r in csv.DictReader(f)}

    assert set(rows) == {
        "snsx/cohort/a__b/fastwarp/run.json",
        "snsx/cohort/a__b/ants/run.json",
        "snsx/cohort/a__b/post.json",
        "snsx/linear/a__b/fastwarp/run.json",
    }
    fw = rows["snsx/cohort/a__b/fastwarp/run.json"]
    assert (fw["kind"], fw["engine"], fw["n_samples"], fw["other_load"]) == (
        "run",
        "fastwarp",
        "2",
        "",
    )
    ants = rows["snsx/cohort/a__b/ants/run.json"]
    assert ants["other_load"] == "cpu+battery"
    assert ants["on_battery"] == "yes"
    post = rows["snsx/cohort/a__b/post.json"]
    assert (post["kind"], post["engine"], post["other_load"]) == ("post", "both", "gpu")
    lin = rows["snsx/linear/a__b/fastwarp/run.json"]
    assert (lin["kind"], lin["n_samples"], lin["other_load"]) == ("linear", "0", "no samples")


def test_records_without_per_process_gpu_use_count_all_of_it_as_others_during_ants(tmp_path):
    work, mon = tmp_path / "work", tmp_path / "monitor"
    mon.mkdir()
    _samples(mon, [{"utc": "2026-11-17T10:00:00Z", "ncores": "8", "gpu_util_pct": "50"}])
    pair = work / "deep7t" / "cohort" / "a__b"
    for engine in ("ants", "fastwarp"):
        _record(
            pair / engine / "run.json",
            "2026-11-17T10:00:00Z",
            "2026-11-17T10:00:01Z",
            engine=engine,
        )
    monitor.main(["records", "--work", str(work), "--monitor", str(mon)])
    with open(mon / "per_record.csv", newline="") as f:
        load = {r["engine"]: r["other_load"] for r in csv.DictReader(f)}
    assert load == {"ants": "gpu", "fastwarp": ""}


def test_records_report_unfinished_records_and_skip_broken_sample_rows(tmp_path, capsys):
    work, mon = tmp_path / "work", tmp_path / "monitor"
    mon.mkdir()
    _samples(mon, [{"utc": "2026-11-17T10:00:00Z", "ncores": "8", "other_cpu_cores": "0"}])
    with open(mon / "samples.csv", "a") as f:
        f.write("2026-11-17T10:0")  # cut off by a power loss
    pair = work / "snsx" / "cohort" / "a__b"
    _record(pair / "fastwarp" / "run.json", "2026-11-17T10:00:00Z", "2026-11-17T10:00:01Z")
    (pair / "ants").mkdir(parents=True)
    (pair / "ants" / "run.json").write_text(json.dumps({"started": "2026-11-17T10:00:02Z"}))
    assert monitor.main(["records", "--work", str(work), "--monitor", str(mon)]) == 0
    out = capsys.readouterr().out
    assert "1 records, 0 with other load, 0 without samples" in out
    assert "1 records without start and finish times" in out
    assert "1 unreadable rows" in out


def test_gpu_nvidia_leaves_the_per_process_use_blank_without_pmon(monkeypatch):
    replies = {
        "--query-gpu=utilization.gpu,memory.used,memory.total": "55, 1000, 16000\n",
        "pmon": "",  # not supported, for example in WDDM mode
        "--query-compute-apps=pid,used_memory": "",
    }
    monkeypatch.setattr(monitor, "_run", lambda cmd: replies[cmd[1]])
    assert monitor.gpu_nvidia("nvidia-smi", set()) == ["55", "1000", "16000", "", "", 0.0]
    replies["pmon"] = "# gpu  pid  type  sm  mem\n# Idx  #  C/G  %  %\n    0  42  C  70  10\n"
    replies["pmon"] += "    0  43  C  20  5\n"
    assert monitor.gpu_nvidia("nvidia-smi", {42})[3:5] == [70.0, 20.0]


def test_alive_tells_a_running_from_a_finished_process():
    assert chain._alive(os.getpid())
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert not chain._alive(p.pid)


def _run_chain(tmp_path, monkeypatch, argv, fail=()):
    calls = []

    def call(cmd, cwd=None, stdout=None, stderr=None, env=None):
        calls.append(cmd)
        return 3 if cmd[3] in fail else 0

    monkeypatch.setattr(chain.subprocess, "call", call)
    monkeypatch.setattr(chain, "keep_awake", lambda pid: None)
    rc = chain.main(
        ["--data", str(tmp_path / "data"), "--work", str(tmp_path / "work"), "--no-monitor", *argv]
    )
    return rc, calls


def test_chain_runs_every_step_in_order_with_the_selection(tmp_path, monkeypatch):
    rc, calls = _run_chain(tmp_path, monkeypatch, ["--dataset", "snsx", "--pairs", "2"])
    assert rc == 0
    assert all(c[:3] == [sys.executable, "-m", "benchmarks.accuracy"] for c in calls)
    steps = [c[3:] for c in calls]
    data, work = str((tmp_path / "data").resolve()), str((tmp_path / "work").resolve())
    folders = ["--data", data, "--work", work]
    sel = ["--dataset", "snsx", "--pairs", "2"]
    assert steps == [
        ["fetch", "--data", data, "--dataset", "snsx"],
        ["run", *sel, *folders],
        ["linear", *sel, *folders],
        ["post", *sel, *folders],
        ["evaluate", *sel, *folders],
        ["report", "--work", work],
    ]
    logged = (tmp_path / "logs" / "commands.txt").read_text().splitlines()
    assert len(logged) == 6
    assert logged[1].endswith(" ".join(["-m", "benchmarks.accuracy", "run", *sel, *folders]))


def test_chain_goes_on_after_a_failed_step_and_exits_1(tmp_path, monkeypatch):
    rc, calls = _run_chain(tmp_path, monkeypatch, ["--quick", "--no-fetch"], fail={"linear"})
    assert rc == 1
    assert [c[3] for c in calls] == ["run", "linear", "post", "evaluate", "report"]
    assert "failed: linear" in (tmp_path / "logs" / "chain.log").read_text()


def test_chain_fails_when_the_monitor_dies(tmp_path, monkeypatch):
    class Dead:
        pid, returncode = 1, 2

        def poll(self):
            return 2

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 2

    monkeypatch.setattr(chain.subprocess, "Popen", lambda *a, **k: Dead())
    monkeypatch.setattr(chain.subprocess, "call", lambda *a, **k: 0)
    monkeypatch.setattr(chain, "keep_awake", lambda pid: None)
    monkeypatch.setattr(chain.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        chain.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "ok", "")
    )
    rc = chain.main(
        ["--data", str(tmp_path / "data"), "--work", str(tmp_path / "work"), "--no-fetch"]
    )
    assert rc == 1
    assert "failed: monitor" in (tmp_path / "logs" / "chain.log").read_text()


def test_chain_removes_only_the_lock_of_a_finished_process(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    lock = work / ".run.lock"
    lock.write_text(f"{os.getpid()} live\n")
    _run_chain(tmp_path, monkeypatch, ["--no-fetch"])
    assert lock.exists()
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    lock.write_text(f"{p.pid} gone\n")
    _run_chain(tmp_path, monkeypatch, ["--no-fetch"])
    assert not lock.exists()


def test_chain_refuses_both_pairs_and_quick(tmp_path):
    with pytest.raises(SystemExit):
        chain.main(["--data", str(tmp_path), "--work", str(tmp_path), "--pairs", "2", "--quick"])
