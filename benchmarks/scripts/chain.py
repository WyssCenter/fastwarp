# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0

"""Run the whole benchmark unattended: fetch, run, linear, post, evaluate, report.

Every step is ``python -m benchmarks.accuracy <step>`` from the repository root, with the same
data and work folders; both engines, all cores, the default datasets and pairs unless a
selection is given. A failed step does not stop the later ones; the script exits with 1 if any
step, the monitor or its summary failed. Every step resumes where it stopped, so running this
script again continues an interrupted run. Meanwhile ``monitor.py`` (next to this script)
records the machine's load, and the machine is kept awake where that needs no admin rights
(``caffeinate`` on macOS, ``systemd-inhibit`` on Linux when allowed,
``SetThreadExecutionState`` on Windows).

Written next to the work folder unless ``--logs`` and ``--monitor`` say otherwise:

- ``logs/commands.txt``: every command, in order, as it was run;
- ``logs/chain.log``: start, end and exit code of every step (UTC);
- ``logs/<step>.log``: the step's output, ``logs/monitor.log`` the monitor's;
- ``monitor/samples.csv``, ``monitor/per_record.csv`` and ``monitor/users_at_start.txt`` (see
  ``monitor.py``).

Run it in the benchmark's environment (see "Running it" in ``docs/benchmarks.md``)::

    uv run --no-sync python benchmarks/scripts/chain.py --data DATA --work WORK
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SYSTEM = platform.system()


def _utc() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _alive(pid: int) -> bool:
    if SYSTEM == "Windows":  # os.kill would terminate the process on Windows
        import ctypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ctypes.get_last_error() == 5  # access denied: it runs, as another user
        code = ctypes.c_ulong()
        k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Chain:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.logs = args.logs
        self.logs.mkdir(parents=True, exist_ok=True)
        self.failed: list[str] = []

    def note(self, text: str) -> None:
        line = f"{_utc()} {text}"
        print(line, flush=True)
        with open(self.logs / "chain.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def record_command(self, cmd: list[str]) -> None:
        with open(self.logs / "commands.txt", "a", encoding="utf-8") as f:
            f.write(f"{_utc()} {shlex.join(cmd)}\n")

    def clear_stale_lock(self) -> None:
        """A step that was killed leaves ``<work>/.run.lock``; remove it if its process is gone."""
        lock = self.args.work / ".run.lock"
        if not lock.exists():
            return
        try:
            pid = int(lock.read_text().split()[0])
        except (OSError, ValueError, IndexError):
            self.note(f"{lock} exists but names no process; delete it if no run is active")
            return
        if not _alive(pid):
            self.note(f"removing the lock of process {pid}, which no longer runs")
            lock.unlink(missing_ok=True)

    def step(self, name: str, argv: list[str]) -> None:
        cmd = [sys.executable, "-m", "benchmarks.accuracy", *argv]
        self.clear_stale_lock()
        self.record_command(cmd)
        self.note(f"start {name}: {shlex.join(argv)}")
        with open(self.logs / f"{name}.log", "a", encoding="utf-8") as log:
            rc = subprocess.call(
                cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=self.env()
            )
        self.note(f"end   {name}: exit {rc}")
        if rc:
            self.failed.append(name)

    @staticmethod
    def env() -> dict[str, str]:
        return dict(os.environ, PYTHONUNBUFFERED="1")


def keep_awake(pid: int) -> subprocess.Popen | None:
    """Keep the machine from sleeping while process ``pid`` runs, where no admin is needed."""
    if SYSTEM == "Darwin" and shutil.which("caffeinate"):
        return subprocess.Popen(["caffeinate", "-dimsu", "-w", str(pid)])
    if SYSTEM == "Linux" and shutil.which("systemd-inhibit") and shutil.which("tail"):
        try:
            return subprocess.Popen(
                [
                    "systemd-inhibit",
                    "--what=sleep:idle",
                    "--who=FastWarp benchmark",
                    "--why=benchmark running",
                    "tail",
                    f"--pid={pid}",
                    "-f",
                    os.devnull,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            return None
    if SYSTEM == "Windows":
        import ctypes

        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: holds while this thread lives
        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001):
            raise OSError("SetThreadExecutionState refused")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chain.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True, help="data folder (outside the repo)")
    parser.add_argument("--work", type=Path, required=True, help="work folder (outside the repo)")
    parser.add_argument("--logs", type=Path, help="log folder (default: <work>/../logs)")
    parser.add_argument("--monitor", type=Path, help="monitor folder (default: <work>/../monitor)")
    parser.add_argument("--no-monitor", action="store_true", help="do not record the load")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between samples")
    parser.add_argument("--no-fetch", action="store_true", help="skip the fetch step")
    parser.add_argument("--dataset", action="append", help="dataset key (repeatable)")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--pairs", type=int, help="number of pairs per dataset")
    g.add_argument("--quick", action="store_true", help="one pair per dataset")
    args = parser.parse_args(argv)
    args.data, args.work = args.data.resolve(), args.work.resolve()
    args.logs = (args.logs or args.work.parent / "logs").resolve()
    args.monitor = (args.monitor or args.work.parent / "monitor").resolve()

    sel: list[str] = []
    for key in args.dataset or []:
        sel += ["--dataset", key]
    if args.pairs:
        sel += ["--pairs", str(args.pairs)]
    if args.quick:
        sel.append("--quick")
    folders = ["--data", str(args.data), "--work", str(args.work)]

    chain = Chain(args)
    chain.note(f"chain starts: {platform.platform()}, Python {platform.python_version()}")
    try:
        awake = keep_awake(os.getpid())
    except OSError as e:
        awake = None
        chain.note(f"cannot keep the machine awake: {e}")
    if awake is not None:
        time.sleep(1.0)
        if awake.poll() is not None:
            chain.note("cannot keep the machine awake: the inhibitor exited (no permission?)")
    monitor = None
    if not args.no_monitor:
        cmd = [
            sys.executable,
            str(HERE / "monitor.py"),
            "sample",
            "--out",
            str(args.monitor),
            "--bench-pid",
            str(os.getpid()),
            "--interval",
            str(args.interval),
        ]
        chain.record_command(cmd)
        with open(args.logs / "monitor.log", "a", encoding="utf-8") as log:
            monitor = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        chain.note(f"monitor started (process {monitor.pid})")
        time.sleep(2 * args.interval)  # a few samples before the first step

    def check_monitor() -> None:
        if monitor is not None and monitor.poll() is not None and "monitor" not in chain.failed:
            chain.note(f"the monitor stopped with exit {monitor.returncode}; see monitor.log")
            chain.failed.append("monitor")

    steps = []
    if not args.no_fetch:
        fetch = ["fetch", "--data", str(args.data)]
        for key in args.dataset or []:
            fetch += ["--dataset", key]
        steps.append(("fetch", fetch))
    steps += [(s, [s, *sel, *folders]) for s in ("run", "linear", "post", "evaluate")]
    steps.append(("report", ["report", "--work", str(args.work)]))
    try:
        for name, step_argv in steps:
            check_monitor()
            chain.step(name, step_argv)
        check_monitor()
    finally:
        if monitor is not None:
            time.sleep(args.interval)
            monitor.terminate()
            try:
                monitor.wait(timeout=30)
            except subprocess.TimeoutExpired:
                monitor.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    monitor.wait(timeout=10)
            cmd = [
                sys.executable,
                str(HERE / "monitor.py"),
                "records",
                "--work",
                str(args.work),
                "--monitor",
                str(args.monitor),
                "--interval",
                str(args.interval),
            ]
            chain.record_command(cmd)
            out = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
            chain.note((out.stdout + out.stderr).strip())
            if out.returncode:
                chain.failed.append("records")
        if awake is not None:
            awake.terminate()
    chain.note("chain done" + (f"; failed: {', '.join(chain.failed)}" if chain.failed else ""))
    return 1 if chain.failed else 0


if __name__ == "__main__":
    sys.exit(main())
