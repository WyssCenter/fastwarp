# SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
# SPDX-License-Identifier: Apache-2.0

"""Record the machine's load while the benchmark runs (Linux, macOS, Windows; standard library).

Two commands:

``sample``
    Every ``--interval`` seconds, append one row to ``<out>/samples.csv``: the 1-minute load
    average (not on Windows), the CPU use of the benchmark's process tree (the process
    ``--bench-pid`` and its descendants) and of all other processes, the GPU's utilisation and
    memory where the platform reports them without admin rights (``nvidia-smi`` for NVIDIA,
    ``ioreg`` on macOS), whether the machine runs on mains power, and the five other processes
    that used the most CPU. It runs until it is stopped, or until the benchmark process ends,
    and appends the logged-in users to ``<out>/users_at_start.txt`` when it starts.

``records``
    For each ``run.json`` and ``post.json`` under ``--work`` (registrations, linear fits,
    post-processing) with ``started`` and ``finished`` times, summarise the samples taken while it
    ran (within half an interval; the nearest sample for a shorter record) into
    ``<monitor>/per_record.csv``. ``other_load`` lists what may have slowed it: ``cpu`` (other
    processes above ``--cpu-share`` of the cores on average), ``gpu`` (other GPU work above
    ``--gpu-pct`` on average, or any GPU use during an ANTs record where the platform does not
    split it by process), ``battery``; ``no samples`` when no sample falls in the record.

``benchmarks/scripts/chain.py`` starts and stops the monitor and writes ``per_record.csv``;
run it by hand as::

    python benchmarks/scripts/monitor.py sample --out MONITOR --bench-pid PID
    python benchmarks/scripts/monitor.py records --work WORK --monitor MONITOR
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

SYSTEM = platform.system()
NCORES = os.cpu_count() or 1
FIELDS = [
    "utc",
    "load1",
    "ncores",
    "cpu_total_pct",
    "bench_cpu_pct",
    "other_cpu_pct",
    "other_cpu_cores",
    "gpu_util_pct",
    "gpu_mem_used_mb",
    "gpu_mem_total_mb",
    "gpu_bench_sm_pct",
    "gpu_other_sm_pct",
    "gpu_other_mem_mb",
    "ac_online",
    "bench_pids",
    "top5_other",
]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, errors="replace", timeout=20
        ).stdout
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""


def _utc() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _descendants(parents: dict[int, int], root: int | None) -> set[int]:
    if root is None or root not in parents:
        return set()
    kids: dict[int, list[int]] = {}
    for pid, ppid in parents.items():
        if ppid != pid:
            kids.setdefault(ppid, []).append(pid)
    seen: set[int] = set()
    stack = [root]
    while stack:
        p = stack.pop()
        if p not in seen:
            seen.add(p)
            stack.extend(kids.get(p, []))
    return seen


# --- processes and CPU, one class per platform ------------------------------------------------
# Each sampler's sample() returns (processes, cpu_total_pct): processes maps a PID to
# (parent PID, user, name, %CPU of one core over the last interval).


class LinuxCpu:
    """CPU times from /proc, so that the percentages cover the interval (ps gives lifetimes)."""

    def __init__(self) -> None:
        self.tick = os.sysconf("SC_CLK_TCK")
        self.prev: dict[int, int] = {}
        self.prev_total: tuple[float, int, int] | None = None
        self.users: dict[int, str] = {}

    def _user(self, uid: int) -> str:
        if uid not in self.users:
            try:
                import pwd

                self.users[uid] = pwd.getpwuid(uid).pw_name
            except (ImportError, KeyError):
                self.users[uid] = str(uid)
        return self.users[uid]

    def sample(self) -> tuple[dict[int, tuple[int, str, str, float]], float | str]:
        now = time.monotonic()
        info: dict[int, tuple[int, str, str]] = {}
        ticks: dict[int, int] = {}
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as f:
                    stat = f.read()
                uid = entry.stat().st_uid
            except OSError:
                continue
            name = stat[stat.index("(") + 1 : stat.rindex(")")]
            fields = stat[stat.rindex(")") + 2 :].split()
            info[pid] = (int(fields[1]), self._user(uid), name)
            ticks[pid] = int(fields[11]) + int(fields[12])
        with open("/proc/stat") as f:
            cpu = [int(x) for x in f.readline().split()[1:]]
        total, idle = sum(cpu), cpu[3] + cpu[4]
        cpu_total: float | str = ""
        procs = {}
        if self.prev_total is not None:
            t0, total0, idle0 = self.prev_total
            if total > total0:
                cpu_total = round(100.0 * (1 - (idle - idle0) / (total - total0)), 1)
            wall = max(now - t0, 1e-6)
            for pid, (ppid, user, name) in info.items():
                d = ticks[pid] - self.prev.get(pid, ticks[pid])
                procs[pid] = (ppid, user, name, 100.0 * d / self.tick / wall)
        else:
            procs = {pid: (*v, 0.0) for pid, v in info.items()}
        self.prev, self.prev_total = ticks, (now, total, idle)
        return procs, cpu_total


class MacCpu:
    """ps on macOS reports a recent, decaying %CPU per process."""

    def sample(self) -> tuple[dict[int, tuple[int, str, str, float]], float | str]:
        procs = {}
        for line in _run(["ps", "-A", "-o", "pid=,ppid=,user=,%cpu=,comm="]).splitlines():
            parts = line.split(None, 4)
            if len(parts) < 5:
                continue
            try:
                pid, ppid, pct = int(parts[0]), int(parts[1]), float(parts[3].replace(",", "."))
            except ValueError:
                continue
            procs[pid] = (ppid, parts[2], os.path.basename(parts[4].strip()), pct)
        total = round(min(100.0, sum(p[3] for p in procs.values()) / NCORES), 1)
        return procs, total


class WindowsCpu:
    """Toolhelp snapshot and GetProcessTimes; works without admin rights.

    Other users' processes cannot be opened without admin rights, so their CPU use is not seen
    one by one: ``other`` is then the machine's total CPU minus the benchmark's.
    """

    def __init__(self) -> None:
        import ctypes
        import ctypes.wintypes as wt

        class Entry(ctypes.Structure):
            _fields_ = [
                ("dwSize", wt.DWORD),
                ("cntUsage", wt.DWORD),
                ("th32ProcessID", wt.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wt.DWORD),
                ("cntThreads", wt.DWORD),
                ("th32ParentProcessID", wt.DWORD),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wt.DWORD),
                ("szExeFile", ctypes.c_wchar * 260),
            ]

        self.ct, self.wt, self.Entry = ctypes, wt, Entry
        self.k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
        self.k32.OpenProcess.restype = wt.HANDLE
        self.prev: dict[int, tuple[int, int]] = {}
        self.prev_sys = self._sys_times()
        self.prev_t = time.monotonic()

    @staticmethod
    def _ft(x) -> int:
        return (x.dwHighDateTime << 32) | x.dwLowDateTime

    def _sys_times(self) -> tuple[int, int]:
        ft = self.wt.FILETIME
        idle, kernel, user = ft(), ft(), ft()
        self.k32.GetSystemTimes(*(self.ct.byref(x) for x in (idle, kernel, user)))
        return self._ft(idle), self._ft(kernel) + self._ft(user)  # kernel includes idle

    def _proc_time(self, pid: int) -> tuple[int, int] | None:
        h = self.k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return None
        ft = self.wt.FILETIME
        created, ended, kernel, user = ft(), ft(), ft(), ft()
        ok = self.k32.GetProcessTimes(
            h, *(self.ct.byref(x) for x in (created, ended, kernel, user))
        )
        self.k32.CloseHandle(h)
        return (self._ft(kernel) + self._ft(user), self._ft(created)) if ok else None

    def sample(self) -> tuple[dict[int, tuple[int, str, str, float]], float | str]:
        h = self.k32.CreateToolhelp32Snapshot(2, 0)  # TH32CS_SNAPPROCESS
        entry = self.Entry()
        entry.dwSize = self.ct.sizeof(self.Entry)
        names: dict[int, tuple[int, str]] = {}
        ok = self.k32.Process32FirstW(h, self.ct.byref(entry))
        while ok:
            names[entry.th32ProcessID] = (entry.th32ParentProcessID, entry.szExeFile)
            ok = self.k32.Process32NextW(h, self.ct.byref(entry))
        self.k32.CloseHandle(h)
        cur = {pid: t for pid in names if pid and (t := self._proc_time(pid)) is not None}
        sys_now, now = self._sys_times(), time.monotonic()
        d_idle, d_total = sys_now[0] - self.prev_sys[0], sys_now[1] - self.prev_sys[1]
        cpu_total: float | str = round(100.0 * (1 - d_idle / d_total), 1) if d_total > 0 else ""
        wall = max(now - self.prev_t, 1e-6) * 1e7  # FILETIME counts 100 ns
        procs = {}
        for pid, (ppid, name) in names.items():
            pct = -1.0  # not readable without admin rights
            if pid in cur and pid in self.prev and self.prev[pid][1] == cur[pid][1]:
                pct = 100.0 * (cur[pid][0] - self.prev[pid][0]) / wall
            procs[pid] = (ppid, "?", name, pct)
        self.prev, self.prev_sys, self.prev_t = cur, sys_now, now
        return procs, cpu_total


# --- GPU and power ------------------------------------------------------------------------------


def gpu_nvidia(nvsmi: str, bench: set[int]) -> list:
    """Utilisation and memory of the first GPU; SM use and memory per process over all GPUs.

    The SM use is blank where ``pmon`` reports nothing (for example a Windows driver in WDDM
    mode), so that "unknown" is not read as "no other GPU work"."""
    util = mem = total = ""
    out = _run(
        [
            nvsmi,
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    if out and out[0].count(",") == 2:
        util, mem, total = (x.strip() for x in out[0].split(","))
    pmon = _run([nvsmi, "pmon", "-c", "1", "-s", "u"])
    bench_sm = other_sm = 0.0
    for line in pmon.splitlines():
        p = line.split()
        if line.startswith("#") or len(p) < 4 or not p[1].isdigit():
            continue
        sm = float(p[3]) if p[3].replace(".", "").isdigit() else 0.0
        if int(p[1]) in bench:
            bench_sm += sm
        else:
            other_sm += sm
    other_mem = 0.0
    for line in _run(
        [nvsmi, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
    ).splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 2 and p[0].isdigit() and int(p[0]) not in bench:
            with contextlib.suppress(ValueError):
                other_mem += float(p[1])
    if "# gpu" not in pmon:
        return [util, mem, total, "", "", other_mem]
    return [util, mem, total, bench_sm, other_sm, other_mem]


def gpu_mac() -> list:
    """Utilisation and memory of the first IOAccelerator that reports free VRAM (a discrete GPU),
    else of the first one (Apple silicon). macOS does not split it by process."""
    out = _run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"])
    first = None
    for block in out.split('"PerformanceStatistics" = {')[1:]:
        vals = {}
        for item in block.split("}", 1)[0].split(","):
            k, _, v = item.partition("=")
            vals[k.strip().strip('"')] = v.strip()
        first = first or vals
        if "vramFreeBytes" in vals:
            try:
                used = int(vals.get("inUseVidMemoryBytes", ""))
                mem, total = (
                    str(round(used / 2**20)),
                    str(round((used + int(vals["vramFreeBytes"])) / 2**20)),
                )
            except ValueError:
                mem = total = ""
            return [vals.get("Device Utilization %", ""), mem, total, "", "", ""]
    return [(first or {}).get("Device Utilization %", ""), "", "", "", "", ""]


def ac_online() -> str:
    """1 on mains power, 0 on battery, blank when the machine does not say (desktops)."""
    if SYSTEM == "Darwin":
        out = _run(["pmset", "-g", "batt"])
        return "1" if "AC Power" in out else ("0" if "Battery Power" in out else "")
    if SYSTEM == "Windows":
        import ctypes
        import ctypes.wintypes as wt

        class Status(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_ubyte),
                ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte),
                ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", wt.DWORD),
                ("BatteryFullLifeTime", wt.DWORD),
            ]

        s = Status()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return ""
        return {0: "0", 1: "1"}.get(s.ACLineStatus, "")
    has_battery, online = False, []
    for d in Path("/sys/class/power_supply").glob("*"):
        try:
            if (d / "scope").exists() and (d / "scope").read_text().strip() == "Device":
                continue  # a wireless mouse's or keyboard's own battery
            if (d / "type").read_text().strip() == "Battery":
                has_battery = True
            else:
                online.append((d / "online").read_text().strip())
        except OSError:
            continue
    if not has_battery or not online:
        return ""
    return "1" if "1" in online else "0"


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


def cmd_sample(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "samples.csv"
    new = not path.exists() or path.stat().st_size == 0
    users = _run(["query", "user"] if SYSTEM == "Windows" else ["who"])
    with open(out / "users_at_start.txt", "a", encoding="utf-8") as f:
        f.write(f"{_utc()}\n{users}\n")
    cpu = {"Linux": LinuxCpu, "Darwin": MacCpu, "Windows": WindowsCpu}.get(SYSTEM, MacCpu)()
    nvsmi = shutil.which("nvidia-smi")
    me = os.getpid()
    cpu.sample()
    time.sleep(1.0)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        while args.bench_pid is None or _alive(args.bench_pid):
            t0 = time.monotonic()
            procs, cpu_total = cpu.sample()
            parents = {pid: p[0] for pid, p in procs.items()}
            mine = _descendants(parents, me)
            bench = _descendants(parents, args.bench_pid) - mine
            bench_pct = sum(procs[p][3] for p in bench if procs[p][3] > 0)
            others = [p for pid, p in procs.items() if pid not in bench and pid not in mine]
            other_pct = sum(p[3] for p in others if p[3] > 0)
            if SYSTEM == "Windows" and cpu_total != "":
                other_pct = max(0.0, cpu_total * NCORES - bench_pct)
            others.sort(key=lambda p: p[3], reverse=True)
            top5 = ";".join(f"{p[2]}:{p[1]}:{p[3]:.0f}" for p in others[:5] if p[3] >= 0)
            if nvsmi:
                gpu = gpu_nvidia(nvsmi, bench)
            elif SYSTEM == "Darwin":
                gpu = gpu_mac()
            else:
                gpu = [""] * 6
            load1 = round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else ""
            w.writerow(
                [
                    _utc(),
                    load1,
                    NCORES,
                    cpu_total,
                    round(bench_pct / NCORES, 1),
                    round(other_pct / NCORES, 1),
                    round(other_pct / 100.0, 2),
                    *gpu,
                    ac_online(),
                    len(bench),
                    top5,
                ]
            )
            f.flush()
            time.sleep(max(0.5, args.interval - (time.monotonic() - t0)))
    return 0


# --- records ------------------------------------------------------------------------------------


def _ts(s: str) -> float:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _num(x: str | None) -> float | None:
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _stats(rows: list[dict], key: str) -> tuple:
    v = [x for x in (_num(r.get(key)) for r in rows) if x is not None]
    return (round(sum(v) / len(v), 2), round(max(v), 2)) if v else ("", "")


def cmd_records(args: argparse.Namespace) -> int:
    work, mon = Path(args.work), Path(args.monitor)
    samples, bad = [], 0
    with open(mon / "samples.csv", newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            try:  # a run cut off by a power loss can leave a broken last line
                samples.append(dict(r, t=_ts(r["utc"])))
            except (TypeError, ValueError, AttributeError):
                bad += 1
    half = args.interval / 2
    out, unfinished = [], []
    for path in sorted(work.rglob("*.json")):
        if path.name not in ("run.json", "post.json"):
            continue
        rel = path.relative_to(work).parts
        try:
            d = json.loads(path.read_text("utf-8"))
            t0, t1 = _ts(d["started"]), _ts(d["finished"])
        except (KeyError, ValueError, OSError, AttributeError, TypeError):
            unfinished.append("/".join(rel))  # a failed, skipped or timed-out record
            continue
        kind = "post" if path.name == "post.json" else ("linear" if "linear" in rel else "run")
        engine = d.get("engine") or (rel[-2] if kind != "post" else "both")
        rows = [r for r in samples if t0 - half <= r["t"] <= t1 + half]
        if not rows and samples:  # a record shorter than the interval: the nearest sample
            mid = (t0 + t1) / 2
            near = min(samples, key=lambda r: abs(r["t"] - mid))
            rows = [near] if abs(near["t"] - mid) <= 6 * args.interval else []
        ncores = _num(rows[0]["ncores"]) if rows else None
        other_mean, other_max = _stats(rows, "other_cpu_cores")
        gpu_mean, gpu_max = _stats(rows, "gpu_util_pct")
        gpu_other_mean, gpu_other_max = _stats(rows, "gpu_other_sm_pct")
        load_mean, load_max = _stats(rows, "load1")
        cpu_mean, cpu_max = _stats(rows, "cpu_total_pct")
        battery = any(r["ac_online"] == "0" for r in rows)
        reasons = []
        if ncores and other_mean != "" and other_mean > args.cpu_share * ncores:
            reasons.append("cpu")
        if gpu_other_mean != "" and gpu_other_mean > args.gpu_pct:
            reasons.append("gpu")
        elif (
            engine == "ants" and gpu_other_mean == "" and gpu_mean != "" and gpu_mean > args.gpu_pct
        ):
            reasons.append("gpu")  # without per-process GPU use: during ANTs, all of it is others'
        if battery:
            reasons.append("battery")
        out.append(
            {
                "record": "/".join(rel),
                "kind": kind,
                "engine": engine,
                "status": d.get("status", ""),
                "started": d["started"],
                "finished": d["finished"],
                "seconds": round(t1 - t0, 1),
                "n_samples": len(rows),
                "ncores": int(ncores) if ncores else "",
                "load1_mean": load_mean,
                "load1_max": load_max,
                "cpu_total_pct_mean": cpu_mean,
                "cpu_total_pct_max": cpu_max,
                "other_cpu_cores_mean": other_mean,
                "other_cpu_cores_max": other_max,
                "gpu_util_pct_mean": gpu_mean,
                "gpu_util_pct_max": gpu_max,
                "gpu_other_sm_pct_mean": gpu_other_mean,
                "gpu_other_sm_pct_max": gpu_other_max,
                "on_battery": "yes" if battery else "no",
                "other_load": "+".join(reasons) if rows else "no samples",
            }
        )
    with open(mon / "per_record.csv", "w", newline="", encoding="utf-8") as f:
        if out:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
    flagged = sum(1 for r in out if r["other_load"] not in ("", "no samples"))
    unsampled = sum(1 for r in out if r["other_load"] == "no samples")
    print(
        f"{len(out)} records, {flagged} with other load, {unsampled} without samples -> "
        f"{mon / 'per_record.csv'}"
    )
    if unfinished:
        print(
            f"{len(unfinished)} records without start and finish times left out, for example "
            f"{unfinished[0]}"
        )
    if bad:
        print(f"{bad} unreadable rows of samples.csv left out")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="monitor.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample", help="append load samples to <out>/samples.csv")
    p.add_argument("--out", type=Path, required=True, help="folder for samples.csv")
    p.add_argument("--bench-pid", type=int, help="PID of the benchmark; stop when it ends")
    p.add_argument("--interval", type=float, default=5.0, help="seconds between samples")
    p.set_defaults(func=cmd_sample)
    p = sub.add_parser("records", help="write <monitor>/per_record.csv")
    p.add_argument("--work", type=Path, required=True, help="the benchmark's work folder")
    p.add_argument("--monitor", type=Path, required=True, help="folder with samples.csv")
    p.add_argument("--interval", type=float, default=5.0, help="the sampling interval used")
    p.add_argument("--cpu-share", type=float, default=0.10, help="share of the cores (0.10)")
    p.add_argument("--gpu-pct", type=float, default=10.0, help="GPU utilisation in %% (10)")
    p.set_defaults(func=cmd_records)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
