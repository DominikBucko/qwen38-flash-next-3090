"""GPU-free checks for runtime/vllm-overlay/qwen38_host.py on synthetic sysfs, /proc and cgroup trees."""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("qwen38_host", ROOT / "runtime/vllm-overlay/qwen38_host.py")
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)
GIB = 1 << 30


def fake_sys(root: Path, cores: list[tuple[list[int], str]]) -> None:
    """cores: [(hardware threads of one physical core, L3 shared_cpu_list)]."""
    for threads, l3 in cores:
        siblings = ",".join(map(str, threads))
        for cpu in threads:
            base = root / f"devices/system/cpu/cpu{cpu}"
            (base / "topology").mkdir(parents=True)
            (base / "topology/physical_package_id").write_text("0\n")
            (base / "topology/thread_siblings_list").write_text(siblings + "\n")
            for index, level in enumerate(("1", "1", "2", "3")):
                cache = base / f"cache/index{index}"
                cache.mkdir(parents=True)
                (cache / "level").write_text(level + "\n")
                (cache / "shared_cpu_list").write_text((l3 if level == "3" else siblings) + "\n")


def threadripper_5975wx() -> list[tuple[list[int], str]]:
    # 4 CCDs x 8 cores, SMT siblings n and n + 32
    return [([c, c + 32], f"{c // 8 * 8}-{c // 8 * 8 + 7},{c // 8 * 8 + 32}-{c // 8 * 8 + 39}") for c in range(32)]


def ryzen_7950x() -> list[tuple[list[int], str]]:
    # 2 CCDs x 8 cores, SMT siblings n and n + 16
    return [([c, c + 16], f"{c // 8 * 8}-{c // 8 * 8 + 7},{c // 8 * 8 + 16}-{c // 8 * 8 + 23}") for c in range(16)]


def core_i9_13900k() -> list[tuple[list[int], str]]:
    # 8 P-cores with HT (0,1), (2,3), ... then 16 E-cores 16-31; one shared L3
    return [([2 * p, 2 * p + 1], "0-31") for p in range(8)] + [([e], "0-31") for e in range(16, 32)]


def ryzen_7800x3d() -> list[tuple[list[int], str]]:
    return [([c, c + 8], "0-15") for c in range(8)]


class HostPlanTest(unittest.TestCase):
    def plan(self, cores, env=None):
        allowed = sorted(cpu for threads, _ in cores for cpu in threads)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, env or {}, clear=False), \
                mock.patch.object(host, "_allowed", return_value=allowed):
            for name in ("QWEN38_CPU_EXPERTS_CPUS", "QWEN38_MAIN_CPUS", "QWEN38_CPU_EXPERTS_MAX_THREADS"):
                if name not in (env or {}):
                    os.environ.pop(name, None)
            fake_sys(Path(tmp), cores)
            pool = host.pool_cpus(sys_root=tmp)
            return host.format_cpus(pool), host.format_cpus(host.main_cpus(pool, sys_root=tmp))

    def test_benchmark_host_matches_measured_layout(self):
        # The layout measured best on the benchmark host: 6 of 8 cores per CCD, serving processes everywhere
        # except the pool cores' SMT siblings.
        self.assertEqual(self.plan(threadripper_5975wx()),
                         ("2-7,10-15,18-23,26-31", "0-33,40-41,48-49,56-57"))

    def test_two_ccd_desktop(self):
        self.assertEqual(self.plan(ryzen_7950x()), ("2-7,10-15", "0-17,24-25"))

    def test_single_ccd_desktop(self):
        self.assertEqual(self.plan(ryzen_7800x3d()), ("2-7", "0-9"))

    def test_intel_hybrid_keeps_p_cores_for_serving(self):
        pool, main = self.plan(core_i9_13900k())
        # 24 physical cores, one L3: the first 6 (P-cores) stay with the serving processes
        self.assertEqual(pool, "12,14,16-31")
        self.assertEqual(main, "0-12,14,16-31")

    def test_thread_cap_spreads_over_domains(self):
        pool, _ = self.plan(threadripper_5975wx(), {"QWEN38_CPU_EXPERTS_MAX_THREADS": "8"})
        self.assertEqual(pool, "2-3,10-11,18-19,26-27")

    def test_explicit_lists_win(self):
        self.assertEqual(self.plan(ryzen_7950x(), {"QWEN38_CPU_EXPERTS_CPUS": "8-15", "QWEN38_MAIN_CPUS": "0-7"}),
                         ("8-15", "0-7"))

    def test_tiny_host(self):
        cores = [([0, 2], "0-3"), ([1, 3], "0-3")]
        self.assertEqual(self.plan(cores)[0], "1")


class MemoryPlanTest(unittest.TestCase):
    def trees(self, tmp: Path, mem_total_gib: float, limit: str | None):
        (tmp / "proc").mkdir()
        (tmp / "proc/meminfo").write_text(f"MemTotal:       {int(mem_total_gib * 2**20)} kB\nMemFree: 1 kB\n")
        (tmp / "cg").mkdir()
        if limit is not None:
            (tmp / "cg/memory.max").write_text(limit + "\n")
        return str(tmp / "proc"), str(tmp / "cg")

    def slots(self, mem_total_gib, limit, env=None, total=48 * 480):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, env or {}, clear=False):
            for name in ("QWEN38_EXPERT_ARENA_SLOTS", "QWEN38_HOST_RESERVE_GIB", "QWEN38_SERVE_OVERHEAD_GIB"):
                if name not in (env or {}):
                    os.environ.pop(name, None)
            proc, cg = self.trees(Path(tmp), mem_total_gib, limit)
            return host.arena_slots(total, 48, proc_root=proc, cgroup_root=cg)

    def test_benchmark_limit(self):
        # 56 GiB container limit -> 46 GiB arena, as in the benchmarks (19,500 slots)
        self.assertEqual(self.slots(125.7, str(56 * GIB)), 19480)

    def test_64gb_desktop_without_limit(self):
        # 62.5 GiB MemTotal -> 64 GiB installed - 8 GiB for the OS = 56 GiB budget -> 46 GiB arena
        self.assertEqual(self.slots(62.5, "max"), 19480)
        self.assertEqual(self.slots(64.0, None), 19480)

    def test_48gb_desktop(self):
        # 46.9 GiB MemTotal -> 48 - 8 = 40 GiB budget -> 30 GiB arena
        self.assertEqual(self.slots(46.9, None), int(30 * GIB // host.SLOT_BYTES))

    def test_large_host_keeps_every_cold_expert(self):
        self.assertEqual(self.slots(125.7, None), 48 * 480)

    def test_explicit_and_floor(self):
        self.assertEqual(self.slots(62.5, "max", {"QWEN38_EXPERT_ARENA_SLOTS": "12345"}), 12345)
        self.assertEqual(self.slots(15.5, "max"), 48)

    def test_cgroup_v1_no_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            cg = Path(tmp) / "memory"
            cg.mkdir()
            (cg / "memory.limit_in_bytes").write_text(str(2**63 - 4096))
            self.assertIsNone(host.cgroup_limit_bytes(tmp))


if __name__ == "__main__":
    unittest.main()
