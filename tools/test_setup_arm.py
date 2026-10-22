"""Tests for setup.py on a 64-bit Arm machine (Jetson, DGX Spark): the CPU is read from /proc/cpuinfo without an x86
"flags" line, it gets no older-x86 floor, a Jetson's CUDA Toolkit is JetPack's and an Arm server's comes from NVIDIA's
"sbsa" repository, and compile jobs fit the RAM. No compiler, no GPU.

    python -m unittest tools.test_setup_arm
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from strata_control import nvidia_uma  # noqa: E402

ORIN_CPUINFO = ("processor\t: 0\nBogoMIPS\t: 62.50\nFeatures\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp "
                "asimdhp cpuid asimdrdm lrcpc dcpop asimddp\nCPU implementer\t: 0x41\nCPU part\t: 0xd42\n")


def cpuinfo(text: str):
    return mock.patch("builtins.open", lambda *a, **k: io.StringIO(text))


class ArmCpu(unittest.TestCase):
    def test_an_arm_cpuinfo_is_read_without_flags(self):
        with mock.patch.object(setup, "WIN", False), mock.patch.object(setup, "arm_host", lambda: True), \
                mock.patch.object(setup, "arm_cpu_name", lambda: "Cortex-A78AE"), cpuinfo(ORIN_CPUINFO):
            self.assertEqual(setup.cpu_info(), ("Cortex-A78AE", False, False))

    def test_an_x86_cpuinfo_still_reads_its_flags(self):
        text = "model name\t: Some Ryzen\nflags\t\t: fpu sse4_2 avx avx2 fma f16c\n"
        with mock.patch.object(setup, "WIN", False), mock.patch.object(setup, "arm_host", lambda: False), cpuinfo(text):
            self.assertEqual(setup.cpu_info(), ("Some Ryzen", True, False))

    def test_arm_has_no_x86_floor(self):
        with mock.patch.dict(os.environ, {"STRATA_ISA_FLOOR": ""}), mock.patch.object(setup, "arm_host", lambda: True):
            self.assertEqual(setup.cpu_floor(False), "")

    def test_the_name_comes_from_lscpu(self):
        with mock.patch.object(setup, "arm_host", lambda: True), \
                mock.patch.object(setup, "out", lambda cmd: "Architecture:  aarch64\nModel name:    Cortex-A78AE\n"):
            self.assertEqual(setup.arm_cpu_name(), "Cortex-A78AE")
        with mock.patch.object(setup, "arm_host", lambda: False):
            self.assertEqual(setup.arm_cpu_name(), "")


class BuildJobs(unittest.TestCase):
    def jobs(self, gib: float, cpus: int = 12) -> int:
        with tempfile.NamedTemporaryFile("w", suffix=".meminfo", delete=False) as f:
            f.write(f"MemTotal: 65000000 kB\nMemAvailable: {int(gib * 1048576)} kB\n")
        self.addCleanup(os.unlink, f.name)
        with mock.patch.object(setup, "WIN", False), mock.patch.object(setup.os, "cpu_count", lambda: cpus):
            return setup.build_jobs(f.name)

    def test_half_the_cpus_when_the_ram_holds_them(self):
        self.assertEqual(self.jobs(50), 6)

    def test_two_gib_a_job_beyond_two_aside(self):
        self.assertEqual(self.jobs(8), 3)
        self.assertEqual(self.jobs(3), 1)
        self.assertEqual(self.jobs(0.5), 1)

    def test_an_unreadable_meminfo_keeps_the_old_count(self):
        with mock.patch.object(setup, "WIN", False), mock.patch.object(setup.os, "cpu_count", lambda: 12):
            self.assertEqual(setup.build_jobs("/nonexistent/meminfo"), 6)


class Toolkit(unittest.TestCase):
    """No nvcc: what install_build_tools does about it on Arm."""

    def run_install(self, tegra: str | None):
        said, fetched = io.StringIO(), []

        def download(url, *a, **k):
            fetched.append(url)
            raise RuntimeError("stop here")

        with mock.patch.object(setup, "WIN", False), mock.patch.object(setup, "arm_host", lambda: True), \
                mock.patch.object(setup, "find_nvcc", lambda below=None: (None, (0, 0))), \
                mock.patch.object(setup.shutil, "which", lambda name: f"/usr/bin/{name}"), \
                mock.patch.object(setup.acquisition, "require", lambda what: None), \
                mock.patch.object(nvidia_uma, "tegra_soc", lambda: tegra), \
                mock.patch.object(setup, "cuda_apt_repo", lambda osr: "ubuntu2404"), \
                mock.patch.object(setup, "download", download), mock.patch.object(setup, "say", lambda *a: said.write(" ".join(map(str, a)) + "\n")):
            try:
                setup.install_build_tools({"arch": "87", "archs": ["87"]}, True)
            except (SystemExit, RuntimeError):
                pass
        return said.getvalue(), fetched

    def test_a_jetson_is_sent_to_jetpack(self):
        said, fetched = self.run_install("tegra234")
        self.assertIn("comes with JetPack", said)
        self.assertIn("nvidia-jetpack", said)
        self.assertEqual(fetched, [])                               # never NVIDIA's desktop repository

    def test_an_arm_server_gets_the_sbsa_repository(self):
        said, fetched = self.run_install(None)
        self.assertEqual(len(fetched), 1)
        self.assertIn("/ubuntu2404/sbsa/cuda-keyring", fetched[0])



class InPlacePack(unittest.TestCase):
    """An NVIDIA GPU that shares the RAM gets the pack's experts.bin when the RAM holds it with 20 GiB to spare: the engine
    then reads every expert in place (--uma-experts) instead of filling a cache. A smaller board or a graphics card does not."""

    ORIN = {"index": 0, "name": "Orin (nvgpu)", "vram_gb": 61.4, "unified_gb": 61.4, "dedicated_gb": 0.0,
            "integrated": True, "uma": True, "arch": "87", "driver": "595.78"}

    def packs(self, ram, gpu):
        import test_setup_golden as golden
        ran = []
        code, text, _cfg, _asked = golden.install(ram, [gpu], golden.argv_for("qwen", "Q2_0"),
                                                  extra=[mock.patch.object(setup, "run", lambda cmd, *a, **k: ran.append(cmd))])
        self.assertEqual(code, 0, text[-2000:])
        return [c for c in ran if any(str(x).endswith("iq_pack.py") for x in c)], text

    def test_a_64_gb_orin_writes_experts_bin(self):
        packs, text = self.packs(61.4, self.ORIN)
        self.assertTrue(any("--experts-bin" in c for c in packs), packs)
        self.assertIn("so the GPU, which shares the RAM, reads them where they lie", text)

    def test_a_32_gb_orin_does_not(self):
        # (its RAM is under Q2_0's, so the low-RAM mode writes experts.bin for its own reason)
        _, text = self.packs(30.6, {**self.ORIN, "vram_gb": 30.6, "unified_gb": 30.6})
        self.assertNotIn("reads them where they lie", text)

    def test_a_graphics_card_does_not(self):
        packs, _ = self.packs(64.0, {"index": 0, "name": "NVIDIA GeForce RTX 4090", "vram_gb": 24.0, "arch": "89",
                                     "driver": "595.78"})
        self.assertFalse(any("--experts-bin" in c for c in packs), packs)


if __name__ == "__main__":
    unittest.main()
