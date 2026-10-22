"""strata_control.nvidia_uma: the NVIDIA GPUs that share the RAM (Jetson, DGX Spark) as the hardware readers see them.
Pure: every file is a throwaway, nvidia-smi is never run."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import nvidia_uma as uma

MEMINFO = "MemTotal:       64386068 kB\nMemFree:        50000000 kB\nMemAvailable:   55223232 kB\n"
ORIN_COMPATIBLE = b"nvidia,p3737-0000+p3701-0005\0nvidia,p3701-0005\0nvidia,tegra234\0"
ORIN_RELEASE = "# R39 (release), REVISION: 2.1, GCID: 46758480, BOARD: generic, EABI: aarch64, DATE: Fri Aug  7 2026\n"


class Files(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)

    def write(self, name: str, data) -> Path:
        path = self.root / name
        path.write_bytes(data if isinstance(data, bytes) else data.encode())
        return path

    def test_memory_is_read_from_meminfo(self):
        meminfo = self.write("meminfo", MEMINFO)
        self.assertAlmostEqual(uma.ram_gb(meminfo), 64386068 / 2**20, places=6)
        self.assertEqual(uma.used_mib(meminfo), (64386068 - 55223232) // 1024)
        self.assertEqual(uma.ram_gb(self.root / "missing"), 0.0)
        self.assertIsNone(uma.used_mib(self.write("partial", "MemTotal: 1024 kB\n")))

    def test_a_jetson_is_named_by_its_device_tree(self):
        self.assertEqual(uma.tegra_soc(self.write("compatible", ORIN_COMPATIBLE)), "tegra234")
        self.assertEqual(uma.board_name(self.write("model", b"NVIDIA Jetson AGX Orin Developer Kit\0")),
                         "NVIDIA Jetson AGX Orin Developer Kit")
        self.assertEqual(uma.l4t_release(self.write("nv_tegra_release", ORIN_RELEASE)), "R39.2.1")

    def test_other_machines_have_no_tegra_answers(self):
        self.assertIsNone(uma.tegra_soc(self.write("compatible", b"raspberrypi,5-model-b\0brcm,bcm2712\0")))
        self.assertIsNone(uma.tegra_soc(self.root / "missing"))
        self.assertIsNone(uma.board_name(self.root / "missing"))
        self.assertIsNone(uma.l4t_release(self.write("garbage", "not a release line\n")))


class MemoryFields(unittest.TestCase):
    def test_a_card_with_its_own_memory_keeps_it(self):
        self.assertEqual(uma.memory_fields("24564", arm64=True), {"vram_gb": 24564 / 1024})
        self.assertEqual(uma.memory_fields("12282", arm64=False), {"vram_gb": 12282 / 1024})

    def test_an_arm64_gpu_without_its_own_memory_shares_the_ram(self):
        with mock.patch.object(uma, "tegra_soc", return_value=None):          # a DGX Spark: no device tree
            fields = uma.memory_fields("[N/A]", arm64=True, total_gb=119.7)
        self.assertEqual(fields, {"vram_gb": 119.7, "unified_gb": 119.7, "dedicated_gb": 0.0, "integrated": True, "uma": True})

    def test_a_jetson_also_names_its_board(self):
        with mock.patch.object(uma, "tegra_soc", return_value="tegra234"), \
             mock.patch.object(uma, "board_name", return_value="NVIDIA Jetson AGX Orin Developer Kit"):
            fields = uma.memory_fields("[N/A]", arm64=True, total_gb=61.4)
        self.assertEqual(fields["board"], "NVIDIA Jetson AGX Orin Developer Kit")
        self.assertTrue(fields["uma"])

    def test_an_unreadable_memory_elsewhere_still_drops_the_row(self):
        self.assertIsNone(uma.memory_fields("[N/A]", arm64=False, total_gb=64.0))   # an x86 PC: as before
        self.assertIsNone(uma.memory_fields("[N/A]", arm64=True, total_gb=0.0))     # no RAM figure: nothing to give


if __name__ == "__main__":
    unittest.main()
