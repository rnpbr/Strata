"""NVIDIA GPUs that share the system memory: the Jetson boards (Tegra) and the DGX Spark (GB10).

nvidia-smi names these GPUs ("Orin (nvgpu)", "NVIDIA GB10") and gives their compute capability, but reports their memory as
[N/A]: they have no memory of their own and read the system RAM, all of it, about as fast as a discrete card reads its VRAM.
The hardware readers (setup.py, tools/strata_mcp.py, the doctor, the tuner, the benchmark facts) describe such a GPU the way
they describe a Mac or a Strix Halo: a unified-memory device the size of the RAM, with no memory beside it.

The rule is narrow on purpose: an NVIDIA GPU whose memory nvidia-smi cannot give, on an arm64 Linux host. An arm64 server
with a discrete card (Grace with an H100 on PCIe, an Altra with an RTX) reports its memory and is not affected.

Pure reads of /proc and /etc; nothing here starts a process.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

MEMINFO = Path("/proc/meminfo")
DT_COMPATIBLE = Path("/proc/device-tree/compatible")
DT_MODEL = Path("/proc/device-tree/model")
TEGRA_RELEASE = Path("/etc/nv_tegra_release")


def _read(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8", "replace")
    except OSError:
        return ""


def arm64_linux() -> bool:
    return platform.system() == "Linux" and platform.machine().lower() in ("aarch64", "arm64")


def ram_gb(meminfo: Path = MEMINFO) -> float:
    """MemTotal in GiB, 0.0 when it cannot be read."""
    for line in _read(meminfo).splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024 / 2**30
    return 0.0


def used_mib(meminfo: Path = MEMINFO) -> int | None:
    """Memory in use system-wide (MemTotal - MemAvailable), in MiB: what "GPU memory used" means when the GPU's memory is
    the RAM. None when /proc/meminfo cannot be read."""
    fields = {}
    for line in _read(meminfo).splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if name in ("MemTotal", "MemAvailable") and parts and parts[0].isdigit():
            fields[name] = int(parts[0])
    if len(fields) != 2:
        return None
    return max(0, fields["MemTotal"] - fields["MemAvailable"]) // 1024


def tegra_soc(compatible: Path = DT_COMPATIBLE) -> str | None:
    """The Tegra SoC of a Jetson ("tegra234" for Orin, "tegra264" for Thor); None anywhere else."""
    for entry in _read(compatible).split("\0"):
        if entry.startswith("nvidia,tegra"):
            return entry.split(",", 1)[1]
    return None


def board_name(model: Path = DT_MODEL) -> str | None:
    """The board as the device tree names it ("NVIDIA Jetson AGX Orin Developer Kit"); None without a device tree."""
    name = _read(model).replace("\0", "").strip()
    return name or None


def l4t_release(release: Path = TEGRA_RELEASE) -> str | None:
    """The Jetson Linux release ("R39.2.1") from /etc/nv_tegra_release; None when the file is missing or unreadable."""
    first = _read(release).splitlines()[:1]
    if not first:
        return None
    head = first[0].lstrip("# ").split(",")
    major = head[0].split()[0] if head and head[0].split() else ""
    revision = next((p.split(":", 1)[1].strip() for p in head if p.strip().startswith("REVISION:")), "")
    if not major.startswith("R") or not revision:
        return None
    return f"{major}.{revision}"


def memory_fields(memory_mib: str, *, arm64: bool | None = None, total_gb: float | None = None) -> dict[str, Any] | None:
    """The card fields for nvidia-smi's memory.total (MiB, ``nounits``).

    A card with memory of its own: ``{"vram_gb": ...}``. A GPU that shares the RAM (memory given as [N/A] on an arm64 Linux
    host): the unified-memory fields the readers and strata_control.gpu_fit know, sized to MemTotal. None when the field is
    neither, so the caller skips the row as it always has.
    """
    try:
        return {"vram_gb": float(memory_mib) / 1024.0}
    except ValueError:
        pass
    if not (arm64_linux() if arm64 is None else arm64):
        return None
    gb = ram_gb() if total_gb is None else total_gb
    if gb <= 0:
        return None
    fields: dict[str, Any] = {"vram_gb": gb, "unified_gb": gb, "dedicated_gb": 0.0, "integrated": True, "uma": True}
    board = board_name() if tegra_soc() else None
    if board:
        fields["board"] = board
    return fields
