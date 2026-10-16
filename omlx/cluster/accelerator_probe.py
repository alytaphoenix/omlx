# SPDX-License-Identifier: Apache-2.0
"""Classify a cluster node's accelerator, including ones MLX cannot drive.

:mod:`omlx.cluster.probe` already detects the *active MLX device*
(``cuda`` / ``metal`` / ``cpu``). That is enough for homogeneous Mac clusters,
but it silently mislabels an AMD ROCm box as ``cpu`` because MLX has no ROCm
backend (PyPI has ``mlx`` CPU/CUDA wheels only; there is no ``mlx-rocm`` /
``mlx-hip``).

This module answers the two questions a heterogeneous placement decision
actually needs:

1. **What is this node?** — ``kind`` (``metal`` / ``cuda`` / ``rocm`` / ``cpu``),
   ``vendor`` (``apple`` / ``nvidia`` / ``amd`` / ``unknown``).
2. **Can it join the MLX collective at all?** — ``mlx_rank_capable``. An AMD
   ROCm node is ``False``: it can only participate as an *external* OpenAI-
   compatible pool (see ``tools/omlx_federated_router.py``), never as an MLX
   Ring/JACCL/NCCL rank.

The module is import-light on purpose: it must import on a headless Linux
worker where ``mlx`` is absent, and on macOS. It never imports torch.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field

# MLX's distributed-backend vocabulary. A backend not in this set cannot be an
# MLX rank even when the hardware is present.
_MLX_BACKENDS = ("ring", "jaccl", "nccl")

# Where ROCm puts its runtime libraries.
_ROCM_LIB_DIRS = ("/opt/rocm/lib", "/opt/rocm/lib64", "/usr/lib/x86_64-linux-gnu")

# A ROCm install is only useful to us if the HIP runtime and RCCL are present.
_HIP_LIB_NAMES = ("libamdhip64.so",)
_RCCL_LIB_NAMES = ("librccl.so",)

# Accelerators with unified (CPU/GPU shared) memory rather than a separate VRAM
# pool. gfx1151 (Strix Halo, Ryzen AI MAX+) is unified.
_UNIFIED_ROCM_FAMILIES = ("gfx1151", "gfx1150", "gfx1103")

# PCI device IDs -> ROCm gfx family, for the boxes we care about. The sysfs
# uevent exposes PCI_ID without needing rocm-smi/rocminfo (and without the
# ROCm environment being sourced).
_PCI_ID_TO_GFX = {
    "1002:1586": "gfx1151",  # Strix Halo / Ryzen AI MAX+ (Radeon 8060S)
    "1002:1587": "gfx1151",
    "1002:150e": "gfx1150",  # Strix Point
    "1002:1900": "gfx1103",  # Phoenix/Hawk Point iGPU
}


@dataclass(frozen=True)
class AcceleratorProfile:
    """A node's accelerator and what it can participate in."""

    kind: str  # "metal" | "cuda" | "rocm" | "cpu"
    vendor: str  # "apple" | "nvidia" | "amd" | "unknown"
    memory_kind: str  # "unified" | "vram" | "system"
    mlx_rank_capable: bool
    collective_backends: tuple = ()
    system: str = ""
    machine: str = ""
    notes: tuple = field(default_factory=tuple)

    @property
    def is_external_pool_only(self) -> bool:
        """True when the box can serve models but not join the MLX collective."""

        return not self.mlx_rank_capable

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "vendor": self.vendor,
            "memory_kind": self.memory_kind,
            "mlx_rank_capable": self.mlx_rank_capable,
            "collective_backends": list(self.collective_backends),
            "system": self.system,
            "machine": self.machine,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# Low-level probes (monkeypatchable in tests)
# --------------------------------------------------------------------------- #


def _mlx_module():
    try:
        import mlx.core as mx  # noqa: PLC0415
        return mx
    except Exception:
        return None


def _mlx_active_kind() -> str | None:
    mx = _mlx_module()
    if mx is None:
        return None
    for name, kind in (("cuda", "cuda"), ("metal", "metal")):
        try:
            if bool(getattr(mx, name).is_available()):
                return kind
        except Exception:
            continue
    return "cpu"


def _mlx_distributed_backends() -> tuple:
    mx = _mlx_module()
    if mx is None:
        return ()
    found = []
    for backend in _MLX_BACKENDS:
        try:
            if bool(mx.distributed.is_available(backend)):
                found.append(backend)
        except Exception:
            continue
    return tuple(found)


def _find_library(names: tuple) -> str | None:
    for directory in _ROCM_LIB_DIRS:
        if not os.path.isdir(directory):
            continue
        try:
            entries = os.listdir(directory)
        except OSError:
            continue
        for entry in entries:
            if any(entry == n or entry.startswith(n) for n in names):
                return os.path.join(directory, entry)
    return None


def _rocm_present() -> bool:
    """ROCm is usable when /opt/rocm exists and the HIP runtime is loadable."""

    if not os.path.isdir("/opt/rocm"):
        return False
    return _find_library(_HIP_LIB_NAMES) is not None


def _rccl_present() -> bool:
    return _find_library(_RCCL_LIB_NAMES) is not None


def _rocm_gfx_family() -> str | None:
    """Best-effort ROCm gfx family from sysfs, without needing rocm-smi.

    Returns e.g. ``"gfx1151"`` when the PCI device id is known, else ``"amd"``
    when an AMD GPU is present but unrecognised, else ``None``.
    """

    drm = "/sys/class/drm"
    if not os.path.isdir(drm):
        return None
    for entry in sorted(os.listdir(drm)):
        if entry.startswith("card") and "-" not in entry:
            family = _gfx_from_drm_card(os.path.join(drm, entry))
            if family:
                return family
    return None


def _gfx_from_drm_card(path: str) -> str | None:
    uevent = os.path.join(path, "device", "uevent")
    try:
        with open(uevent, encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
    except OSError:
        return None
    if "amdgpu" not in text and "0x1002" not in text and "1002:" not in text:
        return None
    for line in text.splitlines():
        if line.startswith("PCI_ID="):
            pci_id = line.split("=", 1)[1].strip().lower()
            if pci_id in _PCI_ID_TO_GFX:
                return _PCI_ID_TO_GFX[pci_id]
    # AMD GPU confirmed, exact family unknown.
    return "amd"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def detect_accelerator_profile() -> AcceleratorProfile:
    system = platform.system()
    machine = platform.machine().lower()
    kind = _mlx_active_kind()
    notes = []

    if kind in ("metal", "cuda"):
        vendor = "apple" if kind == "metal" else "nvidia"
        memory_kind = "unified" if kind == "metal" else (
            "unified" if machine in ("arm64", "aarch64") else "vram"
        )
        return AcceleratorProfile(
            kind=kind,
            vendor=vendor,
            memory_kind=memory_kind,
            mlx_rank_capable=True,
            collective_backends=_mlx_distributed_backends(),
            system=system,
            machine=machine,
        )

    # MLX reports no GPU (or is absent). Before calling it a CPU box, check
    # whether this is an AMD ROCm machine MLX simply cannot drive.
    if _rocm_present():
        rccl = _rccl_present()
        family = _rocm_gfx_family()
        memory_kind = "unified" if (family in _UNIFIED_ROCM_FAMILIES) else "vram"
        if family is None:
            notes.append("rocm gfx family undetected; assuming vram")
        notes.append(
            "MLX has no ROCm backend: this node cannot be an MLX rank; "
            "run it as an external OpenAI-compatible pool"
        )
        if not rccl:
            notes.append("librccl not found; native collectives unavailable")
        return AcceleratorProfile(
            kind="rocm",
            vendor="amd",
            memory_kind=memory_kind,
            mlx_rank_capable=False,
            collective_backends=(("rccl",) if rccl else ()),
            system=system,
            machine=machine,
            notes=tuple(notes),
        )

    return AcceleratorProfile(
        kind="cpu",
        vendor="unknown",
        memory_kind="system",
        mlx_rank_capable=True,  # MLX CPU can still join a ring
        collective_backends=_mlx_distributed_backends(),
        system=system,
        machine=machine,
        notes=tuple(notes),
    )


def describe(profile: AcceleratorProfile) -> str:
    role = "MLX rank" if profile.mlx_rank_capable else "external pool only"
    backends = ",".join(profile.collective_backends) or "none"
    line = (
        f"{profile.kind} ({profile.vendor}, {profile.memory_kind} memory, "
        f"{role}) backends=[{backends}]"
    )
    if profile.notes:
        line += " | " + "; ".join(profile.notes)
    return line


if __name__ == "__main__":  # pragma: no cover - manual inspection
    print(describe(detect_accelerator_profile()))
