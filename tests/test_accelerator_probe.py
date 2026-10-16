# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.cluster.accelerator_probe (no MLX / no hardware required)."""

from __future__ import annotations

import omlx.cluster.accelerator_probe as ap


def _force(monkeypatch, *, mlx_kind, backends=(), rocm=False, rccl=False, family=None):
    monkeypatch.setattr(ap, "_mlx_active_kind", lambda: mlx_kind)
    monkeypatch.setattr(ap, "_mlx_distributed_backends", lambda: tuple(backends))
    monkeypatch.setattr(ap, "_rocm_present", lambda: rocm)
    monkeypatch.setattr(ap, "_rccl_present", lambda: rccl)
    monkeypatch.setattr(ap, "_rocm_gfx_family", lambda: family)


def test_metal_is_an_mlx_rank(monkeypatch):
    _force(monkeypatch, mlx_kind="metal", backends=("ring", "jaccl"))
    p = ap.detect_accelerator_profile()
    assert p.kind == "metal" and p.vendor == "apple"
    assert p.memory_kind == "unified"
    assert p.mlx_rank_capable is True
    assert p.collective_backends == ("ring", "jaccl")


def test_cuda_is_an_mlx_rank(monkeypatch):
    _force(monkeypatch, mlx_kind="cuda", backends=("ring", "nccl"))
    p = ap.detect_accelerator_profile()
    assert p.kind == "cuda" and p.vendor == "nvidia"
    assert p.mlx_rank_capable is True
    assert "nccl" in p.collective_backends


def test_rocm_node_is_external_pool_only(monkeypatch):
    # MLX reports no GPU, but the box is a ROCm Strix Halo machine.
    _force(monkeypatch, mlx_kind="cpu", rocm=True, rccl=True, family="gfx1151")
    p = ap.detect_accelerator_profile()
    assert p.kind == "rocm" and p.vendor == "amd"
    assert p.memory_kind == "unified"  # gfx1151 is unified
    assert p.mlx_rank_capable is False
    assert p.is_external_pool_only is True
    assert p.collective_backends == ("rccl",)
    assert any("no ROCm backend" in n for n in p.notes)


def test_rocm_without_rccl_has_no_collectives(monkeypatch):
    _force(monkeypatch, mlx_kind="cpu", rocm=True, rccl=False, family="gfx1151")
    p = ap.detect_accelerator_profile()
    assert p.kind == "rocm"
    assert p.collective_backends == ()
    assert any("librccl" in n for n in p.notes)


def test_plain_cpu_box(monkeypatch):
    _force(monkeypatch, mlx_kind="cpu", rocm=False)
    p = ap.detect_accelerator_profile()
    assert p.kind == "cpu" and p.vendor == "unknown"
    assert p.mlx_rank_capable is True  # CPU can join a ring


def test_describe_mentions_role(monkeypatch):
    _force(monkeypatch, mlx_kind="cpu", rocm=True, rccl=True, family="gfx1151")
    text = ap.describe(ap.detect_accelerator_profile())
    assert "external pool only" in text and "rccl" in text
