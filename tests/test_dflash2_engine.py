# SPDX-License-Identifier: Apache-2.0
"""Tests for the DFlash 2 engine integration (omlx/engine/dflash2.py).

DFlash 2 is a second, independent speculative-decoding engine — unlike
DFlashEngine (omlx/engine/dflash.py) it does not wrap the dflash-mlx
package, so these tests never need ``pytest.importorskip("dflash_mlx")``.
mlx / mlx_lm are core omlx dependencies and are always importable.
"""

import json
from unittest.mock import AsyncMock

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from omlx.engine.dflash2 import (
    DFlash2DraftModel,
    DFlashConfig,
    DFlashDraftModel,
    GroupedDynamicCausalConv,
    _grouped_dynamic_convolve,
    _rejection_sample,
    _resolve_draft_snapshot_dir,
    _sample_logits,
    _sampling_probs,
    load_draft,
)
from omlx.model_settings import ModelSettings

# =============================================================================
# ModelSettings field plumbing
# =============================================================================


class TestDFlash2ModelSettings:
    def test_default_values(self):
        settings = ModelSettings()
        assert settings.dflash2_enabled is False
        assert settings.dflash2_draft_model is None

    def test_to_dict_includes_dflash2_fields(self):
        settings = ModelSettings(
            dflash2_enabled=True,
            dflash2_draft_model="z-lab/Qwen3.8-27B-DFlash2",
        )
        d = settings.to_dict()
        assert d["dflash2_enabled"] is True
        assert d["dflash2_draft_model"] == "z-lab/Qwen3.8-27B-DFlash2"

    def test_to_dict_excludes_none_draft_model(self):
        settings = ModelSettings(dflash2_enabled=True)
        d = settings.to_dict()
        assert "dflash2_draft_model" not in d

    def test_from_dict_roundtrip(self):
        data = {
            "dflash2_enabled": True,
            "dflash2_draft_model": "z-lab/Qwen3.8-27B-DFlash2",
        }
        settings = ModelSettings.from_dict(data)
        assert settings.dflash2_enabled is True
        assert settings.dflash2_draft_model == "z-lab/Qwen3.8-27B-DFlash2"

    def test_from_dict_missing_fields_uses_defaults(self):
        settings = ModelSettings.from_dict({})
        assert settings.dflash2_enabled is False
        assert settings.dflash2_draft_model is None

    def test_dflash_and_dflash2_mutually_exclusive(self):
        with pytest.raises(ValueError, match="dflash_enabled and dflash2_enabled"):
            ModelSettings(dflash_enabled=True, dflash2_enabled=True)

    def test_mtp_and_dflash2_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mtp_enabled and dflash2_enabled"):
            ModelSettings(mtp_enabled=True, dflash2_enabled=True)

    def test_vlm_mtp_and_dflash2_mutually_exclusive(self):
        with pytest.raises(ValueError, match="dflash2_enabled"):
            ModelSettings(
                vlm_mtp_enabled=True,
                vlm_mtp_draft_model="assistant",
                dflash2_enabled=True,
            )

    def test_dflash2_alone_is_valid(self):
        settings = ModelSettings(
            dflash2_enabled=True,
            dflash2_draft_model="z-lab/Qwen3.8-27B-DFlash2",
        )
        assert settings.dflash2_enabled is True


# =============================================================================
# engine_pool dispatch
# =============================================================================


class TestDFlash2EnginePoolRouting:
    """Settings-side of EnginePool's dflash2 dispatch branch.

    Mirrors TestDFlashEnginePoolRouting in test_dflash_engine.py: the real
    dispatch branch lives inside EnginePool._load_engine and needs a fully
    discovered model entry to exercise end to end, so — like the DFlash1
    tests — this checks the settings values the branch keys off of.
    """

    def test_dflash2_disabled_uses_default_engine(self):
        settings = ModelSettings(dflash2_enabled=False)
        assert not getattr(settings, "dflash2_enabled", False)

    def test_dflash2_enabled_without_draft_model(self):
        settings = ModelSettings(dflash2_enabled=True)
        assert getattr(settings, "dflash2_draft_model", None) is None

    def test_dflash2_enabled_with_draft_model(self):
        settings = ModelSettings(
            dflash2_enabled=True,
            dflash2_draft_model="z-lab/Qwen3.8-27B-DFlash2",
        )
        assert settings.dflash2_enabled is True
        assert settings.dflash2_draft_model == "z-lab/Qwen3.8-27B-DFlash2"


class TestEnginePoolDFlash2Isolation:
    """DFlash2Engine must be treated as a single-stream engine by
    EnginePool._unload_other_dflash_engines, same as DFlashEngine — see the
    updated docstring on that method in omlx/engine_pool.py."""

    class DFlash2Engine:
        def __init__(self, *, active: bool = False):
            self.active = active

        def has_active_requests(self):
            return self.active

    class DFlashEngine:
        def __init__(self, *, active: bool = False):
            self.active = active

        def has_active_requests(self):
            return self.active

    class OtherEngine:
        def has_active_requests(self):
            return False

    @staticmethod
    def _entry(model_id: str, engine):
        from omlx.engine_pool import EngineEntry

        return EngineEntry(
            model_id=model_id,
            model_path=f"/models/{model_id}",
            model_type="llm",
            engine_type="batched",
            estimated_size=1024,
            engine=engine,
        )

    @pytest.mark.asyncio
    async def test_unload_other_dflash_engines_unloads_idle_dflash2_too(self):
        from omlx.engine_pool import EnginePool

        pool = EnginePool()
        pool._entries["old-dflash2"] = self._entry(
            "old-dflash2", self.DFlash2Engine()
        )
        pool._entries["other"] = self._entry("other", self.OtherEngine())
        pool._entries["new-dflash2"] = self._entry("new-dflash2", None)

        unloaded = []

        async def fake_unload(model_id):
            unloaded.append(model_id)
            pool._entries[model_id].engine = None

        pool._unload_engine = fake_unload

        await pool._unload_other_dflash_engines("new-dflash2")

        assert unloaded == ["old-dflash2"]
        assert pool._entries["other"].engine is not None

    @pytest.mark.asyncio
    async def test_unload_other_dflash_engines_treats_dflash1_and_dflash2_alike(self):
        """A loaded DFlashEngine (dflash-mlx) and DFlash2Engine cannot both stay
        resident either -- both are single-stream and share the same
        "one DFlash engine at a time" isolation rule."""
        from omlx.engine_pool import EnginePool

        pool = EnginePool()
        pool._entries["old-dflash1"] = self._entry("old-dflash1", self.DFlashEngine())
        pool._entries["new-dflash2"] = self._entry("new-dflash2", None)

        unloaded = []

        async def fake_unload(model_id):
            unloaded.append(model_id)
            pool._entries[model_id].engine = None

        pool._unload_engine = fake_unload

        await pool._unload_other_dflash_engines("new-dflash2")

        assert unloaded == ["old-dflash1"]

    @pytest.mark.asyncio
    async def test_unload_other_dflash_engines_blocks_active_dflash2(self):
        from omlx.engine_pool import EnginePool

        pool = EnginePool()
        pool._entries["active-dflash2"] = self._entry(
            "active-dflash2", self.DFlash2Engine(active=True)
        )
        pool._entries["new-dflash2"] = self._entry("new-dflash2", None)
        pool._unload_engine = AsyncMock()

        with pytest.raises(RuntimeError, match="active-dflash2"):
            await pool._unload_other_dflash_engines("new-dflash2")

        pool._unload_engine.assert_not_awaited()


# =============================================================================
# Ported reference-code numerics
# =============================================================================


class TestSamplingHelpers:
    def test_sample_logits_greedy_is_argmax(self):
        logits = mx.array([[1.0, 5.0, 2.0, 0.5]])
        token = _sample_logits(logits, temperature=0.0)
        assert token.item() == 1

    def test_sampling_probs_sum_to_one(self):
        logits = mx.array([[1.0, 2.0, 3.0, 4.0]])
        probs = _sampling_probs(logits, temperature=1.0)
        assert abs(float(mx.sum(probs)) - 1.0) < 1e-5

    def test_sampling_probs_top_k_zeroes_out_rest(self):
        logits = mx.array([[1.0, 2.0, 3.0, 4.0]])
        probs = _sampling_probs(logits, temperature=1.0, top_k=2)
        nonzero = (probs > 0).astype(mx.int32)
        assert int(mx.sum(nonzero).item()) == 2
        # The top-2 logits (indices 2, 3) should be the surviving mass.
        assert float(probs[0, 0].item()) == 0.0
        assert float(probs[0, 1].item()) == 0.0

    def test_sampling_probs_top_p_keeps_at_least_top_token(self):
        logits = mx.array([[10.0, -10.0, -10.0, -10.0]])
        probs = _sampling_probs(logits, temperature=1.0, top_p=1e-6)
        # top_p this small should keep essentially just the top token.
        assert float(probs[0, 0].item()) > 0.99


class TestRejectionSample:
    def test_full_acceptance_when_draft_matches_target_exactly(self):
        # accept condition is `uniform() * q < p`; uniform() in [0, 1), so
        # setting q == p for every position guarantees acceptance regardless
        # of the random draw -- this makes the test deterministic without
        # needing to seed mx.random.
        mx.random.seed(0)
        gamma = 3
        vocab = 5
        draft_tokens = mx.array([[0, 1, 2]])
        probs_row = mx.array([0.4, 0.3, 0.2, 0.05, 0.05])
        target_probs = mx.broadcast_to(probs_row, (1, gamma + 1, vocab))
        draft_probs = mx.broadcast_to(probs_row, (1, gamma, vocab))

        accepted, bonus = _rejection_sample(draft_tokens, target_probs, draft_probs)

        assert accepted == gamma
        assert 0 <= bonus < vocab

    def test_immediate_rejection_when_draft_prob_is_zero(self):
        # q == 0 for the drafted token makes `uniform() * 0 < p` true only
        # when p > 0, which cumprod still satisfies -- so instead force
        # rejection by making the *draft* itself never satisfy the strict
        # inequality: q > p at position 0 relative to the sampled uniform
        # is nondeterministic in general, but q=0 with p=0 at position 0
        # forces `0 < 0` == False deterministically, causing rejection at
        # the first position regardless of the random draw.
        mx.random.seed(0)
        vocab = 4
        draft_tokens = mx.array([[0, 1]])
        target_probs = mx.array(
            [[[0.0, 0.4, 0.3, 0.3], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]]
        )
        draft_probs = mx.array([[[0.5, 0.2, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]]])

        accepted, bonus = _rejection_sample(draft_tokens, target_probs, draft_probs)

        assert accepted == 0
        assert 0 <= bonus < vocab


class TestGroupedDynamicConvolve:
    def test_single_tap_reduces_to_weighted_identity(self):
        # kernel_size=1 means the convolve loop only runs the offset=0 term,
        # so output == (base[0] + dynamic[..., 0, :, :]) * hidden elementwise
        # (no mixing with earlier positions).
        batch, length, hidden_size, group_size = 1, 3, 4, 2
        groups = hidden_size // group_size
        hidden = mx.arange(batch * length * hidden_size).reshape(
            batch, length, hidden_size
        ).astype(mx.float32)
        base = mx.ones((1, hidden_size))  # kernel_size=1
        dynamic = mx.zeros((batch, length, 1, groups, 1))  # all-zero dynamic term

        out = _grouped_dynamic_convolve(hidden, dynamic[..., 0, :, :], base, group_size)

        assert out.shape == hidden.shape
        assert mx.allclose(out, hidden).item()

    def test_module_prepare_finish_roundtrip_shapes(self):
        hidden_size, kernel_size, group_size = 8, 2, 4
        conv = GroupedDynamicCausalConv(hidden_size, kernel_size, group_size)
        hidden = mx.random.normal((1, 5, hidden_size))

        prepared, kernel = conv.prepare(hidden)
        assert prepared.shape == hidden.shape

        finished = conv.finish(hidden, kernel)
        assert finished.shape == hidden.shape


# =============================================================================
# load_draft / config parsing against the real z-lab checkpoint layout
# =============================================================================


# The z-lab/Qwen3.8-27B-DFlash2 checkpoint's real config.json (verbatim
# field shapes: nested rope_parameters, nested dflash_config sub-object,
# architectures selecting DFlash2DraftModel).
_REAL_DFLASH2_CONFIG = {
    "architectures": ["DFlash2DraftModel"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": None,
    "is_causal": False,
    "dflash_config": {
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 248070,
        "selector_rank": 256,
        "selector_top_k": 16,
        "target_layer_ids": [5, 19, 33, 47, 61],
    },
    "dtype": "bfloat16",
    "eos_token_id": 248044,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 17408,
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
    ],
    "max_position_embeddings": 262144,
    "max_window_layers": 5,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 5,
    "num_key_value_heads": 8,
    "num_target_layers": 64,
    "pad_token_id": 248044,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
    "sliding_window": 2048,
    "tie_word_embeddings": False,
    "transformers_version": "5.15.0",
    "use_cache": True,
    "use_sliding_window": True,
    "vocab_size": 248320,
}


def _write_draft_checkpoint(tmp_path, cfg: dict, model: "DFlashDraftModel"):
    """Write a synthetic (tiny) checkpoint dir matching load_draft's expectations."""
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    flat = dict(tree_flatten(dict(model.parameters())))
    if isinstance(model, DFlash2DraftModel):
        # The real checkpoint serializes these two embedding tables without
        # the trailing ".weight" -- load_draft's remap step at the bottom of
        # the function compensates for this quirk.
        for name in ("predecessor_codebook", "successor_codebook"):
            key = f"candidate_selector.{name}"
            flat[key] = flat.pop(f"{key}.weight")
    mx.save_safetensors(str(tmp_path / "model.safetensors"), flat)


def _tiny_dflash2_config(**overrides) -> DFlashConfig:
    base = dict(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=100,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=4096,
        block_size=4,
        target_layer_ids=(0,),
        num_target_layers=1,
        mask_token_id=1,
        layer_types=("full_attention",),
        conv_kernel_size=2,
        conv_group_size=8,
        selector_rank=4,
        selector_top_k=4,
    )
    base.update(overrides)
    return DFlashConfig(**base)


class TestLoadDraft:
    def test_real_config_shape_parses_into_dflash_config(self, tmp_path):
        """Verify DFlashConfig field derivation against the actual released
        checkpoint's config.json layout, without downloading real weights."""
        # Build a tiny substitute model whose dims match a shrunk copy of
        # the real config, so load_draft's weight-loading path is exercised
        # too (not just the config-parsing branch).
        cfg = dict(_REAL_DFLASH2_CONFIG)
        cfg.update(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            intermediate_size=64,
            vocab_size=100,
            num_target_layers=1,
        )
        # Structural fields (conv_kernel_size/conv_group_size/selector_rank/
        # selector_top_k/target_layer_ids) must match the tiny synthetic
        # model's own architecture below -- only the non-structural fields
        # (block_size, mask_token_id, rope_theta, is_causal) keep the real
        # checkpoint's values, to prove those round-trip through load_draft.
        cfg["dflash_config"] = dict(cfg["dflash_config"])
        cfg["dflash_config"].update(
            conv_kernel_size=2,
            conv_group_size=8,
            selector_rank=4,
            selector_top_k=4,
            target_layer_ids=[0],
        )
        cfg["layer_types"] = ["full_attention"]
        cfg.pop("sliding_window", None)
        cfg["use_sliding_window"] = False

        model = DFlash2DraftModel(_tiny_dflash2_config())
        _write_draft_checkpoint(tmp_path, cfg, model)

        loaded = load_draft(str(tmp_path))

        assert isinstance(loaded, DFlash2DraftModel)
        assert loaded.config.block_size == 8
        assert loaded.config.mask_token_id == 248070
        assert loaded.config.target_layer_ids == (0,)
        assert loaded.config.rope_theta == 10000000
        assert loaded.config.conv_kernel_size == 2
        assert loaded.config.conv_group_size == 8
        assert loaded.config.selector_rank == 4
        assert loaded.config.selector_top_k == 4
        assert loaded.config.is_causal is False

    def test_non_dflash2_architecture_loads_plain_draft_model(self, tmp_path):
        cfg = dict(_REAL_DFLASH2_CONFIG)
        cfg["architectures"] = ["DFlashDraftModel"]
        cfg.update(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            intermediate_size=64,
            vocab_size=100,
            num_target_layers=1,
        )
        cfg["dflash_config"] = dict(cfg["dflash_config"])
        cfg["dflash_config"]["target_layer_ids"] = [0]
        cfg["layer_types"] = ["full_attention"]
        cfg.pop("sliding_window", None)
        cfg["use_sliding_window"] = False

        model = DFlashDraftModel(_tiny_dflash2_config())
        _write_draft_checkpoint(tmp_path, cfg, model)

        loaded = load_draft(str(tmp_path))

        assert type(loaded) is DFlashDraftModel

    def test_layer_types_length_mismatch_raises(self, tmp_path):
        cfg = dict(_REAL_DFLASH2_CONFIG)
        cfg["num_hidden_layers"] = 5
        cfg["layer_types"] = ["full_attention"]  # length 1 != num_hidden_layers 5
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        with pytest.raises(ValueError, match="layer_types length"):
            load_draft(str(tmp_path))

    def test_unsupported_layer_type_raises(self, tmp_path):
        cfg = dict(_REAL_DFLASH2_CONFIG)
        cfg["num_hidden_layers"] = 1
        cfg["layer_types"] = ["bogus_attention"]
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        with pytest.raises(ValueError, match="Unsupported draft layer_types"):
            load_draft(str(tmp_path))

    def test_sliding_attention_without_sliding_window_raises(self, tmp_path):
        cfg = dict(_REAL_DFLASH2_CONFIG)
        cfg["num_hidden_layers"] = 1
        cfg["layer_types"] = ["sliding_attention"]
        cfg.pop("sliding_window", None)
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        with pytest.raises(ValueError, match="sliding_window"):
            load_draft(str(tmp_path))


class TestResolveDraftSnapshotDir:
    def test_local_directory_used_directly(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")

        resolved = _resolve_draft_snapshot_dir(str(tmp_path))

        assert resolved == tmp_path

    def test_non_local_path_falls_back_to_snapshot_download(self, monkeypatch, tmp_path):
        calls = []

        def fake_snapshot_download(repo_id, **kwargs):
            calls.append(repo_id)
            return str(tmp_path)

        monkeypatch.setattr(
            "omlx.engine.dflash2.snapshot_download", fake_snapshot_download
        )

        resolved = _resolve_draft_snapshot_dir("z-lab/Qwen3.8-27B-DFlash2")

        assert calls == ["z-lab/Qwen3.8-27B-DFlash2"]
        assert resolved == tmp_path

    def test_local_directory_without_config_json_falls_back(self, monkeypatch, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        calls = []

        def fake_snapshot_download(repo_id, **kwargs):
            calls.append(repo_id)
            return str(tmp_path)

        monkeypatch.setattr(
            "omlx.engine.dflash2.snapshot_download", fake_snapshot_download
        )

        _resolve_draft_snapshot_dir(str(empty_dir))

        assert calls == [str(empty_dir)]


# =============================================================================
# DFlash2Engine scaffolding (constructor / properties, no model load)
# =============================================================================


class TestDFlash2EngineInit:
    def test_import(self):
        from omlx.engine import DFlash2Engine  # noqa: F401

    def test_engine_properties(self):
        from omlx.engine.dflash2 import DFlash2Engine

        engine = DFlash2Engine(
            model_name="test-model",
            draft_model_path="test-draft",
        )
        assert engine.model_name == "test-model"
        assert engine.tokenizer is None
        assert engine.model_type is None
        assert engine.has_active_requests() is False

    def test_scheduler_config_snapshot_at_construction(self):
        from omlx.engine.dflash2 import DFlash2Engine
        from omlx.scheduler import SchedulerConfig

        shared_config = SchedulerConfig(
            model_name="model-a", model_path="/models/model-a"
        )
        engine = DFlash2Engine(
            model_name="/models/model-a",
            draft_model_path="test-draft",
            scheduler_config=shared_config,
        )

        # Simulate the pool loading another model afterwards.
        shared_config.model_name = "model-b"
        shared_config.model_path = "/models/model-b"

        assert engine._scheduler_config.model_name == "model-a"
        assert engine._scheduler_config.model_path == "/models/model-a"

    def test_get_stats_before_load(self):
        from omlx.engine.dflash2 import DFlash2Engine

        engine = DFlash2Engine(
            model_name="test-model",
            draft_model_path="test-draft",
        )
        stats = engine.get_stats()
        assert stats["engine_type"] == "dflash2"
        assert stats["model_name"] == "test-model"
        assert stats["draft_model"] == "test-draft"
        assert stats["loaded"] is False
        assert stats["block_size"] is None

    def test_get_cache_stats_returns_none(self):
        from omlx.engine.dflash2 import DFlash2Engine

        engine = DFlash2Engine(model_name="test-model", draft_model_path="test-draft")
        assert engine.get_cache_stats() is None

    def test_has_active_requests_covers_executor_drain_window(self):
        """_active_request must cover the gap between _end_activity (called
        before the executor future is awaited) and the executor thread
        actually finishing -- otherwise TTL eviction could race a drain."""
        from omlx.engine.dflash2 import DFlash2Engine

        engine = DFlash2Engine(model_name="test-model", draft_model_path="test-draft")
        assert engine.has_active_requests() is False

        engine._active_request = True
        assert engine.has_active_requests() is True

        engine._active_request = False
        assert engine.has_active_requests() is False

    def test_detect_needs_think_prefix(self):
        from omlx.engine.dflash2 import DFlash2Engine

        engine = DFlash2Engine(model_name="test-model", draft_model_path="test-draft")

        class _FakeTokenizer:
            think_start_id = 42
            think_end_id = 43

        engine._tokenizer_obj = _FakeTokenizer()
        assert engine._detect_needs_think_prefix([1, 2, 42]) is True
        assert engine._detect_needs_think_prefix([1, 2, 3]) is False
        assert engine._detect_needs_think_prefix([]) is False
        # <think></think> immediately closed -> thinking disabled.
        assert engine._detect_needs_think_prefix([42, 43]) is False

    def test_think_prefix_text_default(self):
        from omlx.engine.dflash2 import DFlash2Engine

        engine = DFlash2Engine(model_name="test-model", draft_model_path="test-draft")
        engine._tokenizer_obj = object()
        assert engine._think_prefix_text() == "<think>\n"
