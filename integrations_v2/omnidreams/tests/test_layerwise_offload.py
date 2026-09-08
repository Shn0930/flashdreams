# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

import pytest
import torch
from omnidreams.impl.transformer import CosmosTransformer, CosmosTransformerConfig
from omnidreams.impl.transformer.modules import AttentionBackend
from omnidreams.impl.transformer.network import (
    CosmosDiTNetwork,
    CosmosDiTNetworkConfig,
)

pytestmark = pytest.mark.ci_cpu


def test_transformer_layerwise_offload_disables_whole_network_acceleration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = CosmosTransformerConfig(
        network=_small_network_config(),
        dtype=torch.float32,
        compile_network=True,
        use_cuda_graph=True,
        enable_layerwise_offload=True,
    )

    transformer = CosmosTransformer(config)

    assert isinstance(transformer.network, CosmosDiTNetwork)
    assert transformer.network.layerwise_offloader is not None
    assert transformer._use_cuda_graph is False
    with pytest.raises(ValueError, match="text-edit LoRA"):
        transformer.set_text_edit_lora(object())


def test_layerwise_offload_covers_cache_initialization_and_text_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    network = CosmosDiTNetwork(_small_network_config()).to(dtype=torch.float32)
    network.eval()
    network.update_parameters_after_loading_checkpoint()
    network.enable_layerwise_offload()
    context = torch.randn(1, 1, 3, 16)

    cache = network.initialize_cache(
        chunk_size=2,
        window_size=4,
        sink_size=0,
        context=context,
    )
    network.replace_text_embeddings(cache, torch.randn_like(context))

    assert len(cache.block_caches) == 2
    assert network.layerwise_offloader is not None
    assert network.layerwise_offloader.resident_layer_indices == (0,)
    assert cache[0].cross_attn.cached_k().shape == (1, 3, 4, 8)


@pytest.mark.parametrize(
    ("native_acceleration", "self_attention_backend", "message"),
    [
        (
            "required",
            AttentionBackend.OMNIDREAMS,
            "native DiT acceleration",
        ),
        (
            "disabled",
            AttentionBackend.OPTIMIZED,
            "OmniDreams self- and cross-attention backends",
        ),
    ],
)
def test_layerwise_offload_rejects_incompatible_dit_paths(
    native_acceleration: Literal["disabled", "required"],
    self_attention_backend: AttentionBackend,
    message: str,
) -> None:
    config = CosmosTransformerConfig(
        network=_small_network_config(self_attention_backend=self_attention_backend),
        dtype=torch.float32,
        compile_network=False,
        use_cuda_graph=False,
        enable_layerwise_offload=True,
        native_dit_acceleration=native_acceleration,
    )

    with pytest.raises(ValueError, match=message):
        CosmosTransformer(config)


def _small_network_config(**overrides: Any) -> CosmosDiTNetworkConfig:
    values: dict[str, Any] = {
        "in_channels": 2,
        "out_channels": 2,
        "patch_spatial": 1,
        "model_channels": 32,
        "num_blocks": 2,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "adaln_lora_dim": 8,
        "crossattn_proj_in_channels": 16,
        "crossattn_emb_channels": 16,
    }
    values.update(overrides)
    return CosmosDiTNetworkConfig(**values)
