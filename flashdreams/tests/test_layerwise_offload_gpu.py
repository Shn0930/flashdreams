# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from flashdreams.infra.acceleration.layerwise_offload import LayerwiseOffloader

pytestmark = pytest.mark.ci_gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_layerwise_offloader_cuda_prefetch_deep_queue_parity() -> None:
    """Exercise ready/reuse events without synchronizing between layer calls."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    layers = nn.ModuleList([nn.Linear(128, 128) for _ in range(6)])
    reference = copy.deepcopy(layers).to(device)
    offloader = LayerwiseOffloader(layers)
    layers.to(device)
    value = torch.randn(8, 128, device=device)

    expected = value
    for layer in reference:
        expected = layer(expected)

    actual_outputs: list[torch.Tensor] = []
    for _ in range(100):
        actual = value
        for layer_index, layer in enumerate(layers):
            with offloader.materialize(layer_index):
                actual = layer(actual)
        actual_outputs.append(actual)

    torch.cuda.synchronize(device)

    torch.testing.assert_close(
        torch.stack(actual_outputs),
        expected.unsqueeze(0).expand(len(actual_outputs), *expected.shape),
    )
    assert offloader.resident_layer_indices == (0,)
