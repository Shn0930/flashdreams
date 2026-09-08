<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OmniDreams Interactive Drive

Install the OmniDreams integration and launch its Interactive Drive application:

```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive --inexact
uv run --no-sync flashdreams-run-v2 interactive-drive-omnidreams \
  --mode webrtc --host 0.0.0.0 --port 8089
```

The default scene and model assets download from Hugging Face on first use.
Export `HF_TOKEN` when the selected repository requires authentication.

Available application slugs:

| Application slug | Pipeline config |
| --- | --- |
| `interactive-drive-omnidreams` | `OMNIDREAMS_PIPELINE_CONFIG` |
| `interactive-drive-omnidreams-layerwise-offload` | `OMNIDREAMS_LAYERWISE_OFFLOAD_PIPELINE_CONFIG` |
| `interactive-drive-omnidreams-optimized-gb300` | `OMNIDREAMS_OPTIMIZED_GB300_PIPELINE_CONFIG` |
| `interactive-drive-omnidreams-optimized-rtx-pro-6000` | `OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_PIPELINE_CONFIG` |
| `interactive-drive-omnidreams-perf` | `OMNIDREAMS_PERF_PIPELINE_CONFIG` |
| `interactive-drive-omnidreams-fast-perf` | `OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG` |

The `perf` variants require a one-time preparation step before launch:

```bash
uv run --no-sync omnidreams-prepare --perf
```

Use the layer-wise offload slug when GPU memory is the constraint:

```bash
uv run --no-sync flashdreams-run-v2 \
  interactive-drive-omnidreams-layerwise-offload \
  --mode webrtc --host 0.0.0.0 --port 8089
```

This opt-in path keeps DiT block weights in pinned CPU memory and streams them
to two reusable GPU slots. It trades 3.61 GiB of host memory for lower GPU
memory on the measured one-GPU configuration, and disables `torch.compile`,
CUDA graphs, native DiT acceleration, and optimized attention. See the
[benchmark report](../../benchmarks/LAYERWISE_OFFLOAD.md) for details.

See the shared [Interactive Drive README](../../../../apps/interactive_drive/README.md)
for controls, application arguments, output modes, and tests. See the
[OmniDreams integration README](../../README.md) for model details.
