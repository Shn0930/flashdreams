<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OmniDreams

OmniDreams is an HD-map-conditioned streaming driving world-model integration.
Its public model variants are `OmnidreamsPipelineConfig` literals in
`omnidreams.config`; reusable applications own interactive I/O.

## Pipeline configurations

- `OMNIDREAMS_PIPELINE_CONFIG`
- `OMNIDREAMS_LAYERWISE_OFFLOAD_PIPELINE_CONFIG`
- `OMNIDREAMS_OPTIMIZED_GB300_PIPELINE_CONFIG`
- `OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_PIPELINE_CONFIG`
- `OMNIDREAMS_PERF_PIPELINE_CONFIG`
- `OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG`
- `OMNIDREAMS_RESPONSIVE_PIPELINE_CONFIG`
- `OMNIDREAMS_PERF_RESPONSIVE_PIPELINE_CONFIG`
- `OMNIDREAMS_FAST_PERF_RESPONSIVE_PIPELINE_CONFIG`
- `OMNIDREAMS_OPTIMIZED_GB300_RESPONSIVE_PIPELINE_CONFIG`
- `OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_RESPONSIVE_PIPELINE_CONFIG`

## Install

```bash
uv sync --package flashdreams-omnidreams --inexact
```

Checkpoints and example scenes download from Hugging Face on first use. Export
`HF_TOKEN` when the selected repository requires authentication.

## Applications

- [Interactive Drive](apps/interactive_drive/README.md)
- [Crazy Robotaxi](apps/crazy_robotaxi/README.md)

Launch Interactive Drive in a browser with:

```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive --inexact
uv run --no-sync flashdreams-run-v2 interactive-drive-omnidreams \
  --mode webrtc --host 0.0.0.0 --port 8089
```

For a lower-VRAM eager path, use the opt-in layer-wise offload application:

```bash
uv run --no-sync flashdreams-run-v2 \
  interactive-drive-omnidreams-layerwise-offload \
  --mode webrtc --host 0.0.0.0 --port 8089
```

This config streams DiT block weights from pinned CPU memory. It reduces GPU
memory use, but adds host-memory use and does not enable `torch.compile`, CUDA
graphs, native DiT acceleration, or optimized attention. See the
[HTML benchmark report](benchmarks/LAYERWISE_OFFLOAD_REPORT.html) for charts and
the [detailed methodology](benchmarks/LAYERWISE_OFFLOAD.md) for measured
tradeoffs and current limitations.

## Programmatic pipeline access

```python
from omnidreams.config import OMNIDREAMS_PIPELINE_CONFIG

pipeline = OMNIDREAMS_PIPELINE_CONFIG.setup().to("cuda").eval()
```

Replace the imported config with
`OMNIDREAMS_LAYERWISE_OFFLOAD_PIPELINE_CONFIG` to use the memory-oriented path.

## Tests

```bash
uv sync --package flashdreams-omnidreams --extra dev --group test --inexact
uv run --no-sync pytest integrations_v2/omnidreams/tests -m ci_cpu
```
