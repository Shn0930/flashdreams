<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OmniDreams layer-wise offload

Open the [self-contained HTML benchmark report](LAYERWISE_OFFLOAD_REPORT.html)
for charts, implementation diagrams, and embedded per-round latency samples.

Layer-wise offload is a **useful opt-in memory optimization**, not a throughput
optimization. On the measured H20-3e stack, it reduced peak PyTorch-allocated
GPU memory by 3.37 GiB (34.16%) versus the matched eager resident path while
changing median full-chunk latency by +0.10%. It consumed an additional
3.61 GiB of pinned host memory.

## How it works

After checkpoint loading and load-time weight transforms, each of the 28 DiT
blocks is packed into aligned, pinned CPU buffers. The original block parameters
are replaced by empty placeholders, so moving the pipeline to CUDA does not make
all block weights resident at once. At inference time, two fixed-size GPU slots
hold the current and next block. A dedicated CUDA copy stream prefetches the next
block while the current block computes; events protect slot reuse. Weights are
immutable during inference, so they are evicted without a device-to-host copy.

The explicit materialization scope covers regular block forward, cross-attention
KV-cache initialization, and text-embedding replacement. Non-block pipeline
weights and runtime activations remain GPU-resident.

Use the public config directly:

```python
from omnidreams.config import OMNIDREAMS_LAYERWISE_OFFLOAD_PIPELINE_CONFIG

pipeline = OMNIDREAMS_LAYERWISE_OFFLOAD_PIPELINE_CONFIG.setup().to("cuda").eval()
```

Or launch Interactive Drive with its dedicated application slug:

```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive --inexact
uv run --no-sync flashdreams-run-v2 \
  interactive-drive-omnidreams-layerwise-offload \
  --mode webrtc --host 0.0.0.0 --port 8089
```

## Limitations

The current implementation is for single-threaded inference with immutable
weights. It does not support training, runtime parameter mutation, DTensor,
tied or storage-aliased parameters, dtype conversion after construction, or
serialization after offload is enabled. Reload the original checkpoint instead.

For OmniDreams, layer-wise offload requires the standard OmniDreams self- and
cross-attention backends. It is incompatible with native DiT acceleration and
runtime text-edit LoRA. `torch.compile` and CUDA graphs are bypassed because they
require stable whole-network parameter storage; the public offload config sets
both flags to false explicitly.

## Benchmark method

Measurements were collected on 2026-09-08 from a branch based on commit
`d9b25167`, using one NVIDIA H20-3e (GPU 6, NUMA node 1), driver 580.95.05,
PyTorch 2.12.1+cu130, CUDA 13.0, and cuDNN 9.2.0. The checkpoint was the cached
`1view-vae-chunk2` BF16 model
(`2b_res720p_30fps_i2v_hdmap_distilled.pt`). Hugging Face offline mode was used.

Each process used batch size 1, one view, 704x1280 output, 512 zero text tokens,
zero initial-image embeddings, and a fixed BF16 random HDMap control generated
with seed 0. The regular model has 28 blocks, `len_t=2`, a six-latent-frame
attention window, no sink tokens, and emits eight decoded frames per steady
chunk. The timed region covered the recurring HDMap encoder, two DiT denoising
steps at timesteps 1000 and 500, TAEHV decode, and cache finalization. One-shot
text and first-frame encoders were intentionally excluded.

The attention window was filled before measurement. Each case then ran five
warmup chunks and 20 measured chunks in a separate process. Checkpoint loading,
pinned-buffer construction, CUDA-graph capture, and first-visible/startup time
were excluded. `torch.backends.cudnn.benchmark` was enabled; warmup absorbed
kernel autotuning. Kernel-level fallback telemetry was not collected. The GPU
had idle service contexts from other processes during collection, but no
competing utilization was observed in the retained clean runs; reported memory
is process-local PyTorch allocator memory rather than total `nvidia-smi` use.

The eager resident and offload cases both disabled compile and CUDA graphs for a
matched comparison. The deployment baseline kept weights resident and enabled
CUDA graphs, with compile disabled. Reproduce the three fresh-process runs with:

```bash
mkdir -p artifacts/benchmark/omnidreams/layerwise

CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 \
  numactl --cpunodebind=1 --membind=1 \
  .venv/bin/python -m pytest \
  'integrations_v2/omnidreams/benchmarks/test_pipeline.py::test_full_pipeline_layerwise_offload_benchmark[eager-resident]' \
  -p no:manual_marker -m manual --benchmark-only \
  --benchmark-json=artifacts/benchmark/omnidreams/layerwise/pipeline-eager-resident.json

CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 \
  numactl --cpunodebind=1 --membind=1 \
  .venv/bin/python -m pytest \
  'integrations_v2/omnidreams/benchmarks/test_pipeline.py::test_full_pipeline_layerwise_offload_benchmark[layerwise-offload]' \
  -p no:manual_marker -m manual --benchmark-only \
  --benchmark-json=artifacts/benchmark/omnidreams/layerwise/pipeline-layerwise-offload.json

CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 \
  numactl --cpunodebind=1 --membind=1 \
  .venv/bin/python -m pytest \
  integrations_v2/omnidreams/benchmarks/test_pipeline.py::test_full_pipeline_layerwise_offload_deployment_baseline \
  -p no:manual_marker -m manual --benchmark-only \
  --benchmark-json=artifacts/benchmark/omnidreams/layerwise/pipeline-cuda-graph-resident.json
```

## Results

| Mode | Median chunk (s) | p90 (s) | Frames/s | Steady allocated (GiB) | Reserved (GiB) | Peak allocated (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eager, resident | 1.553781 | 1.554243 | 5.149 | 9.410452 | 16.716797 | 9.870143 |
| Eager, layer-wise offload | 1.555395 | 1.557140 | 5.143 | 6.038056 | 13.369141 | 6.498617 |
| CUDA graph, resident deployment baseline | 1.548858 | 1.549428 | 5.165 | 9.448424 | 16.833984 | 9.568480 |

Against the matched eager baseline, offload reduced steady allocated memory by
3.3724 GiB (35.84%), reserved memory by 3.3477 GiB (20.03%), and peak allocated
memory by 3.3715 GiB (34.16%). Median latency increased by 0.10%. The offloader
managed 3.6094 GiB of block parameters, used 3.6094 GiB of pinned host storage,
and reserved 0.2578 GiB for its two GPU slots.

Against the CUDA-graph deployment baseline, offload reduced peak allocated GPU
memory by 3.0699 GiB (32.08%) and increased median latency by 0.42%. This single
H20-3e result should not be generalized to other interconnects or model shapes
without rerunning the benchmark.

All three final outputs had the same SHA-256 digest:
`c610fb94cc012fe39e23aa3fd74a9d4d86bcfb365f710d47d154741fd7aa8bd7`.
This validates exact output parity for the controlled final chunk; it is not a
substitute for a broader rollout-quality evaluation.
