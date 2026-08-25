[简体中文](./README.md)

# vLLM for AMD gfx906

[![vLLM](https://img.shields.io/badge/vLLM-gfx906-red)](https://github.com/vllm-project/vllm)
[![ROCm](https://img.shields.io/badge/ROCm-6.3+-purple)](https://rocm.docs.amd.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](https://github.com/vllm-project/vllm/blob/main/LICENSE)

**An LLM inference engine optimized for AMD gfx906 GPUs (Radeon VII, Radeon Pro VII, Instinct MI50/MI60)**

## Copyright Notice

This project is based on [vLLM](https://github.com/vllm-project/vllm). We thank all contributors to the original project.

- **Original project**: [vLLM](https://github.com/vllm-project/vllm) by the UC Berkeley Sky Computing Lab
- **License**: Apache License 2.0
- **This project**: Includes optimizations and fixes for AMD gfx906 GPUs
- **Derived and referenced work**: This branch incorporates and references existing community ROCm/gfx906 adaptation work under the Apache-2.0 license. See [NOTICE](NOTICE) for attribution.

See [NOTICE](NOTICE) for the complete attribution statement.

## Project Status

⚠️ **This project is community-maintained.** The original author has archived their work, but this branch continues to be updated for compatibility with recent vLLM versions.

## Latest Updates

### ✅ August 2026 Update

- **Qwen3.5 multimodal GGUF support** - Added vision configuration, weight-name mapping, Conv3d patch-embedding merging, and multimodal input processing for Qwen3.5 GGUF main models and `mmproj` files.

### Supported Models

**⚠️ Important: model support status**

Many new model implementations (49+) have been synchronized from upstream vLLM, but they **have not been adapted or tested for ROCm/gfx906**. Their presence in the repository does not guarantee that they will run correctly on AMD gfx906 GPUs.

**Tested and confirmed working:**

- ✅ Qwen/Qwen3.5-0.8B (original F16 weights)
- ✅ Qwen/Qwen3.5-2B (original F16 weights)
- ✅ Qwen/Qwen3.5-4B (original F16 weights)
- ✅ Qwen/Qwen3.5-9B (original F16 weights)
- ✅ unsloth/Qwen3.5-27B-GGUF
- ✅ unsloth/Qwen3.6-27B-GGUF
- ✅ unsloth/Qwen3.8-27B-GGUF

**Known not to work:**

- ❌ **Qwen/Qwen3.5-35B-A3B-GPTQ-Int4** - The MoE + GPTQ Int4 combination has compatibility problems that prevent normal startup or inference.
- ❌ **cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit** - Testing on ROCm 6.4 + gfx906 still failed. Observed symptoms include output consisting entirely of `!` characters and crashes during engine initialization or inference.

**Untested models synchronized from upstream (may not work):**

- 🔄 Other MoE models, such as exaone_moe and glm4_moe_lite
- 🔄 Vision models, such as colqwen3 and molmo2
- 🔄 Audio models, such as whisper and funasr
- 🔄 Embedding models, such as colbert and voyage
- 🔄 Other new models

Test basic functionality before relying on these models. Please submit an issue if you encounter a problem.

## Overview

This is a modified version of [vLLM](https://github.com/vllm-project/vllm) for AMD gfx906 GPUs. It includes ROCm compatibility fixes, workarounds, and optimizations for the gfx906 architecture.

**Original author**: [nalanzeyu](https://github.com/nalanzeyu/vllm-gfx906)

## Requirements

- **Hardware**: AMD gfx906 GPU (Radeon VII, Radeon Pro VII, Instinct MI50, or Instinct MI60)
- **ROCm**: One of the following community-supported combinations; both require the kernel-mode driver:
  - ROCm 6.3 + PyTorch 2.9 + triton-gfx906 v3.5.0+gfx906 (original compatibility stack)
  - ROCm 7.2 + PyTorch 2.11 + triton-rocm 3.6.0 (current repository dependency stack)
- **Python**: 3.10+
- **Operating system**: Linux (tested on Ubuntu)

For the ROCm 6.3 Triton setup, see the
[triton-gfx906 installation guide](https://github.com/nlzy/triton-gfx906/tree/v3.5.0+gfx906).
The current dependency files default to the ROCm 7.2 stack. This stack requires
rocBLAS Tensile files compatible with gfx906. Set `ROCBLAS_TENSILE_LIBPATH` when
the files are stored outside the default search path.

## Installation

### Build from Source

```bash
# Install system dependencies
sudo apt install python3-venv python3-dev

# Clone the repository
git clone https://github.com/ttdxq/gfx906-vllm.git
cd gfx906-vllm

# Create a virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install the current ROCm 7.2 build and runtime dependencies
uv pip install -r requirements/rocm-build.txt -r requirements/rocm.txt

# Build and install vLLM
MAX_JOBS=$(nproc) CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
  uv pip install --no-build-isolation -e .
```

The commands above use the current ROCm 7.2 + PyTorch 2.11 dependency stack.
To continue using the original ROCm 6.3 stack, prepare the matching PyTorch and
Triton environment according to the triton-gfx906 installation guide. Do not
mix the two ROCm/PyTorch/Triton dependency stacks.

## Usage

### Start the Server

```bash
# Explicitly disable multimodal inputs for a text model or text-only workload
VLLM_USE_MODELSCOPE=true vllm serve Qwen/Qwen3.5-0.8B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --reasoning-parser qwen3_5 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}'

# Set the GPU memory utilization explicitly
vllm serve Qwen/Qwen3.5-0.8B \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser qwen3_5
```

### Dual-GPU Tensor Parallelism (TP=2)

> Prerequisite: `--tensor-parallel-size 2` requires replacing RCCL. The official AMD RCCL
> (including the one in the ROCm 7.2 repository) contains no gfx906 device kernels, so
> startup fails during initialization with `invalid kernel file` / `invalid device
> function` and similar errors. Install the gfx906-built RCCL drop-in (same 2.27.7
> source as official, built for gfx906 only); it targets torch `+rocm7.2` wheels.

**1. Install the gfx906 RCCL** (download `rccl-gfx906-2.27.7-rocm7.2.4.tar.gz` from Releases and extract):

```bash
# conda / venv environment (activate it first)
conda activate <your-env>
./install.sh
HIP_VISIBLE_DEVICES=0,1 python rccl_probe.py
# Expected output ALLREDUCE_OK: [2.0, 2.0, 2.0, 2.0] means success; roll back with ./install.sh --rollback

# uv project environment (--no-sync is mandatory: an implicit sync can break the custom torch)
uv run --no-sync ./install.sh
```

If no prebuilt package is available, build it yourself (~25 minutes):

```bash
git clone --depth 1 --branch rocm-7.2.4 https://github.com/ROCm/rccl
cd rccl && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm -DGPU_TARGETS="gfx906" -G Ninja
ninja -j$(nproc)
# Replace the librccl inside torch/lib with build/librccl.so.1.0 the same way install.sh does
```

**2. Start the TP=2 server**:

```bash
HIP_VISIBLE_DEVICES=0,1 \
vllm serve /path/to/Qwen3.8-27B-UD-Q6_K_XL.gguf \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --reasoning-parser qwen3_5 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}'
```

Notes:

- TP=1 does not go through RCCL at all; single-GPU deployments need none of the above
  and are unaffected.
- Under TP=2, greedy output may diverge from TP=1 at occasional near-tie tokens
  (floating-point summation order differences). This is expected and not an
  installation or weight problem.

### Qwen3.5 Family Multimodal GGUF Support

Place the main model and its matching `mmproj` file in the same directory and use recognizable paired filenames.

To constrain the dummy image size used during startup profiling, set `width` and `height` to values that match the largest images expected by the workload:

```bash
--limit-mm-per-prompt '{"image":{"count":1,"width":512,"height":512},"video":0}'
```

`width` and `height` constrain the dummy image used during startup profiling; they are not hard limits on runtime images. Production profiling should use dimensions that match the largest expected runtime image.

### Test the API

```bash
# List available models
curl http://localhost:8000/v1/models

# Send a chat completion request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-0.8B",
    "messages": [{"role": "user", "content": "Hello! Please introduce yourself."}],
    "max_tokens": 100
  }'
```

### Python Client Example

```python
from openai import OpenAI

# Initialize the client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-not-required"
)

# Send a chat completion request
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-0.8B",
    messages=[
        {"role": "user", "content": "Write a haiku about programming."}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
```

## Core Changes

### 1. ROCm GEMM Optimization

- **Problem**: ROCm GEMM compatibility and performance on gfx906 vary with batch size, matrix shape, and dtype.
- **Solution**: Select an implementation based on the input shape. Eligible single-token FP16 operations use `LLMM1`, some small non-gfx9 batches use Triton, and other gfx906 cases use the PyTorch/ROCm GEMM fallback.
- **Modified file**: `vllm/model_executor/layers/utils.py`

### 2. CacheConfig Compatibility

- Fixed the `mamba_cache_mode` attribute for hybrid models.
- Updated Qwen3.5 model-loading logic.

### 3. Platform Detection

- Improved ROCm platform detection for gfx906.
- Improved attention backend selection.

## Quantization Support

Based on testing from the original project:

- ⚠️ **GPTQ** - Some models work, but kernel, tensor-shape, and MoE combinations may still have compatibility or performance problems.
- ⚠️ **AWQ** - Some dense models work. The Triton path is used by default, but some models may still produce invalid output or fail during startup.
- ⚠️ **W4A16 INT** - Some models do not work (via llm-compressor).
- ⚠️ **Quantized MoE models** - Significantly slower and not recommended.
- ⚠️ **Unquantized models** - Slightly slower, but usable.

AWQ uses the current conservative Triton path by default on gfx906. To test the
GPTQ-compatible AWQ path based on `vllm-gfx906-mobydick`, set:

```bash
export VLLM_ROCM_USE_GFX906_MOBYDICK_AWQ=1
```

This option is disabled by default. Use it only to investigate Triton AWQ
compatibility or to compare the two AWQ paths.

See [Issue #29](https://github.com/nlzy/vllm-gfx906/issues/29) for more information.

## Known Limitations

1. **Slow first inference** - Triton kernels must be compiled on their first use.
2. **High memory usage** - KV cache preallocation can consume a large amount of GPU memory.
3. **Experimental software** - Use at your own risk.
4. **MoE + GPTQ compatibility** - Large MoE + GPTQ Int4 models such as Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 have known problems:
   - Triton kernels can encounter memory access errors with specific tile sizes.
   - The AITER backend is unstable in complex MoE routing scenarios.
   - Prefer an unquantized MoE model or a smaller MoE + GPTQ model.

## Performance Tuning

1. **Reduce `max-model-len`** - Saves memory, especially for smaller models.
2. **Set `--gpu-memory-utilization`** - Controls memory allocation explicitly.
3. **Keep the default eager decode path** - CUDA/HIP Graph paths for Qwen3.5/Qwen3.8 GGUF are intended only for troubleshooting and experimental validation.
4. **Constrain multimodal profiling dimensions** - Use `--limit-mm-per-prompt` to match the expected image count and maximum test dimensions.
5. **Check process ownership before restarting** - Use `rocm-smi` and `ps` to inspect GPU usage and avoid terminating processes that belong to another GPU workstream.

## Troubleshooting

### The Server Appears to Hang on the First Request

This is expected while Triton compiles kernels. Wait one or two minutes.

### GPU Out-of-Memory Errors

- Use `rocm-smi` to inspect memory use and process ownership on the target GPU.
- Reduce `--max-model-len`, `--max-num-batched-tokens`, or the KV cache size.
- Reduce the profiling image dimensions for multimodal models with `--limit-mm-per-prompt`.
- If the OOM occurs during vision profiling at startup, temporarily use `--skip-mm-profiling` to determine whether the profiling peak is the cause.

## Contributing

Contributions are welcome. You can:

- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## Acknowledgments

- **Original vLLM**: [UC Berkeley Sky Computing Lab](https://sky.cs.berkeley.edu)
- **ROCm port**: [Said-Akbar/vllm-rocm](https://github.com/Said-Akbar/vllm-rocm)
- **gfx906 branch**: [nalanzeyu/vllm-gfx906](https://github.com/nalanzeyu/vllm-gfx906)
- **gfx906 community adaptation**: [ai-infos/vllm-gfx906-mobydick](https://github.com/ai-infos/vllm-gfx906-mobydick)
- **Triton for gfx906**: [nlzy/triton-gfx906](https://github.com/nlzy/triton-gfx906)

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Citation

If you use this branch in research, cite the original vLLM paper and mention the gfx906 adaptation:

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact

- **Issue tracker**: [GitHub Issues](https://github.com/ttdxq/gfx906-vllm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ttdxq/gfx906-vllm/discussions)

---

**Note**: This is a community-maintained branch. Use it at your own risk, especially when making hardware purchasing decisions.
