# GGUF

!!! warning
    Please note that GGUF support in vLLM is highly experimental and under-optimized at the moment, it might be incompatible with other features. Currently, you can use GGUF as a way to reduce memory footprint. If you encounter any issues, please report them to the vLLM team.

!!! note
    This fork includes a GGUF loader with model-specific adapters. Upstream
    vLLM provides GGUF support through
    [vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin).

The built-in loader requires the GGUF Python package:

```bash
uv pip install 'gguf>=0.18.0'
```

To run a GGUF model with vLLM, you can use the `repo_id:quant_type` format to load directly from HuggingFace. For example, to load a Q4_K_M quantized model from [unsloth/Qwen3-0.6B-GGUF](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF):

```bash
# We recommend using the tokenizer from base model to avoid long-time and buggy tokenizer conversion.
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M --tokenizer Qwen/Qwen3-0.6B
```

You can also add `--tensor-parallel-size 2` to enable tensor parallelism inference with 2 GPUs:

```bash
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M \
   --tokenizer Qwen/Qwen3-0.6B \
   --tensor-parallel-size 2
```

Alternatively, you can download and use a local GGUF file:

```bash
wget https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf
vllm serve ./Qwen3-0.6B-Q4_K_M.gguf --tokenizer Qwen/Qwen3-0.6B
```

## Qwen4Exp on ROCm

The built-in adapter supports `qwen4exp` GGUF checkpoints such as
Swift-1.5-Qwen3.8-Flash-Next. Pass the first file of a numbered split; the
loader finds the remaining files in the same directory. The config and text
tokenizer are constructed from the GGUF metadata.

The AMD QSA backend supports FP16 activations and FP16 main and indexer caches.
Use `--dtype half` on gfx906. FP32 accumulation remains enabled for attention
and normalization, and no BF16 opt-in is required.

PLE stays enabled. Its table uses pinned host memory by default and retains
the GGUF quantization format; only requested rows are dequantized on the GPU.
For the IQ2_XS checkpoint, the Q4_0 PLE table occupies about 26.8 GiB of host
memory across the tensor-parallel ranks, rather than about 95.4 GiB as FP16.
GGUF checkpoints with more than one PLE table are currently unsupported.

```bash
vllm serve /path/to/Swift-1.5-Qwen3.8-Flash-Next-IQ2_XS-00001-of-00002.gguf \
  --tensor-parallel-size 2 \
  --dtype half \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 32 \
  --mamba-ssm-cache-dtype float32 \
  --attention-config '{"indexer_kv_dtype": "fp16"}' \
  --engram-config '{"cpu_offload": true}' \
  --enforce-eager
```

The FP32 recurrent cache is independent of the FP16 QSA caches. Choose batch
size and GPU memory utilization to fit the backbone, recurrent state, and
attention cache together. On 32 GiB GPUs, smaller prefill batches may be needed.

On gfx906, Qwen4Exp uses the parallel FLA chunk algorithm for GDN prefill,
including continuation from FP32 recurrent state. Packed recurrent decode
is preserved. Set `VLLM_QWEN4EXP_GDN_CHUNK_PREFILL=0` before starting the
server to use the previous recurrent prefill implementation.

FP16 HC down projections with input widths above 8192 and up to 16384 use
a dedicated GEMV for batches of 1–16 tokens on gfx906. Larger batches,
biased projections, batch invariant mode, and explicitly selected linear
backends retain their existing implementation. Set `VLLM_QWEN4EXP_HC_GEMV=0`
to disable this optimization. The FP16 GDN output projection with shape
`[2560, 3072]` also uses the GEMV for single-token decode on gfx906. Set
`VLLM_QWEN4EXP_GDN_OUT_PROJ_GEMV=0` to disable that dispatch. These switches
apply to Qwen4Exp; Qwen3.5 models such as Qwen3.8-27B retain their existing GDN
path.

On gfx906, Qwen4Exp GDN with 128-wide key and value heads keeps its recurrent
state in K×V order. This enables the existing warp-cooperative single-token
decode kernel and lets chunk prefill write the same cache layout. Mixed
prefill/decode and recurrent fallback calls receive a V×K view of that cache.
Set `VLLM_QWEN4EXP_GDN_KV_STATE=0` before starting the server to restore V×K
storage.

IQ2_XS, IQ3_S, and IQ4_NL experts use grouped quantized matrix multiplication for
prefill batches at or above `VLLM_GGUF_IQ_MOE_MMQ_MIN_TOKENS` (default: 192 on
gfx906, 256 on other GPUs).
The general `VLLM_GGUF_MOE_MMQ_MIN_TOKENS` threshold also applies. The IQ path
uses 8-token tiles on ROCm and supports models with more than 256 experts and
IQ4_NL input widths such as 320. Smaller batches and decode keep MMVQ.
Set `VLLM_GGUF_IQ_MOE_MMQ=0` before starting the server to use MMVQ for these
experts during prefill as well.

GGUF MoE selects the prefill algorithm independently for the gate/up and down
projections. A supported down projection can use grouped MMQ while gate/up
keeps MMVQ, and vice versa. Token alignment is shared only when both products
use the same tile size. Decode retains its MMVQ path.

Eager prefill batches larger than four tokens share the W1 MMVQ output and
Q8 activation scratch across layers and tail shapes. This scratch belongs
to each runner execution slot and grows only during warmup. Larger unwarmed
shapes use temporary storage, and GPU events protect reuse across streams.
Returned outputs have independent storage. Decode and CUDA Graph capture
retain their existing caches. This reduces persistent memory when increasing
`--max-num-batched-tokens`; attention and other temporary allocations still
need sufficient free memory.

UVA-offloaded GGUF experts use a shared prefill staging buffer when a batch
has at least 32 tokens and at least twice as many expert routes as experts.
Each eligible matrix is copied to the buffer immediately before its product;
W1, W2, and successive layers reuse the same storage. Packed weights and
quantized arithmetic are unchanged. CUDA Graph capture and smaller batches
keep direct UVA reads.

`VLLM_GGUF_MOE_PREFILL_STAGING_MB` bounds this buffer per device (default:
512 MiB; set to `0` to disable). Matrices larger than the budget or with
noncontiguous storage keep direct UVA reads. The buffer is allocated lazily
and counted by the normal memory profiling run. If allocation fails, staging
is disabled for that device. Leave sufficient space for this scratch buffer
when choosing `--cpu-offload-gb`; the setting does not increase the offload
budget automatically.

Qwen4Exp FP16 QSA on gfx906 uses narrow 16-token attention tiles for prefill
with 256-wide heads and 12 query heads per KV head. This avoids the register
pressure of the wider GPU profile; single-token decode and other head or cache
formats retain their existing dispatch.

!!! warning
    We recommend using the tokenizer from base model instead of GGUF model. Because the tokenizer conversion from GGUF is time-consuming and unstable, especially for some models with large vocab size.

GGUF assumes that HuggingFace can convert the metadata to a config file. In case HuggingFace doesn't support your model you can manually create a config and pass it as hf-config-path

```bash
# If your model is not supported by HuggingFace you can manually provide a HuggingFace compatible config path
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M \
   --tokenizer Qwen/Qwen3-0.6B \
   --hf-config-path Qwen/Qwen3-0.6B
```

You can also use the GGUF model directly through the LLM entrypoint:

??? code

      ```python
      from vllm import LLM, SamplingParams

      # In this script, we demonstrate how to pass input to the chat method:
      conversation = [
         {
            "role": "system",
            "content": "You are a helpful assistant",
         },
         {
            "role": "user",
            "content": "Hello",
         },
         {
            "role": "assistant",
            "content": "Hello! How can I assist you today?",
         },
         {
            "role": "user",
            "content": "Write an essay about the importance of higher education.",
         },
      ]

      # Create a sampling params object.
      sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

      # Create an LLM using repo_id:quant_type format.
      llm = LLM(
         model="unsloth/Qwen3-0.6B-GGUF:Q4_K_M",
         tokenizer="Qwen/Qwen3-0.6B",
      )
      # Generate texts from the prompts. The output is a list of RequestOutput objects
      # that contain the prompt, generated text, and other information.
      outputs = llm.chat(conversation, sampling_params)

      # Print the outputs.
      for output in outputs:
         prompt = output.prompt
         generated_text = output.outputs[0].text
         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
      ```
