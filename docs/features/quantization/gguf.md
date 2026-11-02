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

IQ2_XS and IQ4_NL experts use grouped quantized matrix multiplication for
prefill batches at or above `VLLM_GGUF_IQ_MOE_MMQ_MIN_TOKENS` (default: 256).
The general `VLLM_GGUF_MOE_MMQ_MIN_TOKENS` threshold also applies. The IQ path
uses 8-token tiles on ROCm and supports models with more than 256 experts and
IQ4_NL input widths such as 320. Smaller batches and decode keep MMVQ.
Set `VLLM_GGUF_IQ_MOE_MMQ=0` before starting the server to use MMVQ for these
experts during prefill as well.

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
