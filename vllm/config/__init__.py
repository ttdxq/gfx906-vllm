from importlib import import_module


_EXPORTS = {
    # attention
    "AttentionConfig": "vllm.config.attention",
    # cache
    "CacheConfig": "vllm.config.cache",
    # compilation
    "CompilationConfig": "vllm.config.compilation",
    "CompilationMode": "vllm.config.compilation",
    "CUDAGraphMode": "vllm.config.compilation",
    "PassConfig": "vllm.config.compilation",
    # device
    "DeviceConfig": "vllm.config.device",
    # transfer/events
    "ECTransferConfig": "vllm.config.ec_transfer",
    "KVEventsConfig": "vllm.config.kv_events",
    "KVTransferConfig": "vllm.config.kv_transfer",
    # load / lora
    "LoadConfig": "vllm.config.load",
    "LoRAConfig": "vllm.config.lora",
    # model
    "ModelConfig": "vllm.config.model",
    "iter_architecture_defaults": "vllm.config.model",
    "try_match_architecture_defaults": "vllm.config.model",
    # multimodal / observability
    "MultiModalConfig": "vllm.config.multimodal",
    "ObservabilityConfig": "vllm.config.observability",
    # parallel
    "EPLBConfig": "vllm.config.parallel",
    "ParallelConfig": "vllm.config.parallel",
    # pooler / scheduler / speculative / speech / structured
    "PoolerConfig": "vllm.config.pooler",
    "SchedulerConfig": "vllm.config.scheduler",
    "SpeculativeConfig": "vllm.config.speculative",
    "SpeechToTextConfig": "vllm.config.speech_to_text",
    "StructuredOutputsConfig": "vllm.config.structured_outputs",
    # utils
    "ConfigType": "vllm.config.utils",
    "SupportsMetricsInfo": "vllm.config.utils",
    "config": "vllm.config.utils",
    "get_attr_docs": "vllm.config.utils",
    "is_init_field": "vllm.config.utils",
    "update_config": "vllm.config.utils",
    # vllm
    "VllmConfig": "vllm.config.vllm",
    "get_cached_compilation_config": "vllm.config.vllm",
    "get_current_vllm_config": "vllm.config.vllm",
    "get_current_vllm_config_or_none": "vllm.config.vllm",
    "set_current_vllm_config": "vllm.config.vllm",
    "get_layers_from_vllm_config": "vllm.config.vllm",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
