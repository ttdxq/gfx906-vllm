# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF utility functions."""

import hashlib
from pathlib import Path

import gguf
from gguf.constants import Keys, VisionProjectorType
from transformers import (
    Gemma3Config,
    PreTrainedTokenizerFast,
    PretrainedConfig,
    SiglipVisionConfig,
)
from transformers.integrations.ggml import convert_gguf_tokenizer
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_TO_TRANSFORMERS_MAPPING,
    _gguf_parse_value,
)

from vllm.logger import init_logger

from .repo_utils import list_filtered_repo_files

logger = init_logger(__name__)


def detect_gguf_multimodal(model: str) -> Path | None:
    """Check if GGUF model has multimodal projector file.

    Args:
        model: Model path string

    Returns:
        Path to mmproj file if found, None otherwise
    """
    if not model.endswith(".gguf"):
        return None

    try:
        model_path = Path(model)
        if not model_path.is_file():
            return None

        model_dir = model_path.parent
        mmproj_patterns = ["mmproj.gguf", "mmproj-*.gguf", "*mmproj*.gguf"]
        for pattern in mmproj_patterns:
            mmproj_files = list(model_dir.glob(pattern))
            if mmproj_files:
                return mmproj_files[0]
        return None
    except Exception:
        return None


def extract_vision_config_from_gguf(mmproj_path: str) -> "SiglipVisionConfig | None":
    """Extract vision config parameters from mmproj.gguf metadata.

    Reads vision encoder configuration from GGUF metadata fields using
    standardized GGUF constants. Automatically detects the projector type
    (e.g., gemma3, llama4) and applies model-specific parameters accordingly.

    The function extracts standard CLIP vision parameters from GGUF metadata
    and applies projector-type-specific customizations. For unknown projector
    types, it uses safe defaults from SiglipVisionConfig.

    Args:
        mmproj_path: Path to mmproj.gguf file (str or Path)

    Returns:
        SiglipVisionConfig if extraction succeeds, None if any required
        field is missing from the GGUF metadata

    Raises:
        Exception: Exceptions from GGUF reading (file not found, corrupted
            file, etc.) propagate directly from gguf.GGUFReader
    """
    reader = gguf.GGUFReader(str(mmproj_path))

    # Detect projector type to apply model-specific parameters
    projector_type = None
    projector_type_field = reader.get_field(Keys.Clip.PROJECTOR_TYPE)
    if projector_type_field:
        try:
            projector_type = bytes(projector_type_field.parts[-1]).decode("utf-8")
        except (AttributeError, UnicodeDecodeError) as e:
            logger.warning("Failed to decode projector type from GGUF: %s", e)

    # Map GGUF field constants to SiglipVisionConfig parameters.
    # Uses official GGUF constants from gguf-py for standardization.
    # Format: {gguf_constant: (param_name, dtype)}
    VISION_CONFIG_FIELDS = {
        Keys.ClipVision.EMBEDDING_LENGTH: ("hidden_size", int),
        Keys.ClipVision.FEED_FORWARD_LENGTH: ("intermediate_size", int),
        Keys.ClipVision.BLOCK_COUNT: ("num_hidden_layers", int),
        Keys.ClipVision.Attention.HEAD_COUNT: ("num_attention_heads", int),
        Keys.ClipVision.IMAGE_SIZE: ("image_size", int),
        Keys.ClipVision.PATCH_SIZE: ("patch_size", int),
        Keys.ClipVision.Attention.LAYERNORM_EPS: ("layer_norm_eps", float),
    }

    # Extract and validate all required fields
    config_params = {}
    for gguf_key, (param_name, dtype) in VISION_CONFIG_FIELDS.items():
        field = reader.get_field(gguf_key)
        if field is None:
            logger.warning(
                "Missing required vision config field '%s' in mmproj.gguf",
                gguf_key,
            )
            return None
        # Extract scalar value from GGUF field and convert to target type
        config_params[param_name] = dtype(field.parts[-1])

    # Apply model-specific parameters based on projector type
    if projector_type == VisionProjectorType.GEMMA3:
        # Gemma3 doesn't use the vision pooling head (multihead attention)
        # This is a vLLM-specific parameter used in SiglipVisionTransformer
        config_params["vision_use_head"] = False
        logger.info("Detected Gemma3 projector, disabling vision pooling head")
    # Add other projector-type-specific customizations here as needed
    # elif projector_type == VisionProjectorType.LLAMA4:
    #     config_params["vision_use_head"] = ...

    # Create config with extracted parameters
    # Note: num_channels and attention_dropout use SiglipVisionConfig defaults
    # (3 and 0.0 respectively) which are correct for all models
    config = SiglipVisionConfig(**config_params)

    if projector_type:
        logger.info(
            "Extracted vision config from mmproj.gguf (projector_type: %s)",
            projector_type,
        )
    else:
        logger.info("Extracted vision config from mmproj.gguf metadata")

    return config


def _read_gguf_scalar(
    reader: gguf.GGUFReader,
    key: str,
    default: object | None = None,
) -> object | None:
    field = reader.get_field(key)
    if field is None:
        return default

    value = field.contents()
    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
        return value
    return value


def qwen35_gguf_config_dict(model: str) -> dict | None:
    """Build a Qwen3.5 config dict from qwen35 GGUF metadata."""
    try:
        reader = gguf.GGUFReader(str(model))
    except Exception:
        return None

    if _read_gguf_scalar(reader, "general.architecture") != "qwen35":
        return None

    tokens = _read_gguf_scalar(reader, "tokenizer.ggml.tokens", [])
    linear_key_head_dim = int(
        _read_gguf_scalar(reader, "qwen35.ssm.state_size", 128)
    )
    linear_value_head_dim = int(
        _read_gguf_scalar(reader, "qwen35.ssm.state_size", 128)
    )
    linear_num_key_heads = int(
        _read_gguf_scalar(reader, "qwen35.ssm.group_count")
    )
    qkv_tensor = next(
        (tensor for tensor in reader.tensors if tensor.name == "blk.0.attn_qkv.weight"),
        None,
    )
    if qkv_tensor is None:
        return None
    qkv_dim = int(qkv_tensor.shape[1])
    linear_num_value_heads = int(
        (qkv_dim - 2 * linear_num_key_heads * linear_key_head_dim)
        // linear_value_head_dim
    )
    num_hidden_layers = int(_read_gguf_scalar(reader, "qwen35.block_count"))
    tensor_names = {tensor.name for tensor in reader.tensors}
    layer_types = [
        (
            "linear_attention"
            if f"blk.{idx}.attn_qkv.weight" in tensor_names
            else "full_attention"
        )
        for idx in range(num_hidden_layers)
    ]

    text_config = {
        "model_type": "qwen3_5_text",
        "vocab_size": len(tokens),
        "hidden_size": int(_read_gguf_scalar(reader, "qwen35.embedding_length")),
        "intermediate_size": int(
            _read_gguf_scalar(reader, "qwen35.feed_forward_length")
        ),
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": int(
            _read_gguf_scalar(reader, "qwen35.attention.head_count")
        ),
        "num_key_value_heads": int(
            _read_gguf_scalar(reader, "qwen35.attention.head_count_kv")
        ),
        "max_position_embeddings": int(
            _read_gguf_scalar(reader, "qwen35.context_length")
        ),
        "rms_norm_eps": float(
            _read_gguf_scalar(reader, "qwen35.attention.layer_norm_rms_epsilon")
        ),
        "head_dim": int(_read_gguf_scalar(reader, "qwen35.attention.key_length")),
        "linear_key_head_dim": linear_key_head_dim,
        "linear_value_head_dim": linear_value_head_dim,
        "linear_conv_kernel_dim": int(
            _read_gguf_scalar(reader, "qwen35.ssm.conv_kernel", 4)
        ),
        "linear_num_key_heads": linear_num_key_heads,
        "linear_num_value_heads": linear_num_value_heads,
        "full_attention_interval": int(
            _read_gguf_scalar(reader, "qwen35.full_attention_interval", 4)
        ),
        "layer_types": layer_types,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": float(_read_gguf_scalar(reader, "qwen35.rope.freq_base")),
            "mrope_section": _read_gguf_scalar(
                reader, "qwen35.rope.dimension_sections", None
            ),
            "mrope_interleaved": True,
        },
        "bos_token_id": _read_gguf_scalar(reader, "tokenizer.ggml.bos_token_id"),
        "eos_token_id": _read_gguf_scalar(reader, "tokenizer.ggml.eos_token_id"),
        "pad_token_id": _read_gguf_scalar(reader, "tokenizer.ggml.padding_token_id"),
        "tie_word_embeddings": not any(
            tensor.name == "output.weight" for tensor in reader.tensors
        ),
    }

    config_dict = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5",
        "text_config": text_config,
    }
    logger.info("Built Qwen3.5 config from qwen35 GGUF metadata: %s", model)
    return config_dict


def _qwen35_gguf_tokenizer_config(
    reader: gguf.GGUFReader,
) -> tuple[dict, dict] | None:
    if _read_gguf_scalar(reader, "general.architecture") != "qwen35":
        return None

    parsed = {k: {} for k in GGUF_TO_TRANSFORMERS_MAPPING}
    for gguf_key, field in reader.fields.items():
        split = gguf_key.split(".")
        prefix = split[0]
        config_key = ".".join(split[1:])
        value = [
            _gguf_parse_value(field.parts[data_index], field.types)
            for data_index in field.data
        ]
        if len(value) == 1:
            value = value[0]

        for parameter, parameter_renames in GGUF_TO_TRANSFORMERS_MAPPING.items():
            if prefix in parameter_renames and config_key in parameter_renames[prefix]:
                renamed_config_key = parameter_renames[prefix][config_key]
                if renamed_config_key not in (-1, None):
                    parsed[parameter][renamed_config_key] = value

    tokenizer_dict = parsed["tokenizer"]
    tokenizer_config = parsed["tokenizer_config"]
    tokens = tokenizer_dict.get("tokens")
    if not tokens:
        return None

    for id_key, token_key in (
        ("bos_token_id", "bos_token"),
        ("eos_token_id", "eos_token"),
        ("pad_token_id", "pad_token"),
        ("unk_token_id", "unk_token"),
    ):
        token_id = tokenizer_config.get(id_key)
        if token_id is not None and token_key not in tokenizer_config:
            tokenizer_config[token_key] = tokens[token_id]

    return tokenizer_dict, tokenizer_config


def qwen35_gguf_tokenizer_path(model: str) -> str | None:
    """Materialize a HF tokenizer directory from qwen35 GGUF metadata."""
    try:
        model_path = Path(model)
        reader = gguf.GGUFReader(str(model_path))
    except Exception:
        return None

    tokenizer_parts = _qwen35_gguf_tokenizer_config(reader)
    if tokenizer_parts is None:
        return None

    stat = model_path.stat()
    digest = hashlib.sha256(
        f"{model_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()[:16]
    tokenizer_dir = Path.home() / ".cache" / "vllm" / "qwen35_gguf_tokenizers" / digest
    if (tokenizer_dir / "tokenizer.json").is_file():
        return str(tokenizer_dir)

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dict, tokenizer_config = tokenizer_parts
    tokenizer_object, additional_kwargs = convert_gguf_tokenizer(
        "qwen3",
        tokenizer_dict,
    )
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_object,
        **additional_kwargs,
        **tokenizer_config,
    )
    fast_tokenizer.save_pretrained(tokenizer_dir)
    logger.info("Built Qwen3.5 tokenizer from qwen35 GGUF metadata: %s", model)
    return str(tokenizer_dir)


def maybe_patch_hf_config_from_gguf(
    model: str,
    hf_config: PretrainedConfig,
) -> PretrainedConfig:
    """Patch HF config for GGUF models.

    Applies GGUF-specific patches to HuggingFace config:
    1. For multimodal models: patches architecture and vision config
    2. For all GGUF models: overrides vocab_size from embedding tensor

    This ensures compatibility with GGUF models that have extended
    vocabularies (e.g., Unsloth) where the GGUF file contains more
    tokens than the HuggingFace tokenizer config specifies.

    Args:
        model: Model path string
        hf_config: HuggingFace config to patch in-place

    Returns:
        Updated HuggingFace config
    """
    # Patch multimodal config if mmproj.gguf exists
    mmproj_path = detect_gguf_multimodal(model)
    if mmproj_path is not None:
        vision_config = extract_vision_config_from_gguf(str(mmproj_path))

        # Create HF config for Gemma3 multimodal
        text_config = hf_config.get_text_config()
        is_gemma3 = hf_config.model_type in ("gemma3", "gemma3_text")
        if vision_config is not None and is_gemma3:
            new_hf_config = Gemma3Config.from_text_vision_configs(
                text_config=text_config,
                vision_config=vision_config,
                architectures=["Gemma3ForConditionalGeneration"],
            )
            hf_config = new_hf_config

    if (
        mmproj_path is None
        and getattr(hf_config, "model_type", "") in ("qwen3_5", "qwen3_5_text")
        and getattr(hf_config, "architectures", None) == ["Qwen3_5ForCausalLM"]
    ):
        hf_config.architectures = ["Qwen3_5ForConditionalGeneration"]
    if getattr(hf_config, "model_type", "") in ("qwen3_5", "qwen3_5_text"):
        rope_parameters = getattr(hf_config, "rope_parameters", None)
        if (
            isinstance(rope_parameters, dict)
            and "theta" in rope_parameters
        ):
            rope_parameters["rope_theta"] = rope_parameters.pop("theta")

    return hf_config


def get_gguf_file_path_from_hf(
    repo_id: str | Path,
    quant_type: str,
    revision: str | None = None,
) -> str:
    """Get the GGUF file path from HuggingFace Hub based on repo_id and quant_type.

    Args:
        repo_id: The HuggingFace repository ID (e.g., "Qwen/Qwen3-0.6B")
        quant_type: The quantization type (e.g., "Q4_K_M", "F16")
        revision: Optional revision/branch name

    Returns:
        The path to the GGUF file on HuggingFace Hub (e.g., "filename.gguf"),
    """
    repo_id = str(repo_id)
    gguf_patterns = [
        f"*-{quant_type}.gguf",
        f"*-{quant_type}-*.gguf",
        f"*/*-{quant_type}.gguf",
        f"*/*-{quant_type}-*.gguf",
    ]
    matching_files = list_filtered_repo_files(
        repo_id,
        allow_patterns=gguf_patterns,
        revision=revision,
    )

    if len(matching_files) == 0:
        raise ValueError(
            "Could not find GGUF file for repo %s with quantization %s.",
            repo_id,
            quant_type,
        )

    # Sort to ensure consistent ordering (prefer non-sharded files)
    matching_files.sort(key=lambda x: (x.count("-"), x))
    gguf_filename = matching_files[0]
    return gguf_filename
