# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF utility functions."""

import hashlib
import re
from pathlib import Path

import gguf
from gguf.constants import Keys, VisionProjectorType
from transformers import (
    Gemma3Config,
    PretrainedConfig,
    PreTrainedTokenizerFast,
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


def _gguf_model_family(path: Path) -> str:
    tokens = re.split(r"[-_.]+", path.stem.lower())
    if tokens and tokens[0] == "mmproj":
        tokens = tokens[1:]

    family_tokens = []
    for idx, token in enumerate(tokens):
        if token == "mmproj":
            break
        if token == "ud" and idx + 1 < len(tokens):
            next_token = tokens[idx + 1]
            if re.fullmatch(r"(?:i?q|tq)\d+", next_token):
                break
        if re.fullmatch(r"(?:i?q|tq)\d+|(?:bf|f|fp)\d+", token):
            break
        family_tokens.append(token)
    return "-".join(family_tokens)


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
        mmproj_files = sorted(
            path for path in model_dir.glob("*.gguf") if "mmproj" in path.stem.lower()
        )
        model_family = _gguf_model_family(model_path)
        for mmproj_file in mmproj_files:
            if _gguf_model_family(mmproj_file) == model_family:
                return mmproj_file

        model_families = {
            _gguf_model_family(path)
            for path in model_dir.glob("*.gguf")
            if "mmproj" not in path.stem.lower()
        }
        if len(mmproj_files) == 1 and model_family and model_families == {model_family}:
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
    config.projector_type = projector_type
    projection_dim = _read_gguf_scalar(reader, "clip.vision.projection_dim")
    if projection_dim is not None:
        config.projection_dim = int(projection_dim)
    spatial_merge_size = _read_gguf_scalar(reader, "clip.vision.spatial_merge_size")
    if spatial_merge_size is not None:
        config.spatial_merge_size = int(spatial_merge_size)

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
    """Build a Qwen3.5 config dict from qwen35/qwen35moe GGUF metadata."""
    try:
        reader = gguf.GGUFReader(str(model))
    except Exception:
        return None

    arch = _read_gguf_scalar(reader, "general.architecture")
    if arch not in ("qwen35", "qwen35moe"):
        return None
    prefix = str(arch)
    is_moe = arch == "qwen35moe"

    tokens = _read_gguf_scalar(reader, "tokenizer.ggml.tokens", [])
    linear_key_head_dim = int(
        _read_gguf_scalar(reader, f"{prefix}.ssm.state_size", 128)
    )
    linear_value_head_dim = int(
        _read_gguf_scalar(reader, f"{prefix}.ssm.state_size", 128)
    )
    linear_num_key_heads = int(_read_gguf_scalar(reader, f"{prefix}.ssm.group_count"))
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
    num_nextn_predict_layers = int(
        _read_gguf_scalar(reader, f"{prefix}.nextn_predict_layers", 0)
    )
    num_hidden_layers = int(_read_gguf_scalar(reader, f"{prefix}.block_count"))
    # GGUF block_count includes the optional MTP/nextn blocks for both dense
    # and MoE Qwen3.5 models.  They are not part of the causal LM backbone.
    num_hidden_layers -= num_nextn_predict_layers
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
        "model_type": "qwen3_5_moe_text" if is_moe else "qwen3_5_text",
        "vocab_size": len(tokens),
        "hidden_size": int(_read_gguf_scalar(reader, f"{prefix}.embedding_length")),
        "hidden_act": "silu",
        "intermediate_size": int(
            _read_gguf_scalar(reader, f"{prefix}.feed_forward_length", 0)
        ),
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": int(
            _read_gguf_scalar(reader, f"{prefix}.attention.head_count")
        ),
        "num_key_value_heads": int(
            _read_gguf_scalar(reader, f"{prefix}.attention.head_count_kv")
        ),
        "max_position_embeddings": int(
            _read_gguf_scalar(reader, f"{prefix}.context_length")
        ),
        "rms_norm_eps": float(
            _read_gguf_scalar(reader, f"{prefix}.attention.layer_norm_rms_epsilon")
        ),
        "head_dim": int(_read_gguf_scalar(reader, f"{prefix}.attention.key_length")),
        "linear_key_head_dim": linear_key_head_dim,
        "linear_value_head_dim": linear_value_head_dim,
        "linear_conv_kernel_dim": int(
            _read_gguf_scalar(reader, f"{prefix}.ssm.conv_kernel", 4)
        ),
        "linear_num_key_heads": linear_num_key_heads,
        "linear_num_value_heads": linear_num_value_heads,
        "full_attention_interval": int(
            _read_gguf_scalar(reader, f"{prefix}.full_attention_interval", 4)
        ),
        "layer_types": layer_types,
        "num_nextn_predict_layers": num_nextn_predict_layers,
        "mtp_num_hidden_layers": num_nextn_predict_layers,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": float(_read_gguf_scalar(reader, f"{prefix}.rope.freq_base")),
            "mrope_section": _read_gguf_scalar(
                reader, f"{prefix}.rope.dimension_sections", None
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
    if is_moe:
        text_config.update(
            {
                "moe_intermediate_size": int(
                    _read_gguf_scalar(reader, f"{prefix}.expert_feed_forward_length")
                ),
                "shared_expert_intermediate_size": int(
                    _read_gguf_scalar(
                        reader, f"{prefix}.expert_shared_feed_forward_length", 0
                    )
                ),
                "num_experts_per_tok": int(
                    _read_gguf_scalar(reader, f"{prefix}.expert_used_count")
                ),
                "num_experts": int(_read_gguf_scalar(reader, f"{prefix}.expert_count")),
                "num_nextn_predict_layers": int(num_nextn_predict_layers),
                "mtp_num_hidden_layers": int(num_nextn_predict_layers),
                "norm_topk_prob": True,
            }
        )

    config_dict = {
        "architectures": [
            "Qwen3_5MoeForConditionalGeneration" if is_moe else "Qwen3_5ForCausalLM"
        ],
        "model_type": "qwen3_5_moe" if is_moe else "qwen3_5",
        "text_config": text_config,
    }
    if is_moe:
        config_dict.update(
            {
                key: value
                for key, value in text_config.items()
                if key
                in {
                    "hidden_size",
                    "hidden_act",
                    "moe_intermediate_size",
                    "shared_expert_intermediate_size",
                    "num_experts_per_tok",
                    "num_experts",
                    "norm_topk_prob",
                    "rms_norm_eps",
                }
            }
        )
    logger.info("Built Qwen3.5 config from %s GGUF metadata: %s", arch, model)
    return config_dict


def _qwen35_gguf_tokenizer_config(
    reader: gguf.GGUFReader,
) -> tuple[dict, dict] | None:
    if _read_gguf_scalar(reader, "general.architecture") not in (
        "qwen35",
        "qwen35moe",
    ):
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


def gguf_multimodal_processor_repo(model: str) -> str | None:
    """Return the original HF repo needed for multimodal GGUF processing."""
    if detect_gguf_multimodal(model) is None:
        return None

    try:
        reader = gguf.GGUFReader(str(model))
    except Exception:
        return None

    repo_url = _read_gguf_scalar(reader, "general.base_model.0.repo_url")
    if not isinstance(repo_url, str):
        return None

    prefix = "https://huggingface.co/"
    if not repo_url.startswith(prefix):
        return None
    repo_id = repo_url.removeprefix(prefix).strip("/").split("/tree/", 1)[0]
    return repo_id or None


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

        is_qwen35 = hf_config.model_type in (
            "qwen3_5",
            "qwen3_5_text",
            "qwen3_5_moe",
            "qwen3_5_moe_text",
        )
        if vision_config is not None and is_qwen35:
            qwen_vision_config = hf_config.vision_config
            qwen_vision_config.depth = vision_config.num_hidden_layers
            qwen_vision_config.hidden_size = vision_config.hidden_size
            qwen_vision_config.intermediate_size = vision_config.intermediate_size
            qwen_vision_config.num_heads = vision_config.num_attention_heads
            qwen_vision_config.patch_size = vision_config.patch_size
            qwen_vision_config.num_position_embeddings = (
                vision_config.image_size // vision_config.patch_size
            ) ** 2
            if hasattr(vision_config, "spatial_merge_size"):
                qwen_vision_config.spatial_merge_size = vision_config.spatial_merge_size
            if hasattr(vision_config, "projection_dim"):
                qwen_vision_config.out_hidden_size = vision_config.projection_dim
            hf_config.architectures = [
                "Qwen3_5MoeForConditionalGeneration"
                if "moe" in hf_config.model_type
                else "Qwen3_5ForConditionalGeneration"
            ]

    if getattr(hf_config, "model_type", "") in ("qwen3_5", "qwen3_5_text"):
        rope_parameters = getattr(hf_config, "rope_parameters", None)
        if isinstance(rope_parameters, dict) and "theta" in rope_parameters:
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
