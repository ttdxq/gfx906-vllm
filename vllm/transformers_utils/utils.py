# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.metadata
import json
import os
import struct
from functools import cache
from os import PathLike
from pathlib import Path
from typing import Any

import gguf
from gguf import GGMLQuantizationType

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

if not hasattr(gguf, "__version__"):
    try:
        gguf.__version__ = importlib.metadata.version("gguf")
    except importlib.metadata.PackageNotFoundError:
        gguf.__version__ = "0.0.0"


def is_s3(model_or_path: str) -> bool:
    return model_or_path.lower().startswith("s3://")


def is_gcs(model_or_path: str) -> bool:
    return model_or_path.lower().startswith("gs://")


def is_cloud_storage(model_or_path: str) -> bool:
    return is_s3(model_or_path) or is_gcs(model_or_path)


@cache
def check_gguf_file(model: str | PathLike) -> bool:
    """Check if the file is a GGUF model."""
    model = Path(model)
    if not model.is_file():
        return False
    elif model.suffix == ".gguf":
        return True

    try:
        with model.open("rb") as f:
            header = f.read(4)

        return header == b"GGUF"
    except Exception as e:
        logger.debug("Error reading file %s: %s", model, e)
        return False


@cache
def is_remote_gguf(model: str | Path) -> bool:
    """Check if the model is a remote GGUF model."""
    model = str(model)
    if is_cloud_storage(model) or model.startswith(("http://", "https://")):
        return False

    if "/" not in model or ":" not in model:
        return False

    quant_type = model.rsplit(":", 1)[1]
    return is_valid_gguf_quant_type(quant_type) or is_nonstandard_gguf_quant_type(
        quant_type
    )


def is_valid_gguf_quant_type(gguf_quant_type: str) -> bool:
    """Check if the quant type is a valid GGUF quant type."""
    if getattr(GGMLQuantizationType, gguf_quant_type, None) is not None:
        return True

    # GGUF repositories commonly append a size suffix to the base type.
    for suffix in ("_M", "_S", "_L", "_XL", "_XS", "_XXS"):
        if gguf_quant_type.endswith(suffix):
            base_type = gguf_quant_type[: -len(suffix)]
            if getattr(GGMLQuantizationType, base_type, None) is not None:
                return True
    return False


def is_nonstandard_gguf_quant_type(quant_type: str) -> bool:
    """Return whether a prefixed quant name contains a known GGUF type."""
    if "-" not in quant_type:
        return False
    _, remainder = quant_type.rsplit("-", 1)
    return is_valid_gguf_quant_type(remainder)


def split_remote_gguf(model: str | Path) -> tuple[str, str]:
    """Split the model into repo_id and quant type."""
    model = str(model)
    if is_remote_gguf(model):
        parts = model.rsplit(":", 1)
        return (parts[0], parts[1])
    raise ValueError(
        f"Wrong GGUF model or invalid GGUF quant type: {model}.\n"
        "- It should be in repo_id:quant_type format.\n"
        f"- Valid GGMLQuantizationType values: {GGMLQuantizationType._member_names_}\n"
        "- Dash-prefixed quant types are accepted when the suffix is valid "
        "(e.g. UD-IQ1_S)."
    )


def is_gguf(model: str | Path) -> bool:
    """Check if the model is a GGUF model.

    Args:
        model: Model name, path, or Path object to check.

    Returns:
        True if the model is a GGUF model, False otherwise.
    """
    model = str(model)

    # Check if it's a local GGUF file
    if check_gguf_file(model):
        return True

    # Check if it's a remote GGUF model (repo_id:quant_type format)
    return is_remote_gguf(model)


def modelscope_list_repo_files(
    repo_id: str,
    revision: str | None = None,
    token: str | bool | None = None,
) -> list[str]:
    """List files in a modelscope repo."""
    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token)
    # same as huggingface_hub.list_repo_files
    files = [
        file["Path"]
        for file in api.get_model_files(
            model_id=repo_id, revision=revision, recursive=True
        )
        if file["Type"] == "blob"
    ]
    return files


def _maybe_json_dict(path: str | PathLike) -> dict[str, str]:
    with open(path) as f:
        try:
            return json.loads(f.read())
        except Exception:
            return dict[str, str]()


def _maybe_space_split_dict(path: str | PathLike) -> dict[str, str]:
    parsed_dict = dict[str, str]()
    with open(path) as f:
        for line in f.readlines():
            try:
                model_name, redirect_name = line.strip().split()
                parsed_dict[model_name] = redirect_name
            except Exception:
                pass
    return parsed_dict


@cache
def maybe_model_redirect(model: str) -> str:
    """
    Use model_redirect to redirect the model name to a local folder.

    :param model: hf model name
    :return: maybe redirect to a local folder
    """

    model_redirect_path = envs.VLLM_MODEL_REDIRECT_PATH

    if not model_redirect_path:
        return model

    if not Path(model_redirect_path).exists():
        return model

    redirect_dict = _maybe_json_dict(model_redirect_path) or _maybe_space_split_dict(
        model_redirect_path
    )
    if redirect_model := redirect_dict.get(model):
        logger.info("model redirect: [ %s ] -> [ %s ]", model, redirect_model)
        return redirect_model

    return model


def parse_safetensors_file_metadata(path: str | PathLike) -> dict[str, Any]:
    with open(path, "rb") as f:
        length_of_metadata = struct.unpack("<Q", f.read(8))[0]
        metadata = json.loads(f.read(length_of_metadata).decode("utf-8"))
        return metadata


def convert_model_repo_to_path(model_repo: str) -> str:
    """When VLLM_USE_MODELSCOPE is True convert a model
    repository string to a Path str."""
    if not envs.VLLM_USE_MODELSCOPE or Path(model_repo).exists():
        return model_repo
    from modelscope.utils.file_utils import get_model_cache_root

    return os.path.join(get_model_cache_root(), model_repo)
