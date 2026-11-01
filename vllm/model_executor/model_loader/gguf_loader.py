# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Generator
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.gguf_adapters import (
    get_mmproj_file,
    get_weights_adapter,
)
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.model_executor.model_loader.weight_utils import (
    download_gguf,
    get_gguf_extra_tensor_names,
    get_gguf_weight_type_map,
    gguf_quant_weights_iterator,
)
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


class GGUFModelLoader(BaseModelLoader):
    """
    Model loader that can load GGUF files. This is useful for loading models
    that are quantized with GGUF and saved in the GGUF format. This loader
    supports loading both full models and sharded models.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    @staticmethod
    def _get_mmproj_file(
        model_config: ModelConfig, model_name_or_path: str | None = None
    ) -> Path | None:
        return get_mmproj_file(model_config, model_name_or_path)

    def _prepare_weights(self, model_config: ModelConfig):
        model_name_or_path = model_config.model
        if os.path.isfile(model_name_or_path):
            return model_name_or_path
        # for raw HTTPS link
        if model_name_or_path.startswith(
            ("http://", "https://")
        ) and model_name_or_path.endswith(".gguf"):
            return hf_hub_download(url=model_name_or_path)
        # repo id/filename.gguf
        if "/" in model_name_or_path and model_name_or_path.endswith(".gguf"):
            repo_id, filename = model_name_or_path.rsplit("/", 1)
            return hf_hub_download(repo_id=repo_id, filename=filename)
        # repo_id:quant_type
        elif "/" in model_name_or_path and ":" in model_name_or_path:
            repo_id, quant_type = model_name_or_path.rsplit(":", 1)
            return download_gguf(
                repo_id,
                quant_type,
                cache_dir=self.load_config.download_dir,
                revision=model_config.revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )

        raise ValueError(
            f"Unrecognised GGUF reference: {model_name_or_path} "
            "(expected local file, raw URL, <repo_id>/<filename>.gguf, "
            "or <repo_id>:<quant_type>)"
        )

    def _get_gguf_weights_map(self, model_config: ModelConfig):
        """
        GGUF uses this naming convention for their tensors from HF checkpoint:
        `blk.N.BB.weight` and `blk.N.BB.bias`
        where N signifies the block number of a layer, and BB signifies the
        attention/mlp layer components.
        See "Standardized tensor names" in
        https://github.com/ggerganov/ggml/blob/master/docs/gguf.md for details.
        """
        return get_weights_adapter(model_config).build_name_map(model_config)

    def _get_gguf_weight_type(
        self,
        model_config: ModelConfig,
        model_name_or_path: str,
        gguf_to_hf_name_map: dict[str, str],
    ) -> dict[str, str]:
        weight_type_map = get_gguf_weight_type_map(
            model_name_or_path, gguf_to_hf_name_map
        )
        mmproj_file = self._get_mmproj_file(model_config, model_name_or_path)
        if mmproj_file is not None:
            logger.info("Loading extra mm_proj weights from %s...", mmproj_file)
            mm_proj_weight_type_map = get_gguf_weight_type_map(
                mmproj_file, gguf_to_hf_name_map
            )
            weight_type_map.update(mm_proj_weight_type_map)
        return weight_type_map

    def _get_weights_iterator(
        self,
        model_config: ModelConfig,
        model_name_or_path: str,
        gguf_to_hf_name_map: dict[str, str],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """
        Iterate over GGUF model weights, loading from both main model file and
        mmproj.gguf for multimodal models.  Family-specific tensor transforms
        (fused-tensor splits, layout permutations) are applied by the matched
        weights adapter.
        """
        adapter = get_weights_adapter(model_config)
        mmproj_file = self._get_mmproj_file(model_config, model_name_or_path)
        if mmproj_file is not None:
            # Load mm_proj (mm_encoder + projector) for multimodal weights
            mmproj_weights = gguf_quant_weights_iterator(
                mmproj_file, gguf_to_hf_name_map
            )
            yield from adapter.transform_mmproj_weights(
                mmproj_weights, model_config, gguf_to_hf_name_map, mmproj_file
            )

        yield from adapter.transform_weights(
            gguf_quant_weights_iterator(model_name_or_path, gguf_to_hf_name_map),
            model_config,
        )

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        local_model_path = self._prepare_weights(model_config)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
        model.load_weights(
            self._get_weights_iterator(model_config, local_model_path, gguf_weights_map)
        )

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> nn.Module:
        device_config = vllm_config.device_config
        local_model_path = self._prepare_weights(model_config)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
        # we can only know if tie word embeddings after mapping weights
        extra_tensor_names = get_gguf_extra_tensor_names(
            local_model_path, gguf_weights_map
        )
        if (
            "lm_head.weight" in extra_tensor_names
            or "language_model.lm_head.weight" in extra_tensor_names
        ):
            model_config.hf_config.update({"tie_word_embeddings": True})

        weight_type_map = self._get_gguf_weight_type(
            model_config, local_model_path, gguf_weights_map
        )
        # filter out unquantized modules to skip
        unquant_names = [
            name.removesuffix(".weight")
            for name, weight_type in weight_type_map.items()
            if weight_type in ("F32", "F16", "BF16") and name.endswith(".weight")
        ]
        logger.debug(
            "GGUF unquantized modules: %s",
            unquant_names,
        )
        vllm_config.quant_config.unquantized_modules.extend(unquant_names)

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                # `prefix` is part of the loader protocol (see BaseModelLoader);
                # forward it so draft models loaded via spec-decode keep their
                # compile tag, matching the other loaders.
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config, prefix=prefix
                )
            self.load_weights(model, model_config)

            process_weights_after_loading(model, model_config, target_device)
        return model
