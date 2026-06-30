from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Generic, Literal, TypeVar

import torch

from vllm.config.parallel import ParallelConfig

try:
    from vllm.config.weight_transfer import WeightTransferConfig
except ModuleNotFoundError:
    WeightTransferConfig = Any

TInitInfo = TypeVar("TInitInfo", bound="WeightTransferInitInfo")
TUpdateInfo = TypeVar("TUpdateInfo", bound="WeightTransferUpdateInfo")


@dataclass
class WeightTransferInitInfo(ABC):  # noqa: B024
    pass


@dataclass
class WeightTransferUpdateInfo(ABC):  # noqa: B024
    _: KW_ONLY
    update_kind: Literal["dense", "sparse_flat"] = "dense"
    num_updates_list: list[int] | None = None

    def __post_init__(self) -> None:
        if self.update_kind not in ("dense", "sparse_flat"):
            raise ValueError(f"Unsupported update_kind: {self.update_kind}")
        if self.update_kind == "dense":
            if self.num_updates_list is not None:
                raise ValueError(
                    "Sparse metadata is only supported for `update_kind='sparse_flat'`"
                )
            return

        if self.num_updates_list is None:
            raise ValueError("`num_updates_list` is required for sparse updates")
        if len(self.num_updates_list) == 0:
            raise ValueError("`num_updates_list` cannot be empty for sparse updates")
        if any(num_updates < 0 for num_updates in self.num_updates_list):
            raise ValueError("Sparse `num_updates_list` entries must be non-negative")

        names = getattr(self, "names", None)
        if names is not None and len(self.num_updates_list) != len(names):
            raise ValueError(
                f"`num_updates_list` should be of the same size as `names`: "
                f"got {len(self.num_updates_list)} and {len(names)}"
            )


@dataclass
class SparseWeightPatch:
    name: str
    indices: torch.Tensor
    values: torch.Tensor


@dataclass
class WeightTransferInitRequest:
    init_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class WeightTransferUpdateRequest:
    update_info: dict[str, Any] = field(default_factory=dict)


class WeightTransferEngine(ABC, Generic[TInitInfo, TUpdateInfo]):
    init_info_cls: type[TInitInfo]
    update_info_cls: type[TUpdateInfo]

    def __init__(
        self,
        config: WeightTransferConfig,
        parallel_config: ParallelConfig,
        model: torch.nn.Module,
    ) -> None:
        self.config = config
        self.parallel_config = parallel_config
        self.model = model

    def parse_init_info(self, init_dict: dict[str, Any]) -> TInitInfo:
        try:
            return self.init_info_cls(**init_dict)
        except TypeError as exc:
            raise ValueError(
                f"Invalid init_info for {self.__class__.__name__}: {exc}"
            ) from exc

    def parse_update_info(self, update_dict: dict[str, Any]) -> TUpdateInfo:
        try:
            return self.update_info_cls(**update_dict)
        except TypeError as exc:
            raise ValueError(
                f"Invalid update_info for {self.__class__.__name__}: {exc}"
            ) from exc

    @abstractmethod
    def init_transfer_engine(self, init_info: TInitInfo) -> None:
        raise NotImplementedError

    @abstractmethod
    def receive_weights(
        self,
        update_info: TUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        raise NotImplementedError

    def receive_sparse_weights(
        self,
        update_info: TUpdateInfo,
        apply_patches: Callable[[list[SparseWeightPatch]], None],
    ) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support sparse weight updates"
        )

    @abstractmethod
    def shutdown(self) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def trainer_send_sparse_weights(
        _iterator: Iterator[SparseWeightPatch],
        _trainer_args: dict[str, Any] | Any,
    ) -> None:
        raise NotImplementedError("Sparse weight updates are not supported")
