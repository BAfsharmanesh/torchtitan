from collections import defaultdict

import torch
import torch.nn as nn

from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._composable.replicate import replicate
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.logging import logger
from torchtitan.parallelisms.parallel_dims import ParallelDims
from torchtitan.parallelisms.utils import check_strided_sharding_enabled


def parallelize_wideresnet(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    if parallel_dims.tp_enabled:
        if (
            job_config.experimental.enable_async_tensor_parallel
            and not job_config.training.compile
        ):
            raise RuntimeError("Async TP requires --training.compile")
        apply_tp(
            model,
            world_mesh["tp"],
            loss_parallel=parallel_dims.loss_parallel_enabled,
            enable_float8=job_config.float8.enable_float8_linear,
            enable_async_tp=job_config.experimental.enable_async_tensor_parallel,
        )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if job_config.training.compile:
        if job_config.model.norm_type == "fused_rmsnorm":
            raise NotImplementedError(
                "fused_rmsnorm is not compatible with torch.compile yet. "
                "Please use rmsnorm or layernorm."
            )
        apply_compile(model)

    if (
        parallel_dims.dp_shard_enabled
    ):  # apply FSDP or HSDP, potentially with Context Parallel

        # TODO: instead of flattening the mesh twice, we could've done in a batter way:
        # dp_mesh = world_mesh["dp_cp"] if parallel_dims.cp_enabled else world_mesh["dp"]
        # However, this leads to an error in `DeviceMesh.__get_item__` which I believe is
        # a bug in DeviceMesh. We should fix it and then use the above line.
        dp_mesh_dim_names = (
            ("dp_replicate", "dp_shard")
            if parallel_dims.dp_replicate_enabled
            else ("dp",)
        )
        # note that mesh can only be flattened from the finest-grained mesh dimensions
        dp_mesh = (
            world_mesh[(*dp_mesh_dim_names, "cp")]._flatten("dp_cp")
            if parallel_dims.cp_enabled
            else world_mesh[dp_mesh_dim_names]
        )

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            tp_enabled=parallel_dims.tp_enabled,
            pp_enabled=parallel_dims.pp_enabled,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")
    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            world_mesh,
            enable_compile=job_config.training.compile,
            enable_compiled_autograd=job_config.experimental.enable_compiled_autograd,
        )
    
def apply_ddp():
    pass

def apply_fsdp():
    pass

def apply_ac():
    pass

def apply_tp():
    pass

def apply_compile():
    pass