# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torchtitan.parallelisms.parallel_dims import ParallelDims
from torchtitan.parallelisms.parallelize_llama import parallelize_llama
from torchtitan.parallelisms.parallelize_moe import parallelize_moe
from torchtitan.parallelisms.pipeline_llama import pipeline_llama
from torchtitan.parallelisms.pipeline_moe import pipeline_moe
from torchtitan.parallelisms.pipeline_wideresnet import pipeline_wideresnet
from torchtitan.parallelisms.parallelize_wideresnet import parallelize_wideresnet


__all__ = [
    "models_parallelize_fns",
    "models_pipelining_fns",
    "ParallelDims",
]

models_parallelize_fns = {
    "llama2": parallelize_llama,
    "llama3": parallelize_llama,
    "moe": parallelize_moe,
    "wideresnet": parallelize_wideresnet,
}
models_pipelining_fns = {
    "llama2": pipeline_llama,
    "llama3": pipeline_llama,
    "moe": pipeline_moe,
    "wideresnet": pipeline_wideresnet,
}
