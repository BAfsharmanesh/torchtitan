import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import DeviceMesh
from torch.distributed._tensor import DTensor, Shard


def run(rank, world_size):
    torch.cuda.set_device(rank)
    # Initialize the tensor on the process with rank 0
    if rank == 0:
        tensor = torch.arange(64, dtype=torch.float32).reshape(8, 8).cuda(rank)
    else:
        tensor = torch.empty(8, 8, dtype=torch.float32).cuda(rank)  # Placeholder tensor

    # Define the device mesh with 4 devices in a 2x2 grid
    device_mesh = DeviceMesh("cuda", torch.arange(world_size).reshape(2, 2))

    # Initialize the tensor across devices with a 2x2 sharding
    # Sharding is done along each axis to split into 4 chunks (2x2 layout)
    if rank == 0:
        print(f"Original tensor on rank {rank}:\n{tensor}")

    # Broadcast and shard the tensor across the device mesh
    tensor_dtensor = DTensor.from_local(
        tensor, device_mesh, placements=[Shard(0), Shard(1)]
    )

    # Access the local shard for this rank
    local_shard = tensor_dtensor.to_local()

    # Print local shard information
    print(f"Rank {rank} local shard:\n{local_shard}")


def init_process(rank, size, fn, backend="nccl"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    device_mesh = DeviceMesh("cuda", torch.arange(world_size).reshape(2, 2))
    fn(rank, size)
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 4
    mp.spawn(init_process, args=(world_size,run), nprocs=world_size, join=True)
