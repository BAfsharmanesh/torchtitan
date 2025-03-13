import subprocess
import os
from train import main, JobConfig
from typing import List, Tuple, Dict
import traceback

def get_command(
    n_process: int,
    model: str,
    batch_size: int,
    tp_degree: int,
    pp_degree: int,
    flavor: str,
) -> Tuple[List[str], List[str]]:
    dec = f"{model} {flavor} training"
    command = [
        "torchrun",
        "--nproc_per_node",
        str(n_process),
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        "localhost:0",
        "--local-ranks-filter",
        "0",
        "--role",
        "rank",
        "--tee",
        "3",
        "train.py",
        "--job.config_file",
        f"./train_configs/{model}.toml",
        "--model.flavor",
        flavor,
        "--training.batch_size",
        str(batch_size),
        "--training.tensor_parallel_degree",
        str(tp_degree),
        "--experimental.pipeline_parallel_degree",
        str(pp_degree),
        "--experimental.pipeline_parallel_microbatches",
        "1",
        "--experimental.pipeline_parallel_schedule",
        "GPipe",
        "--job.description",
        f"{dec}",
    ]

    py_command = ["python "] + command[13:]
    return command, py_command

def execute_a_train(batch_size: int, tp_degree: int, flavor: str, model: str, cuda_visiable : List[int]):
    """ execute train.py to measure the model size and calculate split points, then run the training script"""
    command, py_command = get_command(
        n_process=-1,
        model=model,
        batch_size=batch_size,
        tp_degree=tp_degree,
        pp_degree=-1,
        flavor=flavor,
    )

    # First Process: Evaluate model size and calculate split points
    ## Create a JobConfig instance and parse arguments
    os.environ["WORLD_SIZE"] = "8"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    config = JobConfig()
    config.parse_args(py_command[2:])  # Pass as list
    config.profiling.metis_profiling = True
    ## Call the main function
    split_points = main(config)

    
    ## remove env vars
    del os.environ["WORLD_SIZE"]
    del os.environ["RANK"]
    del os.environ["LOCAL_RANK"]

    # Second Process: Run the training script
    pp_degree = len(split_points) + 1
    ## set pp_degree, n_process and split points based on these results.
    indx = command.index("--experimental.pipeline_parallel_degree")
    command[indx + 1] = str(pp_degree)
    command.extend(
        [
            "--experimental.pipeline_parallel_split_points",
            ",".join(map(str, split_points)),
        ]
    )
    indx = command.index("--nproc_per_node")
    command[indx + 1] = str(pp_degree * tp_degree)
    
    if len(cuda_visiable) < pp_degree * tp_degree:
        print(f"Not enough cuda visiable devices for model={model}, batch_size={batch_size}, tp_degree={tp_degree}, flavor={flavor}")
        return
    

    ## run the command
    cuda_visible_devices = ",".join(map(str, cuda_visiable))
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    subprocess.run(command)



def get_model_runs(model):
    # WR
    if model == "wideresnet":
        flavors = ["250M", "1B", "2B", "4B", "6.8B", "13B"]
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        mbw_tp1 = [1024, 1024, 1024, 1024, 1024, 512]
        max_batch_size = {1: mbw_tp1}
        tp_degree = [1]

    # MOE
    if model == "moe":
        flavors = ["380M", "1.3B", "2.4B", "10B"]
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        mbw_tp1 = [512, 256, 256, 128] #8GPUs
        # mbw_tp1 = [256, 256, 128, 64] # 6GPUs
        max_batch_size = {1: mbw_tp1}
        tp_degree = [1]

    # Llama2
    if model == "llama2":
        flavors = ["271M", "1B", "7B", "13B"] # , "26B"
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        tp_degree = [1, 2, 4, 8]
        mbw_tp1 = [16, 16, 8, 4, 4]     # max batch size for tensor parallel 1
        mbw_tp2 = [32, 32, 16, 8, 8]    # max batch size for tensor parallel 2
        mbw_tp4 = [64, 32, 32, 16, 16]  # max batch size for tensor parallel 4
        mbw_tp8 = [128, 64, 32, 32, 32] # max batch size for tensor parallel 8
        max_batch_size = {1: mbw_tp1, 2: mbw_tp2, 4: mbw_tp4, 8: mbw_tp8}
    
    return flavors, batch_sizes, max_batch_size, tp_degree


def run_one_4_test(cuda_visiable):
    model = 'llama2'
    flavors, batch_sizes, max_batch_size, tensor_parallel = get_model_runs(model)
    flavor = flavors[0]
    tp = tensor_parallel[0]
    bs = batch_sizes[0]
    print("-"*150)
    print(f"-MAIN- Running model {model}_{flavor} batch_size {bs} and tensor_parallel {tp} -MAIN-")
    execute_a_train(bs, tp, flavor, model, cuda_visiable)
    print(f"-MAIN- Running {model}-{flavor}-bs{bs}-tp{tp} is Done! -MAIN-")


def run_all(cuda_visiable):

    for model in ['moe', 'llama2', 'wideresnet']:
        flavors, batch_sizes, max_batch_size, tensor_parallel = get_model_runs(model)
        for fl_i, flavor in enumerate(flavors):
            for tp in tensor_parallel:
                for bs in batch_sizes:
                    if bs > max_batch_size[tp][fl_i]:
                        continue
                    print("-"*150)
                    print(f"-MAIN- Running model {model}_{flavor} batch_size {bs} and tensor_parallel {tp} -MAIN-")

                    try:
                        execute_a_train(bs, tp, flavor, model, cuda_visiable)
                    except Exception as e:
                        print(f"Error in running model={model}, flavor={flavor}, batch_size={bs}, tensor_parallel={tp}")
                        traceback.print_exc()  # Print full traceback to terminal
                        continue
                    print(f"-MAIN- Running {model}-{flavor}-bs{bs}-tp{tp} is Done! -MAIN-")

def run_a_list(run_list, cuda_visiable):
    
    for run in run_list:
        model, flavor, bs, tp = run
        print("-"*150)
        print(f"-MAIN- Running model {model}_{flavor} batch_size {bs} and tensor_parallel {tp} -MAIN-")
        execute_a_train(bs, tp, flavor, model, cuda_visiable)
        print(f"-MAIN- Running {model}-{flavor}-bs{bs}-tp{tp} is Done! -MAIN-")

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"

    cuda_visiable = [3, 5, 1, 0, 7, 6, 2, 4] 
    # run_all(cuda_visiable)
    
    run_list = [
        ('moe', '380M', 512, 1),
        ('moe', '1.3B', 256, 1),
        # ('moe', '10B', 2, 1),
        ('llama2', '271M', 64, 4),
        ('llama2', '271M', 64, 8),
        ('llama2', '1B', 32, 4),
        ('llama2', '1B', 32, 8),
        ('llama2', '7B', 8, 2),
        ('llama2', '7B', 8, 4),
        ('llama2', '7B', 8, 8),
    ]

    run_list = [
        # ('moe', '380M', 4, 1),
        # ('llama2', '271M', 4, 2),
        ('wideresnet', '250M', 4, 1),
    ]
    run_a_list(run_list, cuda_visiable)
    print("All Done!")


