import subprocess
import os
from train import main, JobConfig

def get_command(
    n_process,
    model,
    batch_size,
    tp_degree,
    pp_degree,
    flavor,
):
    dec = f"{model} {flavor} training"
    # gpu_list = ",".join(map(str, cuda_visiable))
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
    
    py_command = ['python '] + command[13:]
    return command, py_command  # " ".join(command)


class TestTrainingScript:
    def __init__(self, runs):
        self.runs = runs

        self.run_all(runs)

    def run_all(self, runs):
        for run in runs:
            print(f"Running {run}")
            self._test_train_script_runs(run)

    def _test_train_script_runs(self, run):
        """Test if train.py runs without errors."""
        command, py_command = get_command(
            n_process=run["n_process"],
            model=run["model"],
            batch_size=run["batch_size"],
            tp_degree=run["tp_degree"],
            pp_degree=run["pp_degree"],
            flavor=run["flavor"],
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
        print(f"{split_points=}")
        ## remove env vars
        del os.environ["WORLD_SIZE"]
        del os.environ["RANK"]
        del os.environ["LOCAL_RANK"]
        
        # Second Process: Run the training script
        ## set pp_degree and split points based on these results.
        indx = command.index("--experimental.pipeline_parallel_degree")
        command[indx+1] = str(len(split_points)+1)
        command.extend(["--experimental.pipeline_parallel_split_points", ",".join(map(str, split_points))])
        cuda_visible_devices = ",".join(map(str, run["cuda_visiable"]))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        subprocess.run(command)


if __name__ == "__main__":
    llama2_runs = [
        {
            "cuda_visiable": [2],
            "n_process": 1,
            "batch_size": 12,
            "tp_degree": 1,
            "pp_degree": 1,
            "flavor": "271M",
            "model": "llama2",
        },
        {
            "cuda_visiable": [2, 3],
            "n_process": 2,
            "batch_size": 12,
            "tp_degree": 2,
            "pp_degree": 1,
            "flavor": "271M",
            "model": "llama2",
        },
        {
            "cuda_visiable": [3, 5],
            "n_process": 2,
            "batch_size": 16,
            "tp_degree": 1,
            "pp_degree": 2,
            "flavor": "271M",
            "model": "llama2",
        },
        {
            "cuda_visiable": [4, 5, 6, 7],
            "n_process": 4,
            "batch_size": 12,
            "tp_degree": 2,
            "pp_degree": 2,
            "flavor": "271M",
            "model": "llama2",
        },
        {
            "cuda_visiable": [7,5,2,3,6,1,0],
            "n_process": 7,
            "batch_size": 2,
            "tp_degree": 1,
            "pp_degree": 7,
            "flavor": "13B",
            "model": "llama2",
        },
        {
            "cuda_visiable": [7,5,2,3,6,1,0,4],
            "n_process": 8,
            "batch_size": 2,
            "tp_degree": 2,
            "pp_degree": 4,
            "flavor": "13B",
            "model": "llama2",
        },        
    ]

    wideresnet_runs = [
        {
            "cuda_visiable": [7],
            "n_process": 1,
            "batch_size": 2,
            "tp_degree": 1,
            "pp_degree": 1,
            "flavor": "250M",
            "model": "wideresnet",
        },
        {
            "cuda_visiable": [7, 6],
            "n_process": 2,
            "batch_size": 2,
            "tp_degree": 1,
            "pp_degree": 2,
            "flavor": "250M",
            "model": "wideresnet",
        },
    ]

    moe_runs = [
        {
            "cuda_visiable": [7],
            "n_process": 1,
            "batch_size": 40,
            "tp_degree": 1,
            "pp_degree": 1,
            "flavor": "380M",
            "model": "moe",
        },
        {
            "cuda_visiable": [7, 6],
            "n_process": 2,
            "batch_size": 40,
            "tp_degree": 1,
            "pp_degree": 2,
            "flavor": "380M",
            "model": "moe",
        },
    ]

    # TestTrainingScript([llama2_runs[0]])

    
    TestTrainingScript([llama2_runs[-1]])

    # TestTrainingScript(wideresnet_runs)
    # TestTrainingScript(moe_runs)
