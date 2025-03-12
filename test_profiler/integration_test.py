import subprocess
import os
from train import main, JobConfig


llama2_runs = [
    {
        "batch_size": 12,
        "tp_degree": 1,
        "flavor": "271M",
        "model": "llama2",
    },
    {
        "batch_size": 12,
        "tp_degree": 2,
        "flavor": "271M",
        "model": "llama2",
    },
    {
        "batch_size": 16,
        "tp_degree": 1,
        "flavor": "271M",
        "model": "llama2",
    },
    {
        "batch_size": 12,
        "tp_degree": 2,
        "flavor": "271M",
        "model": "llama2",
    },
    {
        "batch_size": 2,
        "tp_degree": 1,
        "flavor": "13B",
        "model": "llama2",
    },
    {
        "batch_size": 1,
        "tp_degree": 2,
        "flavor": "7B",
        "model": "llama2",
    },
]

wideresnet_runs = [
    {
        "batch_size": 2,
        "tp_degree": 1,
        "flavor": "250M",
        "model": "wideresnet",
    },
    {
        "batch_size": 2,
        "tp_degree": 1,
        "flavor": "250M",
        "model": "wideresnet",
    },
]

moe_runs = [
    {
        "batch_size": 40,
        "tp_degree": 1,
        "flavor": "380M",
        "model": "moe",
    },
    {
        "batch_size": 40,
        "tp_degree": 1,
        "flavor": "380M",
        "model": "moe",
    },
]


def get_command(
    n_process,
    model,
    batch_size,
    tp_degree,
    pp_degree,
    flavor,
):
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
    return command, py_command  # " ".join(command)


class TestTrainingScript:
    def __init__(self, runs, cuda_visiable):
        self.runs = runs
        self.cuda_visiable = cuda_visiable

        self.run_all(runs)

    def run_all(self, runs):
        for run in runs:
            print(f"Running {run}")
            self._test_train_script_runs(run)

    def _test_train_script_runs(self, run):
        """Test if train.py runs without errors."""
        command, py_command = get_command(
            n_process=-1,
            model=run["model"],
            batch_size=run["batch_size"],
            tp_degree=run["tp_degree"],
            pp_degree=-1,
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
        command[indx + 1] = str(pp_degree * run["tp_degree"])

        ## run the command
        cuda_visible_devices = ",".join(map(str, self.cuda_visiable))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        subprocess.run(command)


if __name__ == "__main__":

    cuda_visiable = [3, 5, 1, 0, 7, 6, 2, 4] 
    run = {
        "batch_size": 512,
        "tp_degree": 1,
        "flavor": "13B",
        "model": "wideresnet",
    }
    TestTrainingScript([run], cuda_visiable)

    # TestTrainingScript(wideresnet_runs)
    # TestTrainingScript(moe_runs)
