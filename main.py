import os
import subprocess

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"

    model_sizes = ["271M", "1B", "7B", "13B", "26B"]
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    tensor_parallel = [1, 2, 4, 8]
    mbw_tp1 = [16, 16, 8, 4, 4]     # max batch size for tensor parallel 1
    mbw_tp2 = [32, 32, 16, 8, 8]    # max batch size for tensor parallel 2
    mbw_tp4 = [64, 32, 32, 16, 16]  # max batch size for tensor parallel 4
    mbw_tp8 = [128, 64, 32, 32, 32] # max batch size for tensor parallel 8
    max_bs = {1: mbw_tp1, 2: mbw_tp2, 4: mbw_tp4, 8: mbw_tp8}
    number_of_layers = [16, 18, 32, 40, 80]
    
    # NGPU = 8
    CONFIG_FILE = "./train_configs/llama2.toml"

    for model in model_sizes:
        for tp in tensor_parallel:
            for bs in batch_sizes:
                if bs > max_bs[tp][model_sizes.index(model)]:
                    continue
                print(f"-- Running model {model} with batch size {bs} and tensor parallel {tp} ...")
                description = f'Llama2 {model} training'
                command = [
                    "torchrun",
                    f"--nproc_per_node={tp}",
                    "--rdzv_backend=c10d",
                    "--rdzv_endpoint=localhost:0",
                    "--local-ranks-filter=0",
                    "--role=rank",
                    "--tee=3",
                    "train.py",
                    f"--job.config_file={CONFIG_FILE}",
                    f"--model.flavor={model}",
                    f"--training.batch_size={bs}",
                    f"--training.tensor_parallel_degree={tp}",
                    f"--job.description='Llama2 {model} training'",
                ]
                subprocess.run(command)
                print(f"-- Running {model}-{bs}-{tp} is Done!")

# runing command to log to a file and run in background
# python3 main.py > log.txt 2>&1 &


# failed:
# 271M_tp8_bs2
# 1B_tp4_bs32
# 26B_tp4_bs1

# torchrun --nproc_per_node 8 --rdzv_backend c10d --rdzv_endpoint localhost:0 --local-ranks-filter 0 --role rank --tee 3 train.py --job.config_file ./train_configs/llama2.toml --model.flavor 271M --training.batch_size 2 --training.tensor_parallel_degree 8 --job.description 'Llama2 271M training'
# torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:0 --local-ranks-filter 0 --role rank --tee 3 train.py --job.config_file ./train_configs/llama2.toml --model.flavor 1B --training.batch_size 32 --training.tensor_parallel_degree 4 --job.description 'Llama2 1B training'
# torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:0 --local-ranks-filter 0 --role rank --tee 3 train.py --job.config_file ./train_configs/llama2.toml --model.flavor 26B --training.batch_size 1 --training.tensor_parallel_degree 4 --job.description 'Llama2 26B training'


# torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:0 \
#   --local-ranks-filter 0 --role rank --tee 3 \
#   train.py --job.config_file ./train_configs/llama2.toml --model.flavor 271M \
#   --training.batch_size 32 --training.tensor_parallel_degree 4 --job.description 'Llama2 1B training'