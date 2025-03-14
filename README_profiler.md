

# run all model profiling and save logs

python -u ./main.py > output.log 2>&1 &



# unit test

Run unittest

`
python -m unittest ./test_profiler/test_time_mem_profile.py
`


# run all tests
`
python -m unittest discover -s test_profiler -p "*.py"
`

# single model run

`
python ./test_profiler/integration_test.py

python -m test_profiler.integration_test
`


runing command to log to a file and run in background

` python
python3 main.py > log.txt 2>&1 &
`

<!-- # failed:
# 271M_tp8_bs2
# 1B_tp4_bs32
# 26B_tp4_bs1

# torchrun --nproc_per_node 8 --rdzv_backend c10d --rdzv_endpoint localhost:0 --local-ranks-filter 0 --role rank --tee 3 train.py --job.config_file ./train_configs/llama2.toml --model.flavor 271M --training.batch_size 2 --training.tensor_parallel_degree 8 --job.description 'Llama2 271M training'
# torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:0 --local-ranks-filter 0 --role rank --tee 3 train.py --job.config_file ./train_configs/llama2.toml --model.flavor 1B --training.batch_size 32 --training.tensor_parallel_degree 4 --job.description 'Llama2 1B training'
# torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:0 --local-ranks-filter 0 --role rank --tee 3 train.py --job.config_file ./train_configs/llama2.toml --model.flavor 26B --training.batch_size 1 --training.tensor_parallel_degree 4 --job.description 'Llama2 26B training' -->


### llama2
TP 1 PP 1 BS 12 , 46GB
`
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 \
  --local-ranks-filter 0 --role rank --tee 3 \
  train.py --job.config_file ./train_configs/llama2.toml --model.flavor 271M \
  --training.batch_size 12 --training.tensor_parallel_degree 1 \
      --job.description 'Llama2 271M training'
`

TP 2 PP 1 BS 12 , 
`
CUDA_VISIBLE_DEVICES=7,6 torchrun --nproc_per_node 2 --rdzv_backend c10d --rdzv_endpoint localhost:0 \
  --local-ranks-filter 0 --role rank --tee 3 \
  train.py --job.config_file ./train_configs/llama2.toml --model.flavor 271M \
  --training.batch_size 12 --training.tensor_parallel_degree 2 \
      --job.description 'Llama2 271M training'
`

TP 1 PP 2 BS 12 ,
`
CUDA_VISIBLE_DEVICES=7,6 torchrun --nproc_per_node 2 --rdzv_backend c10d --rdzv_endpoint localhost:0 \
  --local-ranks-filter 0 --role rank --tee 3 \
  train.py --job.config_file ./train_configs/llama2.toml --model.flavor 271M \
  --training.batch_size 12 --training.tensor_parallel_degree 1 --experimental.pipeline_parallel_degree 2 \
      --experimental.pipeline_parallel_microbatches 1 --experimental.pipeline_parallel_schedule 'GPipe'\
      --job.description 'Llama2 271M training'
 `

TP 2 PP 2 BS 12 
`
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:0 \
  --local-ranks-filter 0 --role rank --tee 3 \
  train.py --job.config_file ./train_configs/llama2.toml --model.flavor 271M \
  --training.batch_size 12 --training.tensor_parallel_degree 2 --experimental.pipeline_parallel_degree 2 \
      --experimental.pipeline_parallel_microbatches 1 --experimental.pipeline_parallel_schedule 'GPipe'\
      --job.description 'Llama2 271M training'  
 ` 
  
### moe

TP 1 PP 1 BS 40 , 45GB
`
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 \
  --local-ranks-filter 0 --role rank --tee 3 \
  train.py --job.config_file ./train_configs/moe.toml --model.flavor 380M \
  --training.batch_size 40 --training.tensor_parallel_degree 1 --job.description 'MOE 380M training'
` 

TP 1 PP 2 BS 40 ,
`
CUDA_VISIBLE_DEVICES=7,6 torchrun --nproc_per_node 2 --rdzv_backend c10d --rdzv_endpoint localhost:0 \
  --local-ranks-filter 0 --role rank --tee 3 \
  train.py --job.config_file ./train_configs/moe.toml --model.flavor 380M \
  --training.batch_size 40 --training.tensor_parallel_degree 1 --experimental.pipeline_parallel_degree 2 \
      --experimental.pipeline_parallel_microbatches 1 --experimental.pipeline_parallel_schedule 'GPipe'\
      --job.description 'MOE 380M training'
 ` 

### wideresnet

TP 1 PP 1 BS 2 , 45GB
`
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 \
  --local-ranks-filter 0 --role rank --tee 3 \
  train.py --job.config_file ./train_configs/wideresnet.toml --model.flavor 250M \
  --training.batch_size 2 --training.tensor_parallel_degree 1 --job.description 'wideresnet 250M training'
` 
  
TP 1 PP 2 BS 2 ,
`
CUDA_VISIBLE_DEVICES=7,6 torchrun --nproc_per_node 2 --rdzv_backend c10d --rdzv_endpoint localhost:0 \
  --local-ranks-filter 0 --role rank --tee 3 \
  train.py --job.config_file ./train_configs/wideresnet.toml --model.flavor 250M \
  --training.batch_size 2 --training.tensor_parallel_degree 1 --experimental.pipeline_parallel_degree 2 \
      --experimental.pipeline_parallel_microbatches 1 --experimental.pipeline_parallel_schedule 'GPipe'\
      --job.description 'wideresnet 250M training'
`

# Batch models run

