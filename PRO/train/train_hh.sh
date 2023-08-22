export OMP_NUM_THREADS=16
root_dir=..

#stage 23
id=$1
data_path=$2
ranking_len=$3
mkdir -p $root_dir/logs/$id/$ranking_len
export WANDB_API_KEY=c0a4d9df0a801da1b53257f0c63d8283af4ae526
accelerate launch --num_processes 1 --config_file ds_config.yaml main.py \
    --task hh \
    --train_file_path $root_dir/data/${data_path} \
    --validation_file_path $root_dir/data/hh_dev \
    --validation_file_name sampled_dev.json \
    --output_dir $root_dir/checkpoints/index_$id/stage_$ranking_len \
    --log_path $root_dir/logs/$id/$ranking_len \
    --log_with=wandb
    --index $id \
    --seed 42 \
    --temperature 1 \
    --sft_weight 0.05 \
    --num_train_epochs 1 \
    --training_stage_num $ranking_len \
    --block_size 512 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --model_name_or_path cerebras/Cerebras-GPT-111M \
    --do_train \
    --do_validation