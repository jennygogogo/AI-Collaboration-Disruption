export DATA_PATH="train_data.jsonl"
export TEST_DATA_PATH="test_data.jsonl"
export VAL_DATA_PATH="val_data.jsonl"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --checkpoint Qwen/Qwen3-8B \
  --data_path "$DATA_PATH" \
  --val_data_path "$VAL_DATA_PATH" \
  --test_data_path "$TEST_DATA_PATH" \
  --runs_dir official_runs \
  --batch_size 1 \
  --total_epochs 3 \
  --learning_rate 1e-4 \
  --max_length 2048