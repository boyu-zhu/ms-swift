# 4 * 32GB
# Multimodal packing currently only supports qwen2_vl, qwen2_5_vl, qwen2_5_omni, internvl2_5/3
# A demo for four modalities that can be run directly
# For local datasets, it is recommended to use streaming: `--streaming true` (save memory)
NPROC_PER_NODE=8 \
ENABLE_AUDIO_OUTPUT=0 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
MAX_PIXELS=1003520 \
swift sft \
    --model omni \
    --dataset 'finetune/beavertails_dataset.jsonl' \
              'finetune/dssass.jsonl' \
              'finetune/fakett.jsonl' \
              'finetune/llava.json' \
              'finetune/safesora.json' \
              'finetune/tikharm.jsonl' \
              'finetune/toxic_chat_dataset.jsonl' \
              'finetune/unsafebench.json' \
              'finetune/vlguard.jsonl' \
    --split_dataset_ratio 0.01 \
    --train_type full \
    --attn_impl flash_attn \
    --packing true \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --freeze_vit true \
    --gradient_accumulation_steps 1 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output_1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --dataset_num_proc 128 \
    --deepspeed zero2
