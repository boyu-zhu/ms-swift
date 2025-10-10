NPROC_PER_NODE=8 \
ENABLE_AUDIO_OUTPUT=0 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
MAX_PIXELS=1003520 \
uv run swift sft \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --dataset \
                '/ms-swift/data/text_ready_sft_data/Aegis_w_gt_sft.jsonl' \
                '/ms-swift/data/text_ready_sft_data/beaver_w_gt_sft.jsonl' \
                '/ms-swift/data/text_ready_sft_data/toxic_w_gt_sft.jsonl' \
                '/ms-swift/data/text_ready_sft_data/wildguardmix_w_gt_sft.jsonl' \
    --split_dataset_ratio 0.01 \
    --load_from_cache_file true \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --attn_impl flash_attn \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --packing true \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output/qwen3_omni_text_only \
    --warmup_ratio 0.05 \
    --dataset_num_proc 1 \
    --dataloader_num_workers 4 \
    --use_hf true