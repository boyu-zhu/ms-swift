# 4 * 32GB
# Multimodal packing currently only supports qwen2_vl, qwen2_5_vl, qwen2_5_omni, internvl2_5/3
# A demo for four modalities that can be run directly
# For local datasets, it is recommended to use streaming: `--streaming true` (save memory)
NPROC_PER_NODE=7 \
ENABLE_AUDIO_OUTPUT=0 \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
MAX_PIXELS=1003520 \
uv run swift sft \
    --model Qwen/Qwen2.5-Omni-7B \
    --dataset \
                  '/ms-swift/data/text_ready_sft_data/Aegis_w_gt_sft.jsonl' \
                  '/ms-swift/data/text_ready_sft_data/beaver_w_gt_sft.jsonl' \
                  '/ms-swift/data/text_ready_sft_data/toxic_w_gt_sft.jsonl' \
                  '/ms-swift/data/text_ready_sft_data/wildguardmix_w_gt_sft.jsonl' \
                  '/ms-swift/data/llavaguard/llavaguard_qwen3_sft.jsonl' \
                  '/ms-swift/data/unsafebench/unsafebench_qwen3_sft.jsonl' \
                  '/ms-swift/data/vlguard/vlguard_qwen3_sft.jsonl' \
                  '/ms-swift/data/vlsbench/vlsbench_qwen3_sft.jsonl' \
                '/ms-swift/data/safesora/safesora_qwen3_sft.jsonl' \
                '/ms-swift/data/fvc/fvc_qwen3_sft.jsonl' \
                '/ms-swift/data/lspd/lspd_qwen3_sft.jsonl' \
                '/ms-swift/data/tikharm_dcsass/dcsass_qwen3_sft.jsonl' \
                '/ms-swift/data/tikharm_dcsass/tikharm_sft.jsonl' \
                  '/ms-swift/data/toxic_chat_audio/toxic_chat_audio_sft.jsonl' \
    --split_dataset_ratio 0.01 \
    --train_type full \
    --attn_impl flash_attn \
    --packing true \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --freeze_vit true \
    --gradient_accumulation_steps 4 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output/qwen2_5_omni_guardrail \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --dataset_num_proc 1 \
    --deepspeed zero2 \
    --use_hf true \
    --push_to_hub true \
    --hub_model_id 'boyuzhuGPT/qwen2_5_omni_guardrail'



                # '/ms-swift/data/fakesv/fakesv_qwen3_sft.jsonl' \
                # '/ms-swift/data/fvc/fvc_qwen3_sft.jsonl' \
                # '/ms-swift/data/lspd/lspd_qwen3_sft.jsonl' \
                # '/ms-swift/data/tikharm_dcsass/dcsass_qwen3_sft.jsonl' \
                # '/ms-swift/data/tikharm_dcsass/tikharm_sft.jsonl' \