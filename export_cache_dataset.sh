OMP_NUM_THREADS=14 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift export \
    --model Qwen/Qwen2.5-Omni-7B \
    --dataset '/vision_distill/output/llavaguard_qwen3_sft.jsonl' \
    --split_dataset_ratio 0.01 \
    --dataset_num_proc 16 \
    --to_cached_dataset true \
    --lazy_tokenize false \
    --output_dir ./qwen2_5_omni_llavaguard \
    --use_hf true