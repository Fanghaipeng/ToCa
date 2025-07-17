
torchrun --nproc_per_node=2 scripts/inference_ddp.py \
    --model_path  /data/fanghaipeng/checkpoints/PixArt-alpha/PixArt-XL-2-256x256.pth \
    --image_size 256 \
    --bs 48 \
    --txt_file /data1/fanghaipeng/paper/PruneCache/ToCa/COCO_caption_prompts_30k.txt \
    --fresh_threshold 3 \
    --fresh_ratio 0.40 \
    --cache_type attention \
    --force_fresh global \
    --soft_fresh_weight 0.25 \
    --ratio_scheduler ToCa \
    --use-ResCa

wait

torchrun --nproc_per_node=2 scripts/inference_ddp.py \
    --model_path  /data/fanghaipeng/checkpoints/PixArt-alpha/PixArt-XL-2-256x256.pth \
    --image_size 256 \
    --bs 48 \
    --txt_file /data1/fanghaipeng/paper/PruneCache/ToCa/COCO_caption_prompts_30k.txt \
    --fresh_threshold 3 \
    --fresh_ratio 0.40 \
    --cache_type attention \
    --force_fresh global \
    --soft_fresh_weight 0.25 \
    --ratio_scheduler ToCa

wait

