export CUDA_VISIBLE_DEVICES=1
# python /data1/fanghaipeng/paper/PruneCache/ToCa/DiT-ToCa/sample.py \
#     --image-size 256 \
#     --num-sampling-steps 250 \
#     --cache-type attention \
#     --fresh-threshold 4 \
#     --fresh-ratio 0.07 \
#     --ratio-scheduler ToCa-ddpm250 \
#     --force-fresh global \
#     --soft-fresh-weight 0.25

# wait

# python /data1/fanghaipeng/paper/PruneCache/ToCa/DiT-ToCa/sample.py \
#     --image-size 256 \
#     --num-sampling-steps 250 \
#     --cache-type attention \
#     --fresh-threshold 4 \
#     --fresh-ratio 0.07 \
#     --ratio-scheduler ToCa-ddpm250 \
#     --force-fresh global \
#     --soft-fresh-weight 0.25 \
#     --use-ResCa

# wait

# python /data1/fanghaipeng/paper/PruneCache/ToCa/DiT-ToCa/sample.py \
#     --image-size 256 \
#     --num-sampling-steps 250 \
#     --cache-type attention \
#     --fresh-threshold 4 \
#     --fresh-ratio 0.07 \
#     --ratio-scheduler ToCa-ddpm250 \
#     --force-fresh global \
#     --soft-fresh-weight 0.25 \
#     --ddim-sample

# wait

python /data1/fanghaipeng/paper/PruneCache/ToCa/DiT-ToCa/sample.py \
    --image-size 256 \
    --num-sampling-steps 250 \
    --cache-type attention \
    --fresh-threshold 4 \
    --fresh-ratio 0.07 \
    --ratio-scheduler ToCa-ddpm250 \
    --force-fresh global \
    --soft-fresh-weight 0.25 \
    --use-ResCa \
    --ddim-sample