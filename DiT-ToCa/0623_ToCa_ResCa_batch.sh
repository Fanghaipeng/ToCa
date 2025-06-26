export CUDA_VISIBLE_DEVICES=4,5

torchrun --nnodes=1 --nproc_per_node=2 --master-port 29519 sample_ddp.py \
    --model DiT-XL/2 --per-proc-batch-size 128 --image-size 256 \
    --num-fid-samples 50000 --cfg-scale 1.5 --num-sampling-steps 50 \
    --cache-type attention --fresh-ratio 0.07 --ratio-scheduler ToCa-ddim50 \
    --force-fresh global --fresh-threshold 3 --soft-fresh-weight 0.25 \
    --ddim-sample

wait

torchrun --nnodes=1 --nproc_per_node=2 --master-port 29519 sample_ddp.py \
    --model DiT-XL/2 --per-proc-batch-size 128 --image-size 256 \
    --num-fid-samples 50000 --cfg-scale 1.5 --num-sampling-steps 50 \
    --cache-type attention --fresh-ratio 0.07 --ratio-scheduler ToCa-ddim50 \
    --force-fresh global --fresh-threshold 3 --soft-fresh-weight 0.25 \
    --ddim-sample \
    --use-ResCa 

wait

torchrun --nnodes=1 --nproc_per_node=2 --master-port 29519 sample_ddp.py \
    --model DiT-XL/2 --per-proc-batch-size 128 --image-size 256 \
    --num-fid-samples 50000 --cfg-scale 1.5 --num-sampling-steps 50 \
    --cache-type attention --fresh-ratio 0.07 --ratio-scheduler ToCa-ddim50 \
    --force-fresh global --fresh-threshold 3 --soft-fresh-weight 0.25 \

wait

torchrun --nnodes=1 --nproc_per_node=2 --master-port 29519 sample_ddp.py \
    --model DiT-XL/2 --per-proc-batch-size 128 --image-size 256 \
    --num-fid-samples 50000 --cfg-scale 1.5 --num-sampling-steps 50 \
    --cache-type attention --fresh-ratio 0.07 --ratio-scheduler ToCa-ddim50 \
    --force-fresh global --fresh-threshold 3 --soft-fresh-weight 0.25 \
    --use-ResCa 

