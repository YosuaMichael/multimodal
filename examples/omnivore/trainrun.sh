python train.py --batch-size=16 --workers=2 --output-dir="train_output"

python train.py --batch-size=2 --workers=2 --device=cpu --output-dir="train_output" \
    --imagenet-data-path="/Users/yosuamichael/Downloads/datasets/mini_omnivore/mini_imagenet" \
    --kinetics-data-path="/Users/yosuamichael/Downloads/datasets/mini_omnivore/mini_kinetics" \
    --sunrgbd-data-path="/Users/yosuamichael/Downloads/datasets/SUN_RGBD" \

python pl_train.py --batch-size=8 --workers=2 --device=cuda --output-dir="train_output_pl" \
    --lr=0.002 \
    --imagenet-data-path="/data/home/yosuamichael/datasets/mini_imagenet" \
    --kinetics-data-path="/data/home/yosuamichael/datasets/mini_kinetics" \
    --sunrgbd-data-path="/data/home/yosuamichael/datasets/SUN_RGBD"

PYTHONPATH=:/data/home/yosuamichael/repos/torchmultimodal/examples/omnivore python \
    -u ~/script/run_with_submitit.py \
    --timeout 3000 --ngpus 2 --nodes 1 --partition train\
    --batch-size=8 \
    --lr=0.5 --lr-warmup-epochs=5 --lr-warmup-method=linear \
    --epochs=600 --weight-decay=0.00002 \
    --label-smoothing=0.1 --mixup-alpha=0.2 --cutmix-alpha=1.0 \
    --train-crop-size=176 --val-resize-size=232 --ra-sampler --ra-reps=4 \
    --imagenet-data-path="/data/home/yosuamichael/datasets/mini_imagenet" \
    --kinetics-data-path="/data/home/yosuamichael/datasets/mini_kinetics" \
    --sunrgbd-data-path="/data/home/yosuamichael/datasets/SUN_RGBD"

# Just to create video cache
env -u SLURM_PROCID python train.py \
    --batch-size=128 --workers=1 \
    --cache-video-dataset \
    --kinetics-dataset-workers=30 \
    --imagenet-data-path="/datasets01_ontap/imagenet_full_size/061417" \
    --kinetics-data-path="/datasets01_ontap/kinetics/070618/400" \
    --sunrgbd-data-path="/data/home/yosuamichael/datasets/SUN_RGBD" \

# Using the full data!
PYTHONPATH=:/data/home/yosuamichael/repos/torchmultimodal/examples/omnivore python \
    -u ~/script/run_with_submitit.py \
    --timeout 3000 --ngpus 8 --nodes 1 --partition train \
    --batch-size=10 --workers=3 \
    --cache-video-dataset \
    --lr=0.002 --lr-warmup-epochs=5 --lr-warmup-method=linear \
    --epochs=100 --weight-decay=0.05 \
    --label-smoothing=0.1 --mixup-alpha=0.2 --cutmix-alpha=1.0 \
    --train-crop-size=176 --val-resize-size=232 \
    --extra-kinetics-dataloader-workers=6 \
    --opt="adamw" --amp \
    --imagenet-data-path="/datasets01_ontap/imagenet_full_size/061417" \
    --kinetics-data-path="/datasets01_ontap/kinetics/070618/400" \
    --sunrgbd-data-path="/data/home/yosuamichael/datasets/SUN_RGBD" \
    --resume="/data/checkpoints/yosuamichael/experiments/22577/checkpoint.pth" \

# Better param (work well with image + depth)
PYTHONPATH=:/data/home/yosuamichael/repos/torchmultimodal/examples/omnivore python \
    -u ~/script/run_with_submitit.py \
    --timeout 30000 --ngpus 8 --nodes 2 --partition train \
    --batch-size=128 --workers=5 \
    --cache-video-dataset \
    --lr=0.5 --lr-warmup-epochs=5 --lr-warmup-method=linear \
    --epochs=500 --weight-decay=0.05 \
    --label-smoothing=0.1 --mixup-alpha=0.2 --cutmix-alpha=1.0 \
    --train-crop-size=176 --val-resize-size=232 \
    --extra-kinetics-dataloader-workers=5 \
    --opt="sgd" --ra-sampler --ra-reps=4 \
    --random-erase=0.1 \
    --color-jitter-factor 0.0 0.0 0.0 0.0 \
    --video-grad-accum-iter=32 \
    --val-data-sampling-factor 1 1 1 \
    --train-data-sampling-factor 1 1 10 \
    --imagenet-data-path="/datasets01_ontap/imagenet_full_size/061417" \
    --kinetics-data-path="/datasets01_ontap/kinetics/070618/400" \
    --sunrgbd-data-path="/data/home/yosuamichael/datasets/SUN_RGBD" \

# Exp param (try to be as similar as the paper)
PYTHONPATH=:/data/home/yosuamichael/repos/torchmultimodal/examples/omnivore python \
    -u ~/script/run_with_submitit.py \
    --timeout 30000 --ngpus 8 --nodes 2 --partition train \
    --batch-size=128 --workers=5 --extra-kinetics-dataloader-workers=5 \
    --cache-video-dataset --num-epoch-per-eval=5\
    --lr=0.002 --lr-warmup-epochs=5 --lr-warmup-method=linear \
    --epochs=500 --weight-decay=0.05 --video-grad-accum-iter=32 \
    --label-smoothing=0.1 --mixup-alpha=0.2 --cutmix-alpha=1.0 \
    --train-crop-size=176 --val-resize-size=232 \
    --opt="adamw" --random-erase=0.1 \
    --color-jitter-factor 0.1 0.1 0.1 0.1 \
    --val-data-sampling-factor 1 1 1 \
    --train-data-sampling-factor 1 1 10 \
    --imagenet-data-path="/datasets01_ontap/imagenet_full_size/061417" \
    --kinetics-data-path="/datasets01_ontap/kinetics/070618/400" \
    --sunrgbd-data-path="/data/home/yosuamichael/datasets/SUN_RGBD" \
    --resume="/data/home/yosuamichael/temp/model_21.pth" \

# Exact same param as paper
PYTHONPATH=:/data/home/yosuamichael/repos/torchmultimodal/examples/omnivore python \
    -u ~/script/run_with_submitit.py \
    --timeout 30000 --ngpus 8 --nodes 1 --partition train \
    --batch-size=128 --workers=5 --extra-kinetics-dataloader-workers=5 \
    --cache-video-dataset --num-epoch-per-eval=10\
    --lr=0.002 --lr-warmup-epochs=5 --lr-warmup-method=linear \
    --epochs=500 --weight-decay=0.05 --video-grad-accum-iter=32 \
    --label-smoothing=0.1 --mixup-alpha=0.2 --cutmix-alpha=1.0 \
    --train-crop-size=224 --val-resize-size=224 \
    --opt="adamw" --random-erase=0.25 \
    --color-jitter-factor 0.4 0.4 0.4 0.4 \
    --val-data-sampling-factor 1 1 1 \
    --train-data-sampling-factor 1 1 10 \
    --imagenet-data-path="/datasets01_ontap/imagenet_full_size/061417" \
    --kinetics-data-path="/datasets01_ontap/kinetics/070618/400" \
    --sunrgbd-data-path="/data/home/yosuamichael/datasets/SUN_RGBD" \


# Only train on image to verify training script correctness!    
env -u SLURM_PROCID python train.py \
    --batch-size=128 --workers=6 \
    --cache-video-dataset \
    --extra-kinetics-dataloader-workers=0 \
    --lr=0.002 --lr-warmup-epochs=5 --lr-warmup-method=linear \
    --epochs=200 --weight-decay=0.05 \
    --label-smoothing=0.1 --mixup-alpha=0.2 --cutmix-alpha=1.0 \
    --train-crop-size=176 --val-resize-size=232 \
    --opt="adamw" --random-erase=0.1 \
    --color-jitter-factor 0.1 0.1 0.1 0.1 \
    --imagenet-data-path="/datasets01_ontap/imagenet_full_size/061417" \
    --kinetics-data-path="/datasets01_ontap/kinetics/070618/400" \
    --sunrgbd-data-path="/data/home/yosuamichael/datasets/SUN_RGBD" \
    --output-dir="temp_output_10" \
    --video-grad-accum-iter=32 \
    --val-data-sampling-factor 1 0 1 \
    --train-data-sampling-factor 1 0 10 \
    --ra-sampler --ra-reps=4 \

