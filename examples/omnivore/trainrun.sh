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

# Using the full data!
PYTHONPATH=:/data/home/yosuamichael/repos/torchmultimodal/examples/omnivore python \
    -u ~/script/run_with_submitit.py \
    --timeout 3000 --ngpus 8 --nodes 1 --partition train \
    --batch-size=128 --workers=3 \
    --cache-video-dataset \
    --lr=0.002 --lr-warmup-epochs=5 --lr-warmup-method=linear \
    --epochs=100 --weight-decay=0.05 \
    --label-smoothing=0.1 --mixup-alpha=0.2 --cutmix-alpha=1.0 \
    --train-crop-size=176 --val-resize-size=232 \
    --opt="adamw" \
    --imagenet-data-path="/datasets01_ontap/imagenet_full_size/061417" \
    --kinetics-data-path="/datasets01_ontap/kinetics/070618/400" \
    --sunrgbd-data-path="/data/home/yosuamichael/datasets/SUN_RGBD" \
