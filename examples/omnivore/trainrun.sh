python train.py --batch-size=16 --workers=2 --output-dir="train_output"

python train.py --batch-size=2 --workers=2 --device=cpu --output-dir="train_output" \
    --imagenet-data-path="/Users/yosuamichael/Downloads/datasets/mini_omnivore/mini_imagenet" \
    --kinetics-data-path="/Users/yosuamichael/Downloads/datasets/mini_omnivore/mini_kinetics" \
    --sunrgbd-data-path="/Users/yosuamichael/Downloads/datasets/SUN_RGBD" \


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
