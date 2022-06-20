python train.py --batch-size=16 --workers=2 --output-dir="train_output"


PYTHONPATH=:/data/home/yosuamichael/repos/torchmultimodal/examples/omnivore python \
    -u ~/script/run_with_submitit.py \
    --timeout 3000 --ngpus 2 --nodes 1 --partition train\
    --batch-size=8 \
    --lr=0.5 --lr-warmup-epochs=5 --lr-warmup-method=linear \
    --epochs=600 --weight-decay=0.00002 \
    --label-smoothing=0.1 --mixup-alpha=0.2 --cutmix-alpha=1.0 \
    --train-crop-size=176 --val-resize-size=232 --ra-sampler --ra-reps=4
