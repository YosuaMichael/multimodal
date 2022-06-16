python train.py --batch-size=16 --workers=2 --output-dir="train_output"


PYTHONPATH=:/data/home/yosuamichael/repos/torchmultimodal/examples/omnivore python \
    -u ~/script/run_with_submitit.py \
    --timeout 3000 --ngpus 1 --nodes 1 --partition train\
    --batch-size=16 \
    --lr=0.5 --lr-scheduler=cosineannealinglr --lr-warmup-epochs=5 --lr-warmup-method=linear \
    --auto-augment=ta_wide --epochs=600 --random-erase=0.1 --weight-decay=0.00002 \
    --norm-weight-decay=0.0 --label-smoothing=0.1 --mixup-alpha=0.2 --cutmix-alpha=1.0 \
    --train-crop-size=176 --model-ema --val-resize-size=232 --ra-sampler --ra-reps=4
