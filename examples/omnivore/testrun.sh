
# Evaluate only
PYTHONPATH=:/data/home/yosuamichael/repos/torchmultimodal/examples/omnivore python \
    -u ~/script/run_with_submitit.py \
    --timeout 3000 --ngpus 4 --nodes 1 --partition train \
    --batch-size=8 --workers=3 \
    --cache-video-dataset \
    --val-resize-size=232 \
    --test-only \
    --val-data-sampling-factor 1 1 1 \
    --resume="/data/home/yosuamichael/repos/torchmultimodal/examples/omnivore/weights/omnivore_swin_t.pth" \
    --imagenet-data-path="/datasets01_ontap/imagenet_full_size/061417" \
    --kinetics-data-path="/datasets01_ontap/kinetics/070618/400" \
    --sunrgbd-data-path="/data/home/yosuamichael/datasets/SUN_RGBD" \
