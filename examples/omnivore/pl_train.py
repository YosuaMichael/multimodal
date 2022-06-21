import pytorch_lightning as pl
import torch
import torchvision
from torchvision.transforms.functional import InterpolationMode

import image_presets
import video_presets
import depth_presets
import transforms
import datasets
import pl_model

def lprint(*x):
    print(f"[{datetime.datetime.now()}]", *x)

def get_single_data_loader_from_dataset(train_dataset, val_dataset, args):  
    collate_fn = None
    num_classes = len(train_dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
        collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731
    
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=None,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=None, num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    return train_data_loader, val_data_loader

def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "kinetics", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def get_kinetics_dataset(kinetics_path, split, transform, args):
    data_dir = os.path.join(kinetics_path, split)
    cache_path = _get_cache_path(data_dir)
    if args.cache_video_dataset and os.path.exists(cache_path):
        print(f"Loading {split} dataset from {cache_path}")
        dataset, _ = torch.load(cache_path)
        dataset.transform = transform
    else:
        if args.distributed:
            print("It is recommended to pre-compute the dataset cache on a single-gpu first, it will be faster!")
        dataset = datasets.OmnivoreKinetics(
            kinetics_path,
            num_classes="400", extensions=("avi", "mp4"), output_format="TCHW",
            frames_per_clip=32, frame_rate=16, step_between_clips=32, 
            split=split, transform=transform
        )
        if args.cache_video_dataset:
            print(f"Saving {split} dataset to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, data_dir), cache_path)
    return dataset

def get_omnivore_data_loader(args):
    imagenet_path = args.imagenet_data_path
    kinetics_path = args.kinetics_data_path
    sunrgbd_path = args.sunrgbd_data_path
    
    train_crop_size = args.train_crop_size
    train_resize_size = args.train_resize_size
    val_crop_size = args.val_crop_size
    val_resize_size = args.val_resize_size
    # Get imagenet data
    imagenet_train_preset = image_presets.ImageNetClassificationPresetTrain(crop_size=train_crop_size, interpolation=InterpolationMode.BICUBIC,
                            auto_augment_policy="ra", random_erase_prob=0.25, )
    imagenet_val_preset = image_presets.ImageNetClassificationPresetEval(crop_size=val_crop_size, interpolation=InterpolationMode.BICUBIC)
    
    imagenet_train_dataset = torchvision.datasets.folder.ImageFolder(f"{imagenet_path}/train", imagenet_train_preset)
    imagenet_val_dataset = torchvision.datasets.folder.ImageFolder(f"{imagenet_path}/val", imagenet_val_preset)
    
    imagenet_train_data_loader, imagenet_val_data_loader = get_single_data_loader_from_dataset(imagenet_train_dataset, imagenet_val_dataset, args)
    
    # Get kinetics data
    video_train_preset = video_presets.VideoClassificationPresetTrain(crop_size=train_crop_size, resize_size=train_resize_size, )
    video_val_preset = video_presets.VideoClassificationPresetEval(crop_size=val_crop_size, resize_size=val_resize_size, )

    start_time = time.time()
    print("Start getting video dataset")
    video_train_dataset = get_kinetics_dataset(kinetics_path, split="train", transform=video_train_preset, args=args)
    video_val_dataset = get_kinetics_dataset(kinetics_path, split="val", transform=video_val_preset, args=args)
    print(f"Took {time.time() - start_time} seconds to get video dataset")
    
    video_train_data_loader, video_val_data_loader = get_single_data_loader_from_dataset(video_train_dataset, video_val_dataset, args)
    
    # Get sunrgbd data
    depth_train_preset = depth_presets.DepthClassificationPresetTrain(crop_size=train_crop_size, interpolation=InterpolationMode.NEAREST,
                                random_erase_prob=0.25, )
    depth_val_preset = depth_presets.DepthClassificationPresetEval(crop_size=val_crop_size, interpolation=InterpolationMode.NEAREST,)

    depth_train_dataset = datasets.OmnivoreSunRgbdDatasets(root=sunrgbd_path, split="train", transform=depth_train_preset)
    depth_val_dataset = datasets.OmnivoreSunRgbdDatasets(root=sunrgbd_path, split="val", transform=depth_val_preset)
    
    depth_train_data_loader, depth_val_data_loader = get_single_data_loader_from_dataset(depth_train_dataset, depth_val_dataset, args)
    
    train_data_loader = datasets.ConcatIterable(
        [imagenet_train_data_loader, video_train_data_loader, depth_train_data_loader],
        ['image', 'video', 'rgbd'],
        [1, 1, 1]
    )
    
    val_data_loader = datasets.ConcatIterable(
        [imagenet_val_data_loader, video_val_data_loader, depth_val_data_loader],
        ['image', 'video', 'rgbd'],
        [0.25, 0.25, 0.25]
    )

    return train_data_loader, val_data_loader


class DataLoaderCb(pl.Callback):
    def __init__(self, train_data_loader, val_data_loader):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_data_loader.init_indices(epoch=pl_module.current_epoch, shuffle=True)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_data_loader.init_indices(epoch=0, shuffle=False)


def main(args):
    model = pl_model.OmnivoreLightningModule(args)

    train_data_loader, val_data_loader = get_omnivore_data_loader(args)
    if args.device == "cpu":
        trainer = pl.Trainer()
    elif args.device == "cuda":
        trainer = pl.Trainer(accelerator="gpu")
    trainer.callbacks.append(DataLoaderCb(train_data_loader, val_data_loader))

    trainer.fit(model, train_data_loader, val_data_loader)


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="TorchMultimodal Omnivore Classification Training", add_help=add_help)
    
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument(
        "-j", "--workers", default=1, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--train-resize-size", default=256, type=int, help="the resize size used for training (default: 256)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    # distributed training parameters
    # parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    # parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--imagenet-data-path", type=str, help="Root directory path of imagenet dataset")
    parser.add_argument("--kinetics-data-path", type=str, help="Root directory path of kinetics dataset")
    parser.add_argument("--sunrgbd-data-path", type=str, help="Root directory path of sunrgbd dataset")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--cache-video-dataset",
        dest="cache_video_dataset",
        help="Cache the video datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)




