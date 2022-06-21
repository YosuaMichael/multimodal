# Build simple omnivore training loop from bottom up

# TODO:
# - [P0] Able to run distributed [DONE]
# - [P0] Refactor the dataset to separate file (and update sunrgbd class to not using manual json) [DONE?]
# - [P0] Enable to resume checkpoint and do evaluation only [DONE?]
# - [P1] Better logging using MetricLogger [DONE]
# - [P1] Use mixed precision [DONE?]
# - [P2] Enable using EMA during training
#
# NOTE: [DONE?] -> need more testing

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data.dataloader import default_collate
import torchvision
import torchmultimodal.models.omnivore as omnivore
from torchvision.transforms.functional import InterpolationMode


import time
import os
import datetime

import image_presets
import video_presets
import depth_presets
import utils
import transforms
from sampler import RASampler
import datasets

def lprint(*x):
    print(f"[{datetime.datetime.now()}]", *x)

def get_single_data_loader_from_dataset(train_dataset, val_dataset, args):  
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(train_dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

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
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    return train_data_loader, val_data_loader

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

    video_train_dataset = datasets.OmnivoreKinetics(
        f"{kinetics_path}", 
        frames_per_clip=32, frame_rate=16, step_between_clips=32, 
        split="train", transform= video_train_preset
    )
    video_val_dataset = datasets.OmnivoreKinetics(
        f"{kinetics_path}", 
        frames_per_clip=32, frame_rate=16, step_between_clips=32, 
        split="val", transform= video_val_preset
    )
    
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

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    data_loader.init_indices(epoch=epoch, shuffle=True)
    header = f"Epoch: [{epoch}]"
    for i, ((image, target), input_type) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image, input_type)
        loss = criterion(output, target)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters[f"{input_type}_acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters[f"{input_type}_acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

def evaluate(model, criterion, val_data_loader, device, args):
    model.eval()
    val_data_loader.init_indices(epoch=0, shuffle=False)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "
    with torch.inference_mode():
        for i, ((image, target), input_type) in enumerate(metric_logger.log_every(val_data_loader, args.print_freq, header)):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image, input_type)
            loss = criterion(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters[f"{input_type}_acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters[f"{input_type}_acc5"].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print(f"{header} Image Acc@1 {metric_logger.image_acc1.global_avg:.3f} Image Acc@5 {metric_logger.image_acc5.global_avg:.3f}")
    print(f"{header} Video Acc@1 {metric_logger.video_acc1.global_avg:.3f} Video Acc@5 {metric_logger.video_acc5.global_avg:.3f}")
    print(f"{header} RGBD Acc@1 {metric_logger.rgbd_acc1.global_avg:.3f} RGBD Acc@5 {metric_logger.rgbd_acc5.global_avg:.3f}")



def main(args):
    # Will check if it has distributed setup,
    # it will set args.distributed and args.gpu accordingly
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    
    if args.output_dir:
        utils.mkdir(args.output_dir)
    
    print("Creating model")
    # TODO: Let user choose omnivore variant from args
    model = omnivore.omnivore_swin_t()
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # Get dataset
    # TODO: Finish the function get_data_loader()!
    train_data_loader, val_data_loader = get_omnivore_data_loader(args)

    
    # Preparing scheduler and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
    )
    
    opt_name = args.opt.lower()
    optimizer = torch.optim.SGD(
        parameters,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
    )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        evaluate(model, criterion, val_data_loader, device=device, args=args)
        return

    # Start training
    # TODO: EDIT!
    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, optimizer, train_data_loader, device, epoch, args, scaler)
        lr_scheduler.step()
        evaluate(model, criterion, val_data_loader, device=device, args=args)
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")




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
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
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
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)