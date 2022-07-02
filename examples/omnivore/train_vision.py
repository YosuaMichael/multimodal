import datetime
import os
import time
import warnings

import presets
import torch
import torch.utils.data
import torchvision
import transforms
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

# Additional import
import datetime
import datasets
import image_presets
import video_presets
import depth_presets
import torchmultimodal.models.omnivore as omnivore
import torchvision.datasets.samplers as video_samplers

# Funciton to get dataset
def lprint(*x):
    print(f"[{datetime.datetime.now()}]", *x)


def get_sampler(train_dataset, val_dataset, dataset_name, args):
    if dataset_name == "kinetics":
        train_sampler = video_samplers.RandomClipSampler(
            train_dataset.video_clips, args.train_clips_per_video
        )
        val_sampler = video_samplers.UniformClipSampler(
            val_dataset.video_clips, args.val_clips_per_video
        )
        if args.distributed:
            train_sampler = video_samplers.DistributedSampler(train_sampler)
            val_sampler = video_samplers.DistributedSampler(val_sampler)
    else:
        if args.distributed:
            if hasattr(args, "ra_sampler") and args.ra_sampler:
                train_sampler = RASampler(
                    train_dataset, shuffle=True, repetitions=args.ra_reps
                )
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset
                )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False
            )
        else:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
            val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    return train_sampler, val_sampler


def get_single_data_loader_from_dataset(train_dataset, val_dataset, dataset_name, args):
    train_sampler, val_sampler = get_sampler(
        train_dataset, val_dataset, dataset_name, args
    )
    collate_fn = None
    num_classes = len(train_dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(
            transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha)
        )
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(
            transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha)
        )
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
        collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731

    # Have extra 3 workers for kinetics
    num_train_workers = args.workers
    num_val_workers = args.workers
    if dataset_name == "imagenet":
        if args.train_data_sampling_factor[0] == 0:
            num_train_workers = 1
        if args.val_data_sampling_factor[0] == 0:
            num_val_workers = 1
    elif dataset_name == "kinetics":
        num_train_workers += args.extra_kinetics_dataloader_workers
        num_val_workers += args.extra_kinetics_dataloader_workers
        if args.train_data_sampling_factor[1] == 0:
            num_train_workers = 1
        if args.val_data_sampling_factor[1] == 0:
            num_val_workers = 1
    elif dataset_name == "sunrgbd":
        if args.train_data_sampling_factor[2] == 0:
            num_train_workers = 1
        if args.val_data_sampling_factor[2] == 0:
            num_val_workers = 1

    # Reduce the amount of validation worker needed
    num_val_workers = (num_val_workers // 2) + 1

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=num_train_workers,
        pin_memory=args.loader_pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=num_val_workers,
        pin_memory=args.loader_pin_memory,
        drop_last=True,
    )
    return train_data_loader, val_data_loader


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join(
        "~", ".torch", "torchmultimodal", "omnivore", "kinetics", h[:10] + ".pt"
    )
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def get_kinetics_dataset(
    kinetics_path,
    split,
    transform,
    step_between_clips,
    args,
    frame_rate=16,
    frames_per_clip=32,
):
    data_dir = os.path.join(kinetics_path, split)
    cache_path = _get_cache_path(data_dir)
    lprint(f"cache_path: {cache_path}")
    if args.cache_video_dataset and os.path.exists(cache_path):
        lprint(f"Loading {split} dataset from {cache_path}")
        dataset, _ = torch.load(cache_path)
        dataset.transform = transform
    else:
        if args.distributed:
            print(
                "It is recommended to pre-compute the dataset cache on a single-gpu first, it will be faster!"
            )
        lprint("Building kinetics dataset")
        dataset = datasets.OmnivoreKinetics(
            kinetics_path,
            num_classes="400",
            extensions=("avi", "mp4"),
            output_format="TCHW",
            frames_per_clip=frames_per_clip,
            frame_rate=frame_rate,
            step_between_clips=step_between_clips,
            split=split,
            transform=transform,
            num_workers=args.kinetics_dataset_workers,
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
    lprint("Start getting imagenet dataset")

    imagenet_train_preset = image_presets.ImageNetClassificationPresetTrain(
        crop_size=train_crop_size,
        interpolation=InterpolationMode.BILINEAR,
        auto_augment_policy="ra",
        random_erase_prob=args.random_erase,
        color_jitter_factor=args.color_jitter_factor,
    )
    imagenet_val_preset = image_presets.ImageNetClassificationPresetEval(
        crop_size=val_crop_size, interpolation=InterpolationMode.BILINEAR
    )
    #"""

    imagenet_train_dataset = torchvision.datasets.folder.ImageFolder(
        f"{imagenet_path}/train", imagenet_train_preset
    )
    imagenet_val_dataset = torchvision.datasets.folder.ImageFolder(
        f"{imagenet_path}/val", imagenet_val_preset
    )

    (
        imagenet_train_data_loader,
        imagenet_val_data_loader,
    ) = get_single_data_loader_from_dataset(
        imagenet_train_dataset, imagenet_val_dataset, "imagenet", args
    )

    lprint("Finish getting imagenet dataset")

    # Get kinetics data
    #"""
    video_train_preset = video_presets.VideoClassificationPresetTrain(
        crop_size=train_crop_size,
        resize_size=train_resize_size,
    )
    video_val_preset = video_presets.VideoClassificationPresetEval(
        crop_size=val_crop_size,
        resize_size=val_resize_size,
    )

    start_time = time.time()
    lprint("Start getting video dataset")
    video_train_dataset = get_kinetics_dataset(
        kinetics_path,
        split="train",
        transform=video_train_preset,
        step_between_clips=1,
        args=args,
    )
    video_val_dataset = get_kinetics_dataset(
        kinetics_path,
        split="val",
        transform=video_val_preset,
        step_between_clips=1,
        args=args,
    )
    lprint(f"Took {time.time() - start_time} seconds to get video dataset")

    (
        video_train_data_loader,
        video_val_data_loader,
    ) = get_single_data_loader_from_dataset(
        video_train_dataset, video_val_dataset, "kinetics", args
    )
    #"""

    #"""
    # Get sunrgbd data
    lprint("Start creating depth dataset")
    depth_train_preset = depth_presets.DepthClassificationPresetTrain(
        crop_size=train_crop_size,
        interpolation=InterpolationMode.NEAREST,
        random_erase_prob=args.random_erase,
        max_depth=75.0,
        mean=(0.485, 0.456, 0.406, 0.0418),
        std=(0.229, 0.224, 0.225, 0.0295),
        color_jitter_factor=args.color_jitter_factor,
    )
    depth_val_preset = depth_presets.DepthClassificationPresetEval(
        crop_size=val_crop_size,
        interpolation=InterpolationMode.NEAREST,
        max_depth=75.0,
        mean=(0.485, 0.456, 0.406, 0.0418),
        std=(0.229, 0.224, 0.225, 0.0295),
    )

    depth_train_dataset = datasets.OmnivoreSunRgbdDatasets(
        root=sunrgbd_path, split="train", transform=depth_train_preset
    )
    depth_val_dataset = datasets.OmnivoreSunRgbdDatasets(
        root=sunrgbd_path, split="val", transform=depth_val_preset
    )

    (
        depth_train_data_loader,
        depth_val_data_loader,
    ) = get_single_data_loader_from_dataset(
        depth_train_dataset, depth_val_dataset, "sunrgbd", args
    )

    lprint("Finish getting depth dataset")
    #"""

    #"""
    train_data_loader = datasets.ConcatIterable(
        [imagenet_train_data_loader, video_train_data_loader, depth_train_data_loader],
        ["image", "video", "rgbd"],
        args.train_data_sampling_factor,
    )

    val_data_loader = datasets.ConcatIterable(
        [imagenet_val_data_loader, video_val_data_loader, depth_val_data_loader],
        ["image", "video", "rgbd"],
        args.val_data_sampling_factor,
    )


    """
    train_data_loader = datasets.ConcatIterable([imagenet_train_data_loader],
        ["image"], [1])
    val_data_loader = datasets.ConcatIterable([imagenet_val_data_loader],
        ["image"], [1])
    #"""

    """
    train_data_loader = datasets.ConcatIterable([imagenet_train_data_loader, depth_train_data_loader],
        ["image", "rgbd"], [1, 0])
    val_data_loader = datasets.ConcatIterable([imagenet_val_data_loader, depth_train_data_loader],
        ["image", "rgbd"], [1, 0])
    #"""

    return train_data_loader, val_data_loader


def _chunk_forward_backward(model, image, target, input_type, 
    chunk_start, chunk_end, realized_accum_iter, 
    criterion, optimizer, device, args, scaler=None):

    chunk_image, chunk_target = image[chunk_start:chunk_end, ...].to(device), target[chunk_start:chunk_end, ...].to(device)

    with torch.cuda.amp.autocast(enabled=scaler is not None):
        chunk_output = model(chunk_image, input_type)
        loss = criterion(chunk_output, chunk_target)

    # Normalize the loss
    loss /= realized_accum_iter

    if scaler is not None:
        scaler.scale(loss).backward()
        if args.clip_grad_norm is not None:
            # we should unscale the gradients of optimizer's assigned params if do gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
    else:
        loss.backward()
        if args.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

    return loss, chunk_output




def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    data_loader.init_indices(epoch=epoch, shuffle=True)

    header = f"Epoch: [{epoch}]"
    for i, ((image, target), input_type) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
    #for i, (image, target)  in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        #input_type = "image"

        # Try doing gradient accumulation
        accum_iter = 1
        if input_type == "video":
            accum_iter = args.video_grad_accum_iter
        start_time = time.time()
        b, c, t, h, w = image.shape

        # Accumulate loss from smaller chunk size (compared to batch_size)
        chunk_start = 0
        chunk_size = (b + accum_iter - 1) // accum_iter
        realized_accum_iter = (b + chunk_size - 1) // chunk_size
        all_chunk_outputs = []
        accum_loss = 0
        for chunk_num in range(realized_accum_iter):
            chunk_end = chunk_start + chunk_size
            if args.distributed and chunk_num < realized_accum_iter - 1:
                # We dont synchronized unless it is the last chunk in DDP mode
                with model.no_sync():
                    loss, chunk_output = _chunk_forward_backward(model, image, target, input_type, 
                        chunk_start, chunk_end, realized_accum_iter, 
                        criterion, optimizer, device, args, scaler)
            else:
                loss, chunk_output = _chunk_forward_backward(model, image, target, input_type, 
                    chunk_start, chunk_end, realized_accum_iter, 
                    criterion, optimizer, device, args, scaler)

            all_chunk_outputs.append(chunk_output)
            accum_loss += loss.item()
            chunk_start = chunk_end

        # Weight update
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        
        # if scaler is not None:
        #     scaler.scale(loss).backward()
        #     if args.clip_grad_norm is not None:
        #         # we should unscale the gradients of optimizer's assigned params if do gradient clipping
        #         scaler.unscale_(optimizer)
        #         nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        #     scaler.step(optimizer)
        #     scaler.update()
        # else:
        #     loss.backward()
        #     if args.clip_grad_norm is not None:
        #         nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        #     optimizer.step()

        optimizer.zero_grad()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)
        
        output = torch.cat(all_chunk_outputs, dim=0)
        target = target.to(device)
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=accum_loss, lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters[f"{input_type}_acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters[f"{input_type}_acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def evaluate(model, criterion, data_loader, device, args, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    data_loader.init_indices(epoch=0, shuffle=False)

    num_processed_samples = 0
    with torch.inference_mode():
        for (image, target), input_type in metric_logger.log_every(data_loader, print_freq, header):
        #for (image, target)  in metric_logger.log_every(data_loader, print_freq, header):
            #input_type = "image"

            # We do the evaluation in chunks to reduce memory usage for video
            accum_iter = 1
            if input_type == "video":
                accum_iter = args.video_grad_accum_iter
            b, c, t, h, w = image.shape

            chunk_start = 0
            chunk_size = (b + accum_iter - 1) // accum_iter
            realized_accum_iter = (b + chunk_size - 1) // chunk_size
            accum_loss = 0
            all_chunk_outputs = []
            for chunk_num in range(realized_accum_iter):
                chunk_end = chunk_start + chunk_size
                
                chunk_image = image[chunk_start:chunk_end, ...].to(device, non_blocking=True)
                chunk_target = target[chunk_start:chunk_end, ...].to(device, non_blocking=True)
                chunk_output = model(chunk_image, input_type)
                loss = criterion(chunk_output, chunk_target)

                accum_loss += loss.item()
                all_chunk_outputs.append(chunk_output)
                chunk_start = chunk_end

            output = torch.cat(all_chunk_outputs, dim=0)
            target = target.to(device, non_blocking=True)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=accum_loss)
            metric_logger.meters[f"{input_type}_acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters[f"{input_type}_acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    """
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )
    """
    metric_logger.synchronize_between_processes()

    try:
        print(
            f"{header} Image Acc@1 {metric_logger.image_acc1.global_avg:.3f} Image Acc@5 {metric_logger.image_acc5.global_avg:.3f}"
        )
    except Exception:
        pass
    try:
        print(
            f"{header} Video Acc@1 {metric_logger.video_acc1.global_avg:.3f} Video Acc@5 {metric_logger.video_acc5.global_avg:.3f}"
        )
    except Exception:
        pass
    try:
        print(
            f"{header} RGBD Acc@1 {metric_logger.rgbd_acc1.global_avg:.3f} RGBD Acc@5 {metric_logger.rgbd_acc5.global_avg:.3f}"
        )
    except Exception:
        pass

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    
    data_loader, data_loader_test = get_omnivore_data_loader(args)

    print("Creating model")
    # model = torchvision.models.__dict__[args.model](weights=args.weights, num_classes=num_classes)
    model = omnivore.omnivore_swin_t()
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
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

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, args=args, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device, args=args)
        return

    # TODO: Delete this later! (only for testing)
    evaluate(model, criterion, data_loader_test, device=device, args=args)

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
        lr_scheduler.step()
        if epoch % args.num_epoch_per_eval == args.num_epoch_per_eval - 1:
            evaluate(model, criterion, data_loader_test, device=device, args=args)
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, args=args, log_suffix="EMA")
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
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
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

    parser.add_argument(
        "--train-resize-size",
        default=256,
        type=int,
        help="the resize size used for training (default: 256)",
    )
    parser.add_argument(
        "--imagenet-data-path", type=str, help="Root directory path of imagenet dataset"
    )
    parser.add_argument(
        "--kinetics-data-path", type=str, help="Root directory path of kinetics dataset"
    )
    parser.add_argument(
        "--sunrgbd-data-path", type=str, help="Root directory path of sunrgbd dataset"
    )
    parser.add_argument(
        "--cache-video-dataset",
        dest="cache_video_dataset",
        help="Cache the video datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--train-clips-per-video",
        default=1,
        type=int,
        help="maximum number of clips per video to consider during training",
    )
    parser.add_argument(
        "--val-clips-per-video",
        default=4,
        type=int,
        help="maximum number of clips per video to consider during validation",
    )
    parser.add_argument(
        "--kinetics-dataset-workers",
        default=4,
        type=int,
        help="number of kinetics dataset reader workers (default=4)",
    )
    parser.add_argument(
        "--extra-kinetics-dataloader-workers",
        default=8,
        type=int,
        help="number of kinetics data loader workers (default=8)",
    )
    parser.add_argument(
        "--num-epoch-per-eval",
        default=5,
        type=int,
        help="Number of epoch between each evaluation on validation dataset",
    )
    parser.add_argument(
        "--val-data-sampling-factor",
        default=[1.0, 1.0, 1.0],
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--train-data-sampling-factor",
        default=[1.0, 1.0, 10.0],
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--loader-pin-memory",
        help="Do we use pin_memory in data_loader",
        action="store_true",
    )
    parser.add_argument("--color-jitter-factor", nargs=4, type=float, help="Color jitter factor in brightness, contrast, saturation, and hue")
    parser.add_argument("--video-grad-accum-iter", type=int, default=1, help="[EXPERIMENT] number of gradient accumulation to reduce batch size for video")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
