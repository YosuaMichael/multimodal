# Build simple omnivore training loop from bottom up
import torch
import torch.nn as nn
import torchvision
import torchmultimodal.models.omnivore as omnivore
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms.functional import InterpolationMode


from pathlib import Path
import json
import time
import os
import PIL
import datetime
import numpy as np

import image_presets
import video_presets
import depth_presets
import utils



# TODO: Put the dataset class on separate file
class OmnivoreKinetics(torchvision.datasets.kinetics.Kinetics):
    def __getitem__(self, idx):
        video, audio, label = super().__getitem__(idx)
        return video, label
    
class OmnivoreSunRgbdDatasets(VisionDataset):
    def __init__(self, root, transform = None, target_transform = None, split="train"):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._data_dir = Path(self.root) / "SUNRGBD"
        self._meta_dir = Path(self.root) / "SUNRGBDtoolbox"
        
        if not self._check_exists():
            print(f"data_dir: {self._data_dir}\nmeta_dir: {self._meta_dir}")
            raise RuntimeError("Dataset not found.")
            
        self.classes = ['bathroom',
             'bedroom',
             'classroom',
             'computer_room',
             'conference_room',
             'corridor',
             'dining_area',
             'dining_room',
             'discussion_area',
             'furniture_store',
             'home_office',
             'kitchen',
             'lab',
             'lecture_theatre',
             'library',
             'living_room',
             'office',
             'rest_space',
             'study_space'
        ]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        
        # TODO: Need to change later!
        # Currently the file "sunrgbd_trainval_path.json" is manually created with a script
        # We should create this file from script that is downloaded!
        with open(Path(self.root) / "sunrgbd_trainval_path.json", "r") as fin:
            self.trainval_image_dir_map = json.load(fin)
            
        self.image_dirs = [key for key, value in self.trainval_image_dir_map.items() if value == split]
        
        
    def _check_exists(self):
        return self._data_dir.is_dir() and self._meta_dir.is_dir()
    
    def __len__(self):
        return len(self.image_dirs)

    def _read_sunrgbd_image(self, image_dir):
        rgb_dir = os.path.join(image_dir, "image")
        rgb_path = os.path.join(rgb_dir, os.listdir(rgb_dir)[0])
        img_rgb = PIL.Image.open(rgb_path)
        arr_rgb = np.asarray(img_rgb)

        # Using depth_bfx, but maybe can also consider just using depth
        depth_dir = os.path.join(image_dir, "depth_bfx")
        depth_path = os.path.join(depth_dir, os.listdir(depth_dir)[0])
        img_d = PIL.Image.open(depth_path)
        if img_d.mode == "I":
            arr_d = (np.asarray(img_d) * 255.99999 / 2**16).astype(np.uint8)

        arr_rgbd = np.dstack((arr_rgb, arr_d))
        return arr_rgbd
    
    def _get_sunrgbd_scene_class(self, image_dir):
        with open(os.path.join(image_dir, "scene.txt"), "r") as fin:
            scene_class = fin.read().strip()
        return scene_class
    
    def __getitem__(self, idx):
        # return tuple of image (H W C==4) and scene class index
        image_dir = self.image_dirs[idx]
        x_rgbd = torch.tensor(self._read_sunrgbd_image(image_dir), dtype=torch.uint8)
        x_rgbd = x_rgbd.permute(2, 0, 1) # H W C -> C H W
        scene_class = self._get_sunrgbd_scene_class(image_dir)
        scene_idx = self.class_to_idx[scene_class]
        
        if self.transform:
            x_rgbd = self.transform(x_rgbd)
            
        if self.target_transform:
            scene_idx = self.target_transform(scene_idx)
            
        return x_rgbd, scene_idx
        
        
class ConcatIterable:
    def __init__(self, iterables, output_keys, repeat_factors, seed=42): 
        self.iterables = iterables
        self.output_keys = output_keys
        self.repeat_factors = repeat_factors
        self.seed = seed
        self.num_iterables = len(self.iterables)
        assert self.num_iterables == len(output_keys)
        assert self.num_iterables == len(repeat_factors)
        
        
        # The iterator len is adjusted with repeat_factors
        self.iterator_lens = [int(repeat_factors[i] * len(itb)) for i, itb in enumerate(self.iterables)]
        self.max_total_steps = sum(self.iterator_lens)
        self.indices = None
        self.iterators = None
        
        # self.step_counter == None indicate that self.indices are not yet initialized
        self.step_counter = None
        
    def init_indices(self, epoch=0, shuffle=False):
        # We should initiate indices for each epoch, especially if we want to shuffle
        self.step_counter = 0
    
        self.iterators = [iter(dl) for dl in self.iterables]
        self.indices = torch.cat([torch.ones(self.iterator_lens[i], dtype=torch.int32) * i for i in range(self.num_iterables)])
        assert self.max_total_steps == len(self.indices)
        
        if shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + epoch)
            shuffle_indices = torch.randperm(len(self.indices), generator=g)
            self.indices = self.indices[shuffle_indices]
            
    def __next__(self):
        if self.step_counter == None:
            # Initiate the indices without shuffle as default!
            self.init_indices()
        if self.step_counter >= self.max_total_steps:
            raise StopIteration
        
        idx = self.indices[self.step_counter]
        output_key = self.output_keys[idx]
        # print(idx)
        try:
            batch = next(self.iterators[idx])
        except StopIteration:
            # We cycle over the data_loader to the beginning. This can happen when repeat_factor > 1
            # Take note that in this case we always use same shuffling from same data_loader in an epoch
            self.iterators[idx] = iter(self.iterables[idx])
            batch = next(self.iterators[idx])
        
        self.step_counter += 1
        # Return batch and output_key
        return batch, output_key
    
    def __len__(self):
        return self.max_total_steps
    
    def __iter__(self):
        return self
    
    

# TODO: Complete the functions!
def get_single_data_loader_from_dataset(train_dataset, val_dataset, args):  
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    test_sampler = torch.utils.data.SequentialSampler(val_dataset)
    
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=torch.utils.data.dataloader.default_collate,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )
    return train_data_loader, val_data_loader

def get_omnivore_data_loader(args):
    # TODO: Make sure this is not hardcoded
    imagenet_path = "/Users/yosuamichael/Downloads/datasets/mini_omnivore/mini_imagenet"
    kinetics_path = "/Users/yosuamichael/Downloads/datasets/mini_omnivore/mini_kinetics"
    sunrgbd_path = "/Users/yosuamichael/Downloads/datasets/SUN_RGBD"
    
    train_crop_size = 224
    val_crop_size = 224
    train_resize_size = 256
    val_resize_size = 256
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

    video_train_dataset = OmnivoreKinetics(
        f"{kinetics_path}", 
        frames_per_clip=32, frame_rate=16, step_between_clips=8, 
        split="train", transform= video_train_preset
    )
    video_val_dataset = OmnivoreKinetics(
        f"{kinetics_path}", 
        frames_per_clip=32, frame_rate=16, step_between_clips=32, 
        split="val", transform= video_val_preset
    )
    
    video_train_data_loader, video_val_data_loader = get_single_data_loader_from_dataset(video_train_dataset, video_val_dataset, args)
    
    # Get sunrgbd data
    depth_train_preset = depth_presets.DepthClassificationPresetTrain(crop_size=train_crop_size, interpolation=InterpolationMode.NEAREST,
                                random_erase_prob=0.25, )
    depth_val_preset = depth_presets.DepthClassificationPresetEval(crop_size=val_crop_size, interpolation=InterpolationMode.NEAREST,)

    depth_train_dataset = OmnivoreSunRgbdDatasets(root=sunrgbd_path, split="train", transform=depth_train_preset)
    depth_val_dataset = OmnivoreSunRgbdDatasets(root=sunrgbd_path, split="val", transform=depth_val_preset)
    
    depth_train_data_loader, depth_val_data_loader = get_single_data_loader_from_dataset(depth_train_dataset, depth_val_dataset, args)
    
    train_data_loader = ConcatIterable(
        [imagenet_train_data_loader, video_train_data_loader, depth_train_data_loader],
        ['image', 'video', 'depth'],
        [1, 1, 1]
    )
    
    val_data_loader = ConcatIterable(
        [imagenet_val_data_loader, video_val_data_loader, depth_val_data_loader],
        ['image', 'video', 'depth'],
        [1, 1, 1]
    )

    return train_data_loader, val_data_loader

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args):
    model.train()
    data_loader.init_indices(epoch=epoch, shuffle=True)
    acc1s, acc5s = [], []
    for i, ((image, target), input_type) in enumerate(data_loader):
        image, target = image.to(device), target.to(device)
        output = model(image, input_type)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        acc1s.append(acc1)
        acc5s.append(acc5)
        print(f"Train epoch {epoch}, batch_num {i}, input_type: {input_type}")
    print(f"[Train #{epoch}] acc1: {np.mean(acc1s)}, acc5: {np.mean(acc5s)}")

def evaluate(model, criterion, val_data_loader, device):
    model.eval()
    with torch.inference_mode():
        for (image, target), input_type in data_loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image, input_type)
            loss = criterion(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            print(f"[Eval] acc1: {acc1}, acc5: {acc5}")



def main(args):
    print(args)
    device = torch.device(args.device)
    
    if args.output_dir:
        utils.mkdir(args.output_dir)
    
    print("Creating model")
    # TODO: Let user choose omnivore variant from args
    model = omnivore.omnivore_swin_t()
    model.to(device)
    
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
        
    # Start training
    # TODO: EDIT!
    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, optimizer, train_data_loader, device, epoch, args)
        lr_scheduler.step()
        evaluate(model, criterion, val_data_loader, device=device)
        if args.output_dir:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")




def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser(description="TorchMultimodal Omnivore Classification Training")
    
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
    parser.add_argument("--device", default="cpu", type=str, help="device (Use cuda or cpu Default: cpu)")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
