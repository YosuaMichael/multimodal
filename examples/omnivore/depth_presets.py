import torch
import numpy as np
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode
from rand_aug3d import RandAugment3d

from torch import nn

class ColorJitter3d(transforms.ColorJitter):
    def forward(self, img):
        assert isinstance(img, torch.Tensor)
        img[:3, :, :] = super().forward(img[:3, :, :])
        return img

# From original implementation https://www.internalfb.com/code/fbsource/[f1a98f41bcce7ee621f0248a6e0235a3e3dea628]/fbcode/deeplearning/projects/omnivore/vissl/data/ssl_transforms/depth_transforms.py?lines=13
class DropChannels(nn.Module):
    """
    Drops Channels with predefined probability values.
    Pads the dropped channels with `pad_value`.
    Channels can be tied using `tie_channels`
    For example, for RGBD input, RGB can be tied by using `tie_channels=[0,1,2]`.
    In this case, channels [0,1,2] will be dropped all at once or not at all.
    Assumes input is of the form CxHxW or TxCxHxW
    """

    def __init__(
        self, channel_probs, fill_values, tie_channels=None, all_channel_drop=False
    ):
        """
        channel_probs: List of probabilities
        fill_values: List of values to fill the dropped channels with
        tie_channels: List of indices. Tie dropping of certain channels.
        all_channel_drop: Bool variable to prevent cases where all channels are dropped.
        """
        super().__init__()
        channel_probs = np.array(channel_probs, dtype=np.float32)

        self.channel_probs = channel_probs
        self.fill_values = fill_values
        self.tie_channels = tie_channels
        self.all_channel_drop = all_channel_drop
        
        if tie_channels is not None:
            tie_probs = [channel_probs[x] for x in tie_channels]
            assert len(set(tie_probs)) == 1, "All tie_channel probs must be equal"

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        if x.ndim == 3:
            # CxHxW
            num_channels = x.shape[0]
            channel_index = 0
        elif x.ndim == 4:
            # TxCxHxW
            num_channels = x.shape[1]
            channel_index = 1
        else:
            raise ValueError(f"Unexpected number of dims {x.ndim}. Expected 3 or 4.")

        assert num_channels == len(
            self.channel_probs
        ), f"channel_probs is {len(self.channel_probs)} but got {num_channels} channels"

        to_drop = [
            np.random.random() < self.channel_probs[c] for c in range(num_channels)
        ]
        # print(f"TRYING TO DROP CHANNELS WITH: {to_drop}, channel_probs: {self.channel_probs}")
        if self.tie_channels is not None:
            first_drop = to_drop[self.tie_channels[0]]
            for idx in self.tie_channels[1:]:
                to_drop[idx] = first_drop

        if all(to_drop) and self.all_channel_drop is False:
            # all channels will be dropped, prevent it
            to_drop = [False for _ in range(num_channels)]

        for c in range(num_channels):
            if not to_drop[c]:
                continue
            if channel_index == 0:
                x[c, ...] = self.fill_values[c]
            elif channel_index == 1:
                x[:, c, ...] = self.fill_values[c]
            else:
                raise NotImplementedError()
        return x

# From original implementation https://www.internalfb.com/code/fbsource/[f1a98f41bcce7ee621f0248a6e0235a3e3dea628]/fbcode/deeplearning/projects/omnivore/vissl/data/ssl_transforms/depth_normalize.py
class RGBToFloatAndDepthNorm(nn.Module):
    """
    Normalize the depth channel: in an RGBD input of shape (4, H, W),
    only the last channel is modified.
    The depth channel is also clamped at 0.0. The Midas depth prediction
    model outputs inverse depth maps - negative values correspond
    to distances far away so can be clamped at 0.0
    """

    def __init__(
        self,
        max_depth: float,
        clamp_max_before_scale: bool = False,
        min_depth: float = 0.01,
    ):
        """
        Args:
            max_depth (float): The max value of depth for the dataset
            clamp_max (bool): Whether to clamp to max_depth or to divide by max_depth
        """
        super().__init__()
        self.max_depth = max_depth
        self.clamp_max_before_scale = clamp_max_before_scale
        self.min_depth = min_depth

    def forward(self, image: torch.Tensor):
        C, H, W = image.shape
        if C != 4:
            err_msg = (
                f"This transform is for 4 channel RGBD input only; got {image.shape}"
            )
            raise ValueError(err_msg)
        
        assert image.dtype == torch.uint8
        
        color_img = image[:3, ...].float()  # (3, H, W)
        depth_img = image[3:4, ...].float()  # (1, H, W)
        
        # Convert color_img to float with range [0, 1]
        color_img /= 255

        # Clamp to 0.0 to prevent negative depth values
        depth_img = depth_img.clamp(min=self.min_depth)

        # divide by max_depth
        if self.clamp_max_before_scale:
            depth_img = depth_img.clamp(max=self.max_depth)

        depth_img /= self.max_depth

        img = torch.cat([color_img, depth_img], dim=0)
        return img
    

class DepthClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406, 0.8049),
        std=(0.229, 0.224, 0.225, 0.2116),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        random_erase_prob=0.0,
    ):
        trans = [
            transforms.RandomResizedCrop(crop_size, interpolation=interpolation)
        ]

        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
            
        trans.extend(
            [
                RandAugment3d(interpolation=interpolation, num_ops=1),
                ColorJitter3d(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                # transforms.ConvertImageDtype(torch.float),
                RGBToFloatAndDepthNorm(max_depth=75, clamp_max_before_scale=True),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))
            
        trans.append(transforms.Normalize(mean=mean, std=std))
        trans.append(DropChannels(channel_probs=[0.5, 0.5, 0.5, 0], tie_channels=[0, 1, 2], fill_values=[0, 0, 0, 0]))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class DepthClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406, 0.0418),
        std=(0.229, 0.224, 0.225, 0.0295),
        interpolation=InterpolationMode.BILINEAR,
    ):

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)
