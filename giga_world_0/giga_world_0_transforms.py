import copy
import random

import numpy as np
import torch
from giga_datasets import video_utils
from giga_train import TRANSFORMS
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


@TRANSFORMS.register
class GigaWorld0Transform:
    """Video transformation class for GigaWorld0 training.

    Handles video sampling, resizing, cropping, normalization, and reference frame generation.
    """

    def __init__(self, num_frames: int, height: int, width: int, image_cfg: dict, fps: int = 16):
        """Initialize the transform.

        Args:
            num_frames: Number of frames to sample from the video.
            height: Target height for the output frames.
            width: Target width for the output frames.
            image_cfg: Configuration dictionary containing mask generator settings.
            fps: Frames per second for the video (default: 16).
        """
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.fps = fps
        # Normalization transform: convert [0, 1] to [-1, 1]
        self.normalize = transforms.Normalize([0.5], [0.5])
        self.mask_generator = MaskGenerator(**image_cfg['mask_generator'])

    def __call__(self, data_dict):
        """Apply transformations to the input data.

        Args:
            data_dict: Dictionary containing 'video' and 'prompt_embeds'.

        Returns:
            new_data_dict: Transformed data dictionary with processed images and masks.
        """
        video = data_dict['video']
        video_legnth = len(video)
        sample_indexes = np.linspace(0, video_legnth - 1, self.num_frames, dtype=int)
        input_images = video_utils.sample_video(video, sample_indexes, method=2)
        # Convert to tensor and rearrange dimensions: (T, H, W, C) -> (T, C, H, W)
        input_images = torch.from_numpy(input_images).permute(0, 3, 1, 2).contiguous()

        image_height = input_images.shape[2]
        image_width = input_images.shape[3]
        dst_width, dst_height = self.width, self.height

        # Calculate new dimensions maintaining aspect ratio
        if float(dst_height) / image_height < float(dst_width) / image_width:
            new_height = int(round(float(dst_width) / image_width * image_height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / image_height * image_width))

        # Random crop coordinates
        x1 = random.randint(0, new_width - dst_width)
        y1 = random.randint(0, new_height - dst_height)

        # Apply resize and crop
        input_images = F.resize(input_images, (new_height, new_width), InterpolationMode.BILINEAR)
        input_images = F.crop(input_images, y1, x1, dst_height, dst_width)

        # ===== Normalize =====
        # Scale to [0, 1]
        input_images = input_images / 255.0
        # Normalize to [-1, 1]
        input_images = self.normalize(input_images)

        # ===== Generate Reference Images and Masks =====
        # Get masks for reference frames
        ref_masks, ref_latent_masks = self.mask_generator.get_mask(input_images.shape[0])
        # Expand dimensions for broadcasting: (T,) -> (T, 1, 1, 1)
        ref_masks = ref_masks[:, None, None, None]
        # Expand for latent space: (T_latent,) -> (1, T_latent, 1, 1)
        ref_latent_masks = ref_latent_masks[None, :, None, None]
        # Create reference images by masking
        ref_images = copy.deepcopy(input_images)
        ref_images = ref_images * ref_masks

        new_data_dict = dict(
            fps=self.fps,
            images=input_images,
            ref_images=ref_images,
            ref_masks=ref_latent_masks,
            prompt_embeds=data_dict['prompt_embeds'],
        )
        return new_data_dict


class MaskGenerator:
    """Generates binary masks for reference frames in video sequences.

    Used to control which frames are treated as reference (conditioning) frames during training.
    """

    def __init__(self, max_ref_frames: int, factor: int = 8, start: int = 1):
        """Initialize the mask generator.

        Args:
            max_ref_frames: Maximum number of reference frames (must satisfy: (max_ref_frames - 1) % factor == 0).
            factor: Downsampling factor between frame space and latent space (default: 8).
            start: Minimum number of reference latents to generate (default: 1).
        """
        assert max_ref_frames > 0 and (max_ref_frames - 1) % factor == 0
        self.max_ref_frames = max_ref_frames
        self.factor = factor
        self.start = start
        # Calculate maximum reference latents based on factor
        self.max_ref_latents = 1 + (max_ref_frames - 1) // factor
        assert self.start <= self.max_ref_latents

    def get_mask(self, num_frames: int):
        """Generate binary masks for reference frames and latents.

        Args:
            num_frames: Total number of frames in the sequence.

        Returns:
            ref_masks: Binary mask tensor for frame space (shape: (num_frames,)).
                      1.0 for reference frames, 0.0 for non-reference frames.
            ref_latent_masks: Binary mask tensor for latent space (shape: (num_latents,)).
                             1.0 for reference latents, 0.0 for non-reference latents.
        """
        # Validate input dimensions
        assert num_frames > 0 and (num_frames - 1) % self.factor == 0 and num_frames >= self.max_ref_frames

        # Calculate number of latents based on downsampling factor
        num_latents = 1 + (num_frames - 1) // self.factor

        # Randomly select number of reference latents
        num_ref_latents = random.randint(self.start, self.max_ref_latents)

        # Calculate corresponding number of reference frames
        if num_ref_latents > 0:
            num_ref_frames = 1 + (num_ref_latents - 1) * self.factor
        else:
            num_ref_frames = 0

        # Create binary mask for frames
        ref_masks = torch.zeros((num_frames,), dtype=torch.float32)
        ref_masks[:num_ref_frames] = 1  # Mark first N frames as reference

        # Create binary mask for latents
        ref_latent_masks = torch.zeros((num_latents,), dtype=torch.float32)
        ref_latent_masks[:num_ref_latents] = 1  # Mark first N latents as reference

        return ref_masks, ref_latent_masks
