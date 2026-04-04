# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Union
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF
from ....optimization.memory_manager import is_mps_available

class SideResize:
    def __init__(
        self,
        size: int,
        max_size: int = 0,
        downsample_only: bool = False,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        self.size = size
        self.max_size = max_size
        self.downsample_only = downsample_only
        self.interpolation = interpolation
        if is_mps_available():
            self.interpolation = InterpolationMode.BILINEAR

    def __call__(self, image: Union[torch.Tensor, Image.Image]):
        """
        Resize image with shortest edge set to size, optionally limiting longest edge.
        
        Args:
            image (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image with shortest edge = size,
                                 and no edge exceeding max_size (if max_size > 0).
        """
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            raise NotImplementedError

        if self.downsample_only and min(width, height) < self.size:
            size = min(width, height)
        else:
            size = self.size

        # Resize to shortest edge (disable antialias only for MPS tensors - not supported)
        antialias = not (isinstance(image, torch.Tensor) and image.device.type == 'mps')
        
        # Calculate scale factor to avoid shared memory issues
        current_min = min(width, height)
        scale_factor = size / current_min if current_min > 0 else 1.0
        
        # For large scale factors (especially downscaling), resize in stages to avoid
        # "Too much shared memory required" error
        # PyTorch has a limit of ~49152 for shared memory, so we limit scale factor to ~8x per step
        MAX_SCALE_FACTOR = 8.0
        
        if scale_factor < 1.0 / MAX_SCALE_FACTOR or scale_factor > MAX_SCALE_FACTOR:
            # Multi-stage resize for extreme scale factors
            resized = image
            target_min = size
            
            while True:
                if isinstance(resized, torch.Tensor):
                    h, w = resized.shape[-2:]
                else:
                    w, h = resized.size
                
                current_min = min(h, w)
                if abs(current_min - target_min) < 1:
                    break
                
                # Calculate next step size (limit scale factor)
                if current_min > target_min:
                    # Downscaling
                    next_min = max(target_min, current_min / MAX_SCALE_FACTOR)
                else:
                    # Upscaling
                    next_min = min(target_min, current_min * MAX_SCALE_FACTOR)
                
                resized = TVF.resize(resized, int(round(next_min)), self.interpolation, antialias=antialias)
        else:
            # Single-stage resize for normal scale factors
            resized = TVF.resize(image, size, self.interpolation, antialias=antialias)
        
        # Apply max_size constraint if specified
        if self.max_size > 0:
            if isinstance(resized, torch.Tensor):
                h, w = resized.shape[-2:]
            else:
                w, h = resized.size
            
            if max(h, w) > self.max_size:
                scale = self.max_size / max(h, w)
                new_h, new_w = round(h * scale), round(w * scale)
                
                # Check if this resize also needs staging
                current_max = max(h, w)
                scale_factor = self.max_size / current_max
                
                if scale_factor < 1.0 / MAX_SCALE_FACTOR:
                    # Multi-stage resize for max_size constraint
                    while max(h, w) > self.max_size:
                        if isinstance(resized, torch.Tensor):
                            h, w = resized.shape[-2:]
                        else:
                            w, h = resized.size
                        
                        if max(h, w) <= self.max_size:
                            break
                        
                        # Limit scale factor per step
                        next_max = max(self.max_size, max(h, w) / MAX_SCALE_FACTOR)
                        scale = next_max / max(h, w)
                        new_h, new_w = round(h * scale), round(w * scale)
                        resized = TVF.resize(resized, (new_h, new_w), self.interpolation, antialias=antialias)
                else:
                    resized = TVF.resize(resized, (new_h, new_w), self.interpolation, antialias=antialias)
        
        return resized
