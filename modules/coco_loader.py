"""
Extended Image Loader Node for ComfyUI with resize options
"""

import os
import logging
import numpy as np
import torch
from PIL import Image, ImageOps
import OpenEXR
import Imath
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class image_loader:
    """Image loader node with resize options."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {
                    "default": "",
                    "description": "Full path to image file"
                }),
                "channel_mode": (["RGB", "R", "G", "B", "A", "Z", "Luminance"], {
                    "default": "RGB",
                    "description": "Channel selection for EXR images"
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "description": "Normalize image values to 0-1 range"
                }),
                "resize_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "description": "Target width (0 for no resize)"
                }),
                "resize_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "description": "Target height (0 for no resize)"
                }),
                "resize_mode": (["stretch", "crop", "pad"], {
                    "default": "stretch",
                    "description": "How to handle aspect ratio differences"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "metadata")
    FUNCTION = "load_image"
    CATEGORY = "image/loaders"

    def normalize_image(self, image: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
        """Normalize image tensor to specified range."""
        if image is None:
            raise ValueError("Input image cannot be None")
        
        min_current = torch.min(image)
        max_current = torch.max(image)
        
        if min_current == max_current:
            return torch.full_like(image, min_val)
        
        normalized = (image - min_current) / (max_current - min_current)
        scaled = normalized * (max_val - min_val) + min_val
        
        return scaled

    def resize_tensor(self, tensor: torch.Tensor, width: int, height: int, mode: str) -> torch.Tensor:
        """Resize image tensor using specified mode."""
        if width <= 0 or height <= 0:
            return tensor
        
        # Get current dimensions
        c, h, w = tensor.shape
        
        if mode == "stretch":
            # Simple resize
            resized = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
        elif mode == "crop":
            # Calculate resize to maintain aspect ratio while ensuring image is large enough
            target_ratio = width / height
            current_ratio = w / h
            
            if current_ratio > target_ratio:
                # Image is wider than target
                resize_height = height
                resize_width = int(height * current_ratio)
            else:
                # Image is taller than target
                resize_width = width
                resize_height = int(width / current_ratio)
            
            # Resize first
            resized = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(resize_height, resize_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            # Then crop
            start_y = (resize_height - height) // 2
            start_x = (resize_width - width) // 2
            resized = resized[:, start_y:start_y+height, start_x:start_x+width]
            
        else:  # pad
            # Calculate resize to maintain aspect ratio while fitting within target
            target_ratio = width / height
            current_ratio = w / h
            
            if current_ratio > target_ratio:
                # Image is wider than target ratio
                resize_width = width
                resize_height = int(width / current_ratio)
            else:
                # Image is taller than target ratio
                resize_height = height
                resize_width = int(height * current_ratio)
            
            # Resize first
            resized = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(resize_height, resize_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            # Create padding tensor
            padded = torch.zeros(c, height, width)
            
            # Calculate padding
            pad_y = (height - resize_height) // 2
            pad_x = (width - resize_width) // 2
            
            # Place resized image in center
            padded[:, pad_y:pad_y+resize_height, pad_x:pad_x+resize_width] = resized
            
            resized = padded
        
        return resized

    def load_exr_channels(self, exr_path: str, channel_mode: str) -> torch.Tensor:
        """Load specific channels from EXR file."""
        if not OpenEXR.isOpenExrFile(exr_path):
            raise ValueError(f"Invalid EXR file: {exr_path}")
        
        exr_file = OpenEXR.InputFile(exr_path)
        header = exr_file.header()
        dw = header['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        
        # Determine channels based on mode
        if channel_mode == "RGB":
            channels = ['R', 'G', 'B']
        elif channel_mode in ["R", "G", "B", "A"]:
            channels = [channel_mode]
        elif channel_mode == "Z":
            channels = ['Z']
        elif channel_mode == "Luminance":
            channels = ['R', 'G', 'B']
        
        channel_data = []
        for channel in channels:
            try:
                data = np.frombuffer(exr_file.channel(channel, pt), dtype=np.float32)
                channel_data.append(data.reshape(size[1], size[0]))
            except Exception as e:
                logger.warning(f"Could not read channel {channel}: {e}")
        
        if not channel_data:
            raise ValueError("No valid channels found")
        
        # Handle luminance conversion
        if channel_mode == "Luminance" and len(channel_data) == 3:
            rgb = np.stack(channel_data, axis=-1)
            luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            channel_data = [luminance]
        
        # Stack channels and convert to tensor
        if len(channel_data) == 1:
            # Single channel - expand to 3 channels for RGB
            image_array = np.stack([channel_data[0]]*3, axis=-1)
        else:
            image_array = np.stack(channel_data, axis=-1)
        
        # Convert to tensor and ensure [B,H,W,C] format
        image_tensor = torch.from_numpy(image_array).float()
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return image_tensor

    def load_regular_image(self, path: str) -> torch.Tensor:
        """Load regular image formats (PNG, JPG, etc.)."""
        image = Image.open(path)
        image = ImageOps.exif_transpose(image)  # Handle EXIF orientation
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array and normalize
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_np)
        
        # ComfyUI expects [B,H,W,C] format
        # PIL/numpy gives us [H,W,C], so we need to add batch dimension
        if len(image_tensor.shape) == 3:  # [H,W,C]
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension -> [1,H,W,C]
        elif len(image_tensor.shape) == 2:  # [H,W]
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(-1)  # Add batch and channel -> [1,H,W,1]
            
        return image_tensor

    def load_image(
        self,
        image_path: str,
        channel_mode: str = "RGB",
        normalize: bool = True,
        resize_width: int = 0,
        resize_height: int = 0,
        resize_mode: str = "stretch"
        ) -> Tuple[torch.Tensor, str]:

        """Main loading function that handles all image loading scenarios."""
        
        try:
            # Validate path
            if not os.path.isfile(image_path):
                raise ValueError(f"Invalid file path: {image_path}")
            
            # Load image based on format
            if image_path.lower().endswith('.exr'):
                image = self.load_exr_channels(image_path, channel_mode)
            else:
                image = self.load_regular_image(image_path)
            
            if normalize:
                image = self.normalize_image(image)
            
            # Handle resizing
            if resize_width > 0 and resize_height > 0:
                image = self.resize_tensor(image, resize_width, resize_height, resize_mode)
            
            # Ensure we have a batch dimension and correct channel order
            if len(image.shape) == 3:
                image = image.unsqueeze(0)  # Add batch dimension if not present
            
            # Double check tensor shape
            if len(image.shape) != 4:
                raise ValueError(f"Invalid image tensor shape: {image.shape}")
            
            # Ensure float32 type
            image = image.float()
            
            # Prepare metadata
            metadata = {
                "file_path": image_path,
                "channel_mode": channel_mode,
                "image_size": image.shape,
                "resize_mode": resize_mode if resize_width > 0 and resize_height > 0 else "none"
            }
            
            return (image, str(metadata))
        
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs) -> float:
        """Check if the image file has been modified."""
        try:
            image_path = kwargs.get('image_path', '')
            if not os.path.isfile(image_path):
                return float("nan")
            return os.path.getmtime(image_path)
        except:
            return float("nan")

NODE_CLASS_MAPPINGS = {
    "image_loader": image_loader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "image_loader": "Load Image (Extended)"
}