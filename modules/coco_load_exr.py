import os
import logging
import numpy as np
import torch
from typing import Tuple

try:
    import OpenImageIO as oiio
    OIIO_AVAILABLE = True
except ImportError:
    OIIO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class load_exr:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {
                    "default": "path/to/image.exr",
                    "description": "Full path to the EXR file"
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "description": "Normalize image values to the 0-1 range"
                })
            },
            "hidden": {"node_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "metadata")
    FUNCTION = "load_exr_image"
    CATEGORY = "COCO Tools/Loaders"

    def load_exr_image(
        self, image_path: str, normalize: bool = True, node_id: str = None
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Load and process an EXR image using OpenImageIO.
        Guarantees output format compatibility with ComfyUI:
        - Image output shape: [1, H, W, 3] (batch, height, width, RGB channels)
        - Mask output shape: [1, H, W] (batch, height, width)
        
        Supports:
        - Greyscale (1 channel)
        - RGB (3 channels)
        - RGBA (4 channels)
        """
        if not OIIO_AVAILABLE:
            raise ImportError("OpenImageIO not installed. Cannot load EXR files.")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"EXR file not found: {image_path}")

        input_file = None
        try:
            # Open the image
            input_file = oiio.ImageInput.open(image_path)
            if not input_file:
                raise IOError(f"Could not open {image_path}")

            # Read the image spec
            spec = input_file.spec()
            width = spec.width
            height = spec.height
            channels = spec.nchannels

            # Validate dimensions
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions: {width}x{height}")

            # Read the image data
            pixels = input_file.read_image()
            if pixels is None:
                raise IOError("Failed to read image data")

            # Convert to numpy array with correct shape
            img_array = np.array(pixels, dtype=np.float32).reshape(height, width, channels)

            # Handle different channel configurations
            if channels == 1:  # Greyscale
                # Duplicate single channel to RGB
                rgb_array = np.dstack([img_array] * 3)  # Shape: [H, W, 3]
                alpha_array = None
            elif channels == 3:  # RGB
                rgb_array = img_array  # Shape: [H, W, 3]
                alpha_array = None
            elif channels == 4:  # RGBA
                rgb_array = img_array[:, :, :3]  # Shape: [H, W, 3]
                alpha_array = img_array[:, :, 3]  # Shape: [H, W]
            else:
                raise ValueError(f"Unsupported number of channels: {channels}. Expected 1, 3, or 4.")

            # Verify shapes
            assert rgb_array.shape == (height, width, 3), f"Unexpected RGB shape: {rgb_array.shape}"

            # Convert to torch tensors with correct shapes for ComfyUI
            rgb_tensor = torch.from_numpy(rgb_array).float()
            rgb_tensor = rgb_tensor.unsqueeze(0)  # Add batch dimension: [1, H, W, 3]

            if alpha_array is not None:
                mask_tensor = torch.from_numpy(alpha_array).float()
                mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension: [1, H, W]
            else:
                mask_tensor = torch.ones((1, height, width))  # [1, H, W]

            # Normalize if requested
            if normalize:
                # Avoid division by zero
                rgb_range = rgb_tensor.max() - rgb_tensor.min()
                if rgb_range > 0:
                    rgb_tensor = (rgb_tensor - rgb_tensor.min()) / rgb_range
                mask_tensor = mask_tensor.clamp(0, 1)

            # Verify final tensor shapes
            assert rgb_tensor.shape == (1, height, width, 3), f"Invalid RGB tensor shape: {rgb_tensor.shape}"
            assert mask_tensor.shape == (1, height, width), f"Invalid mask tensor shape: {mask_tensor.shape}"

            metadata = {
                "dimensions": f"{width}x{height}",
                "original_channels": channels,
                "type": "greyscale" if channels == 1 else ("rgba" if channels == 4 else "rgb")
            }

            return rgb_tensor, mask_tensor, str(metadata)

        except Exception as e:
            logger.error(f"Error loading EXR file {image_path}: {str(e)}")
            raise

        finally:
            if input_file:
                input_file.close()

NODE_CLASS_MAPPINGS = {
    "load_exr": load_exr
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "load_exr": "Load EXR Image"
}