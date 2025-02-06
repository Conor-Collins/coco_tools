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
        Main function to load and process an EXR image using OpenImageIO.
        """
        if not OIIO_AVAILABLE:
            raise ImportError("OpenImageIO not installed. Cannot load EXR files.")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"EXR file not found: {image_path}")

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

            # Read the image data
            pixels = input_file.read_image()
            input_file.close()

            # Convert to numpy array and reshape
            img_array = np.array(pixels).reshape(height, width, channels)

            # Extract RGB and Alpha channels
            rgb_array = img_array[:, :, :3]
            alpha_array = img_array[:, :, 3] if channels > 3 else None

            # Convert to torch tensors
            rgb_tensor = torch.from_numpy(rgb_array).float().unsqueeze(0)  # [B, H, W, C]
            
            if alpha_array is not None:
                mask_tensor = torch.from_numpy(alpha_array).float().unsqueeze(0)  # [B, H, W]
            else:
                mask_tensor = torch.ones((1, height, width))  # Default mask if no alpha channel

            # Normalize if requested
            if normalize:
                rgb_tensor = (rgb_tensor - rgb_tensor.min()) / (rgb_tensor.max() - rgb_tensor.min())
                mask_tensor = mask_tensor.clamp(0, 1)

            # Prepare metadata
            metadata = {
                "file_path": image_path,
                "dimensions": f"{width}x{height}",
                "channels": channels
            }

            return rgb_tensor, mask_tensor, str(metadata)

        except Exception as e:
            logger.error(f"Error loading EXR file {image_path}: {str(e)}")
            raise

NODE_CLASS_MAPPINGS = {
    "load_exr": load_exr
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "load_exr": "Load EXR Image"
}