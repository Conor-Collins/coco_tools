import os
import logging
import numpy as np
import torch
from typing import Tuple

try:
    import OpenEXR
    import Imath
    OPENEXR_AVAILABLE = True
except ImportError:
    OPENEXR_AVAILABLE = False

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
        Main function to load and process an EXR image.
        """
        if not OPENEXR_AVAILABLE:
            raise ImportError("OpenEXR or Imath not installed. Cannot load EXR files.")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"EXR file not found: {image_path}")

        try:
            exr_file = OpenEXR.InputFile(image_path)
            header = exr_file.header()

            # Check required channels
            channels_available = set(header["channels"].keys())
            missing_channels = {"R", "G", "B"} - channels_available
            if missing_channels:
                raise ValueError(f"EXR file missing required channels: {missing_channels}")

            # Get dimensions
            dw = header["dataWindow"]
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            # Read RGB channels
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            rgb_data = [
                np.frombuffer(exr_file.channel(c, pt), dtype=np.float32).reshape(height, width)
                for c in ["R", "G", "B"]
            ]
            rgb_tensor = torch.from_numpy(np.stack(rgb_data, axis=-1)).unsqueeze(0)

            # Handle alpha channel if present
            alpha_tensor = (
                torch.from_numpy(
                    np.frombuffer(exr_file.channel("A", pt), dtype=np.float32).reshape(height, width)
                ).unsqueeze(0).unsqueeze(-1) if "A" in channels_available else torch.ones_like(rgb_tensor[:, :, :, :1])
            )

            # Normalize tensors if requested
            if normalize:
                rgb_tensor = self.normalize_image(rgb_tensor)
                alpha_tensor = self.normalize_image(alpha_tensor)

            # Prepare metadata
            metadata = {
                "file_path": image_path,
                "tensor_shape": tuple(rgb_tensor.shape),
                "format": ".exr"
            }

            return rgb_tensor, alpha_tensor, str(metadata)

        except Exception as e:
            logger.error(f"Error loading EXR file {image_path}: {e}")
            raise ValueError(f"Error loading EXR file {image_path}: {e}")

    @staticmethod
    def normalize_image(image: torch.Tensor) -> torch.Tensor:
        """
        Normalize a tensor to the 0-1 range.
        """
        min_val, max_val = image.min(), image.max()
        return (image - min_val) / (max_val - min_val) if min_val != max_val else torch.zeros_like(image)


NODE_CLASS_MAPPINGS = {
    "load_exr": load_exr
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "load_exr": "Load EXR Image"
}
