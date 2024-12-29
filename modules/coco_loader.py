import os
import logging
import numpy as np
import torch
from PIL import Image, ImageOps
import tifffile

try:
    import OpenEXR
    import Imath
    OPENEXR_AVAILABLE = True
except ImportError:
    OPENEXR_AVAILABLE = False

from typing import Tuple, Optional, Set
from server import PromptServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add supported image formats validation
supported_image_extensions: Set[str] = {
    '.png', '.jpg', '.jpeg', '.webp', '.avif', '.tif', '.tiff', '.exr'
}


class image_loader:
    MAX_RESOLUTION = 8192
    MEMORY_WARNING_SIZE = 4096
    
    # For EXR, we might want to check presence of required channels
    EXR_REQUIRED_CHANNELS = {'R', 'G', 'B'}  # Minimum required channels
    EXR_OPTIONAL_CHANNELS = {'A', 'Z'}       # Optional channels

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {
                    "default": "path/to/image.png",
                    "description": "Full path to image file"
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "description": "Normalize image values to 0-1 range"
                })
            },
            "hidden": {"node_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "metadata")

    FUNCTION = "load_image"
    CATEGORY = "image/loaders"

    @staticmethod
    def process_channel(channel: torch.Tensor) -> torch.Tensor:
        """
        Example helper if you want a method that processes a single channel into
        a 3-channel shape. Currently unused, but you can adapt if you need it.
        """
        return channel.unsqueeze(-1).expand(-1, -1, -1, 3)

    @staticmethod
    def detect_bit_depth(image_path: str, image: Image.Image = None) -> dict:
        """
        Detect bit depth from PIL image or from path if 'image' not provided.
        Returns a dict with keys: bit_depth, mode, format
        """
        mode_to_bit_depth = {
            "1": 1,     # binary
            "L": 8,     # grayscale
            "P": 8,     # palette
            "RGB": 8,   # RGB
            "RGBA": 8,  # RGBA
            "I;16": 16, # 16-bit integer
            "I": 32,    # 32-bit signed integer
            "F": 32     # 32-bit float
        }

        # Open a temporary PIL image if not provided
        if image is None:
            with Image.open(image_path) as img:
                mode = img.mode
                fmt = img.format
        else:
            mode = image.mode
            fmt = image.format

        bit_depth = mode_to_bit_depth.get(mode, 8)

        return {
            "bit_depth": bit_depth,
            "mode": mode,
            "format": fmt
        }

    @staticmethod
    def pil2tensor(image: Image.Image, bit_depth: int) -> torch.Tensor:
        """
        Convert a PIL Image to a normalized PyTorch tensor.
        """
        image_np = np.array(image)
        if bit_depth == 8:
            image_tensor = torch.from_numpy(image_np.astype(np.float32) / 255.0)
        elif bit_depth == 16:
            image_tensor = torch.from_numpy(image_np.astype(np.float32) / 65535.0)
        elif bit_depth == 32:
            # Already float, but we can ensure float32
            image_tensor = torch.from_numpy(image_np.astype(np.float32))
        else:
            logger.warning(f"Unsupported bit depth: {bit_depth}. Defaulting to 8-bit normalization.")
            image_tensor = torch.from_numpy(image_np.astype(np.float32) / 255.0)
        
        # Add batch dimension if not present
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
            
        return image_tensor

    def check_dimensions(self, width: int, height: int, node_id: str) -> None:
        """
        Check image dimensions and send warnings if necessary.
        """
        if width > self.MAX_RESOLUTION or height > self.MAX_RESOLUTION:
            msg = (
                f"Image dimensions ({width}x{height}) exceed "
                f"maximum allowed size of {self.MAX_RESOLUTION}"
            )
            PromptServer.instance.send_sync("image_loader_warning", {
                "node_id": node_id,
                "message": msg
            })
            
        if width > self.MEMORY_WARNING_SIZE or height > self.MEMORY_WARNING_SIZE:
            msg = (
                f"Large image detected ({width}x{height}). "
                "This may consume significant memory."
            )
            PromptServer.instance.send_sync("image_loader_warning", {
                "node_id": node_id,
                "message": msg
            })

    def normalize_image(self, image: torch.Tensor,
                        min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
        """
        Normalize an image tensor into the specified range [min_val, max_val].
        """
        if image is None:
            raise ValueError("Input image cannot be None")
        
        min_current = torch.min(image)
        max_current = torch.max(image)
        
        # Edge case for constant images
        if min_current == max_current:
            return torch.full_like(image, min_val)
        
        # Normalize
        normalized = (image - min_current) / (max_current - min_current)
        scaled = normalized * (max_val - min_val) + min_val
        return scaled

    def load_regular_image(self, path: str, node_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load PNG/JPG/WEBP/AVIF etc. via PIL, handle exif transpose, alpha -> mask, etc.
        """
        try:
            with Image.open(path) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)

                # Check dimensions
                self.check_dimensions(pil_img.width, pil_img.height, node_id)

                # If alpha present, separate it out
                has_alpha = pil_img.mode in ('RGBA', 'LA')
                if has_alpha:
                    if pil_img.mode == 'RGBA':
                        rgb_image = pil_img.convert('RGB')
                        alpha = pil_img.split()[3]
                    else:  # LA mode
                        rgb_image = pil_img.convert('RGB')
                        alpha = pil_img.split()[1]
                else:
                    rgb_image = pil_img.convert('RGB')
                    alpha = None

                # Determine bit depth from the converted (RGB) image
                info = self.detect_bit_depth(path, rgb_image)
                bit_depth = info['bit_depth']

                # Convert RGB to tensor
                rgb_tensor = self.pil2tensor(rgb_image, bit_depth)
                
                # Create mask tensor
                if alpha is not None:
                    mask_tensor = self.pil2tensor(alpha, bit_depth)
                    # Ensure mask is [B,H,W,1]
                    if len(mask_tensor.shape) == 3:
                        mask_tensor = mask_tensor.unsqueeze(-1)
                else:
                    # If no alpha, make a mask of all 1s
                    mask_tensor = torch.ones_like(rgb_tensor[:, :, :, :1])

                return rgb_tensor, mask_tensor

        except Exception as e:
            logger.error(f"Error loading regular image: {str(e)}")
            raise

    def load_exr_channels(self, exr_path: str, node_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load an EXR file’s channels into a PyTorch tensor. Returns (RGB, alpha).
        """
        if not OPENEXR_AVAILABLE:
            raise ImportError("OpenEXR or Imath not installed. Cannot load EXR.")

        exr_file = None
        try:
            exr_file = OpenEXR.InputFile(exr_path)
            header = exr_file.header()

            # Check channels
            channels_available = set(header['channels'].keys())
            missing = self.EXR_REQUIRED_CHANNELS - channels_available
            if missing:
                raise ValueError(
                    f"EXR missing one or more required channels {missing}. "
                    f"Available channels: {channels_available}"
                )

            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            self.check_dimensions(width, height, node_id)

            pt = Imath.PixelType(Imath.PixelType.FLOAT)

            # Read RGB
            rgb_data = []
            for c in ['R', 'G', 'B']:
                channel_data = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
                rgb_data.append(channel_data.reshape(height, width))

            # Stack into [H, W, 3]
            rgb_tensor = torch.from_numpy(np.stack(rgb_data, axis=-1)).unsqueeze(0)

            # Alpha channel if present
            if 'A' in channels_available:
                alpha_data = np.frombuffer(exr_file.channel('A', pt), dtype=np.float32)
                alpha_tensor = torch.from_numpy(alpha_data.reshape(height, width)).unsqueeze(0).unsqueeze(-1)
            else:
                alpha_tensor = torch.ones_like(rgb_tensor[:, :, :, :1])

            return rgb_tensor, alpha_tensor

        finally:
            if exr_file is not None:
                exr_file.close()

    def load_tiff(self, path: str, node_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a TIFF image with tifffile, handle alpha if present.
        """
        try:
            with tifffile.TiffFile(path) as tiff:
                page = tiff.pages[0]
                image_data = page.asarray()

                height, width = image_data.shape[:2]
                self.check_dimensions(width, height, node_id)

                # Normalize by dtype
                dtype = image_data.dtype
                if dtype == np.uint8:
                    image_data = image_data.astype(np.float32) / 255.0
                elif dtype == np.uint16:
                    image_data = image_data.astype(np.float32) / 65535.0
                elif dtype in (np.float32, np.float64):
                    image_data = image_data.astype(np.float32)
                else:
                    logger.warning(f"Unknown TIFF dtype {dtype}. Will attempt float32 cast / 255.")
                    image_data = image_data.astype(np.float32) / 255.0

                # Handle channel dimension
                if len(image_data.shape) == 2:
                    # Grayscale, replicate to 3 channels
                    image_data = np.stack([image_data] * 3, axis=-1)
                    mask_data = np.ones((height, width, 1), dtype=np.float32)
                else:
                    # shape => [H, W, channels]
                    channels = image_data.shape[2]
                    if channels >= 4:
                        # alpha is last channel
                        mask_data = image_data[..., 3:4]
                        image_data = image_data[..., :3]
                    else:
                        # no alpha
                        mask_data = np.ones((height, width, 1), dtype=np.float32)

                rgb_tensor = torch.from_numpy(image_data).unsqueeze(0)
                mask_tensor = torch.from_numpy(mask_data).unsqueeze(0)

                return rgb_tensor, mask_tensor

        except Exception as e:
            logger.error(f"Error loading TIFF: {str(e)}")
            raise

    def load_image(self,
                   image_path: str,
                   normalize: bool = True,
                   node_id: str = None
                   ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Main loading function that routes to the appropriate loader 
        based on file extension. 
        """
        if not os.path.exists(image_path):
            msg = f"File not found: {image_path}"
            logger.error(msg)
            if node_id:
                PromptServer.instance.send_sync("image_loader_error", {
                    "node_id": node_id,
                    "message": msg
                })
            raise FileNotFoundError(msg)

        # Optional check for supported extensions
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in supported_image_extensions:
            msg = f"Unsupported image format: {ext}"
            logger.error(msg)
            if node_id:
                PromptServer.instance.send_sync("image_loader_error", {
                    "node_id": node_id,
                    "message": msg
                })
            raise ValueError(msg)

        try:
            if ext == '.exr':
                rgb_tensor, mask_tensor = self.load_exr_channels(image_path, node_id)
            elif ext in {'.tif', '.tiff'}:
                rgb_tensor, mask_tensor = self.load_tiff(image_path, node_id)
            else:
                rgb_tensor, mask_tensor = self.load_regular_image(image_path, node_id)

            # Normalize if requested
            if normalize:
                rgb_tensor = self.normalize_image(rgb_tensor)
                mask_tensor = self.normalize_image(mask_tensor)

            # Prepare metadata
            metadata = {
                "file_path": image_path,
                "tensor_shape": tuple(rgb_tensor.shape),
                "format": ext
            }
            # Return everything as strings or Tensors
            return (rgb_tensor, mask_tensor, str(metadata))

        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            if node_id:
                PromptServer.instance.send_sync("image_loader_error", {
                    "node_id": node_id,
                    "message": f"Error loading image: {str(e)}"
                })
            raise

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs) -> float:
        """
        Check if the file changed. Return NaN if not found or error. 
        Otherwise return last modification time as float, so Comfy can 
        decide if node needs re-execution.
        """
        try:
            image_path = kwargs.get('image_path', '')
            if not os.path.isfile(image_path):
                return float('nan')
            return os.path.getmtime(image_path)
        except:
            return float('nan')


NODE_CLASS_MAPPINGS = {
    "image_loader": image_loader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "image_loader": "Load Image (multi-file-type)"
}
