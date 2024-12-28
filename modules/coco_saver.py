import os
import torch
import numpy as np
import imageio
import folder_paths
from typing import Dict, List, Tuple, Union

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv

class saver:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "file_path": ("STRING", {"default": "ComfyUI"}),
                "file_type": (["png", "jpg", "jpeg", "webp", "tiff", "exr"],),
                "bit_depth": (["8", "16", "32"],),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "sRGB_to_linear": ("BOOLEAN", {"default": True}),
                "version": ("INT", {"default": 1, "min": -1, "max": 999}),
                "start_frame": ("INT", {"default": 1001, "min": 0, "max": 99999999}),
                "frame_pad": ("INT", {"default": 4, "min": 1, "max": 8}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Coco Tools"

    def _validate_format_bitdepth(self, file_type: str, bit_depth: int) -> Tuple[str, int]:
        """Validates and adjusts file format and bit depth compatibility."""
        format_depth_limits = {
            "png": (8, 32),
            "jpg": (8, 8),
            "jpeg": (8, 8),
            "webp": (8, 8),
            "tiff": (8, 16),
            "exr": (16, 32)
        }
        
        min_depth, max_depth = format_depth_limits[file_type]
        adjusted_depth = min(max(bit_depth, min_depth), max_depth)
        
        if adjusted_depth != bit_depth:
            print(f"Warning: {file_type} format only supports {min_depth}-{max_depth} bit depth. Adjusting to {adjusted_depth} bit.")
        
        return file_type, adjusted_depth

    def _get_file_extension(self, file_type: str) -> str:
        """Returns the appropriate file extension with dot."""
        return f".{file_type}"

    def _convert_bit_depth(self, img: np.ndarray, bit_depth: int, sRGB_to_linear: bool) -> np.ndarray:
        """Converts image to specified bit depth with optional sRGB to linear conversion."""
        if sRGB_to_linear and bit_depth != 32:
            img = image_utils.sRGBtoLinear(img)

        if bit_depth == 8:
            return (np.clip(img, 0, 1) * 255).astype(np.uint8)
        elif bit_depth == 16:
            return (np.clip(img, 0, 1) * 65535).astype(np.uint16)
        else:  # 32-bit
            return img.astype(np.float32)

    def _prepare_exr_image(self, img: np.ndarray) -> np.ndarray:
        """Prepares image specifically for EXR format."""
        # Convert RGB to BGR for OpenCV
        if img.shape[-1] >= 3:
            bgr = img.copy()
            bgr[..., 0] = img[..., 2]
            bgr[..., 2] = img[..., 0]
            
            # Handle alpha channel if present
            if img.shape[-1] == 4:
                bgr[..., 3] = np.clip(1 - img[..., 3], 0, 1)
            return bgr
        return img

    def save_images(self, images: torch.Tensor, file_path: str, file_type: str, bit_depth: str,
                   quality: int = 95, sRGB_to_linear: bool = True, version: int = 1,
                   start_frame: int = 1001, frame_pad: int = 4, prompt=None, extra_pnginfo=None) -> Dict:
        try:
            bit_depth = int(bit_depth)
            file_type, bit_depth = self._validate_format_bitdepth(file_type.lower(), bit_depth)
            file_ext = self._get_file_extension(file_type)

            # Handle absolute vs relative paths
            if not os.path.isabs(file_path):
                full_output_folder, filename, _, subfolder, _ = folder_paths.get_save_image_path(
                    file_path, self.output_dir, images.shape[2], images.shape[3]
                )
                base_path = os.path.join(full_output_folder, filename)
            else:
                base_path = file_path
                os.makedirs(os.path.dirname(base_path), exist_ok=True)

            # Handle versioning
            version_str = f"_v{version:03}" if version >= 0 else ""
            
            # Process each image in the batch
            for i, img_tensor in enumerate(images):
                # Convert tensor to numpy
                img_np = img_tensor.cpu().numpy() if isinstance(img_tensor, torch.Tensor) else img_tensor
                
                # Handle grayscale images
                if len(img_np.shape) == 2 or (len(img_np.shape) == 3 and img_np.shape[-1] == 1):
                    img_np = np.expand_dims(img_np, axis=-1)

                # Generate output filename
                frame_num = f".{str(start_frame + i).zfill(frame_pad)}" if file_type == "exr" else f"_{i:05d}"
                out_path = f"{base_path}{version_str}{frame_num}{file_ext}"

                # Avoid overwriting existing files
                if os.path.exists(out_path):
                    counter = 1
                    while os.path.exists(f"{base_path}{version_str}{frame_num}_{counter}{file_ext}"):
                        counter += 1
                    out_path = f"{base_path}{version_str}{frame_num}_{counter}{file_ext}"

                # Convert bit depth and color space
                img_np = self._convert_bit_depth(img_np, bit_depth, sRGB_to_linear)

                # Save the image based on format
                if file_type == "exr":
                    img_np = self._prepare_exr_image(img_np)
                    cv.imwrite(out_path, img_np)
                elif file_type in ["jpg", "jpeg", "webp"]:
                    img_np = cv.cvtColor(img_np, cv.COLOR_RGB2BGR)
                    cv.imwrite(out_path, img_np, [cv.IMWRITE_JPEG_QUALITY, quality])
                elif file_type == "png":
                    if bit_depth == 32:
                        # For 32-bit PNG, we need to handle it specially with proper flags
                        cv.imwrite(out_path, cv.cvtColor(img_np, cv.COLOR_RGB2BGR), [cv.IMWRITE_PNG_COMPRESSION, 9])
                    else:
                        cv.imwrite(out_path, cv.cvtColor(img_np, cv.COLOR_RGB2BGR))
                elif file_type == "tiff":
                    imageio.imwrite(out_path, img_np)

            return {"ui": {"images": []}}

        except Exception as e:
            print(f"Error saving images: {e}")
            return {"ui": {"error": str(e)}}
