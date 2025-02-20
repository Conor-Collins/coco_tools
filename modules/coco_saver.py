import os
import torch
import numpy as np
import tifffile
import folder_paths
from typing import Dict, Tuple
import OpenImageIO as oiio
from datetime import datetime
import sys

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
                "file_type": (["exr", "png", "jpg", "jpeg", "webp", "tiff"],),
                "bit_depth": (["8", "16", "32"], {"default": "16"}),
                "exr_compression": (["none", "zip", "zips", "rle", "pxr24", "b44", "b44a", "dwaa", "dwab"], {"default": "zips"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "save_as_grayscale": ("BOOLEAN", {"default": False}),
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
    CATEGORY = "COCO Tools/Savers"

    @staticmethod
    def is_grayscale(image: np.ndarray) -> bool:
        """Check if an image is grayscale by either being single channel or having identical RGB channels."""
        if len(image.shape) == 2 or image.shape[-1] == 1:
            return True
        if image.shape[-1] == 3:
            return np.allclose(image[..., 0], image[..., 1]) and np.allclose(image[..., 1], image[..., 2])
        return False

    def _validate_format_bitdepth(self, file_type: str, bit_depth: int) -> Tuple[str, int]:
        valid_combinations = {
            "exr": [16, 32],  # OpenEXR supports half and full float
            "png": [8, 16, 32],  # Now supporting 32-bit PNGs through OpenImageIO
            "jpg": [8],
            "jpeg": [8],
            "webp": [8],
            "tiff": [8, 16, 32]
        }
        
        if bit_depth not in valid_combinations[file_type]:
            sys.stderr.write(f"Warning: {file_type} format only supports {valid_combinations[file_type]} bit depth. Adjusting to {valid_combinations[file_type][0]} bit.\n")
            bit_depth = valid_combinations[file_type][0]
        
        return file_type, bit_depth

    def increment_filename(self, filepath: str) -> str:
        base, ext = os.path.splitext(filepath)
        counter = 1
        new_filepath = f"{base}_{counter:05d}{ext}"
        while os.path.exists(new_filepath):
            counter += 1
            new_filepath = f"{base}_{counter:05d}{ext}"
        return new_filepath

    def _convert_bit_depth(self, img: np.ndarray, bit_depth: int) -> np.ndarray:
        """Convert image to specified bit depth."""
        if bit_depth == 8:
            return (np.clip(img, 0, 1) * 255).astype(np.uint8)
        elif bit_depth == 16:
            if isinstance(img.dtype, np.floating):
                # For floating point data (like EXR), preserve as float16
                return img.astype(np.float16)
            else:
                # For integer data, scale to 16-bit range
                return (np.clip(img, 0, 1) * 65535).astype(np.uint16)
        else:  # 32-bit
            return img.astype(np.float32)

    def _prepare_image_for_saving(self, img: np.ndarray, file_type: str, save_as_grayscale: bool = False) -> np.ndarray:
        """Prepare image for saving by handling color space and channel conversion."""
        # Handle single channel images
        if len(img.shape) == 2:
            img = img[..., np.newaxis]
        
        # Convert to grayscale if requested or if image is already grayscale
        if save_as_grayscale or self.is_grayscale(img):
            if img.shape[-1] == 3:
                # For EXR workflow, just take R channel to preserve original values
                img = img[..., 0:1]
        # Convert to BGR for OpenCV formats only (jpg, jpeg, webp) if not saving as grayscale
        elif file_type in ['jpg', 'jpeg', 'webp'] and img.shape[-1] >= 3:
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        
        return img

    def _prepare_image_data(self, img_tensor: torch.Tensor, file_type: str, bit_depth: int, 
                           save_as_grayscale: bool) -> np.ndarray:
        """Prepare image data for saving by handling conversion and preprocessing."""
        # Convert from torch tensor if needed
        if isinstance(img_tensor, torch.Tensor):
            # Remove batch dimension if present
            if img_tensor.ndim == 4 and img_tensor.shape[0] == 1:
                img_np = img_tensor.squeeze(0).cpu().numpy()
            else:
                img_np = img_tensor.cpu().numpy()
        else:
            img_np = img_tensor
        
        sys.stderr.write(f"Initial data - Shape: {img_np.shape}, Min: {img_np.min()}, Max: {img_np.max()}\n")
        
        # Prepare image (handles channel conversion if needed)
        img_np = self._prepare_image_for_saving(img_np, file_type, save_as_grayscale)
        
        # Convert bit depth for all formats
        img_np = self._convert_bit_depth(img_np, bit_depth)
        sys.stderr.write(f"After bit_depth conversion min: {img_np.min()}, max: {img_np.max()}\n")
        
        return img_np

    def _save_exr(self, img_np: np.ndarray, out_path: str, save_as_grayscale: bool, exr_compression: str, bit_depth: int) -> None:
        """Save image data as EXR file using OpenImageIO."""
        try:
            # Determine channels and prepare data
            if img_np.ndim == 2 or img_np.shape[-1] == 1 or save_as_grayscale:
                channels = 1
                if img_np.ndim == 3 and img_np.shape[-1] > 1:
                    exr_data = img_np[..., 0:1]  # Extract first channel for grayscale
                else:
                    exr_data = img_np[..., np.newaxis] if img_np.ndim == 2 else img_np
            else:
                channels = img_np.shape[-1]
                exr_data = img_np

            sys.stderr.write(f"\nSaving EXR - Shape: {exr_data.shape}, Channels: {channels}\n")
            sys.stderr.write(f"Value range - Min: {exr_data.min()}, Max: {exr_data.max()}\n")
            
            # Ensure data is contiguous and in the right format
            if bit_depth == 16:
                exr_data = np.ascontiguousarray(exr_data.astype(np.float16))
                pixel_type = oiio.HALF
            else:  # 32-bit
                exr_data = np.ascontiguousarray(exr_data.astype(np.float32))
                pixel_type = oiio.FLOAT
            
            # Create spec with detected channels
            spec = oiio.ImageSpec(
                exr_data.shape[1],  # width
                exr_data.shape[0],  # height
                channels,  # channels
                pixel_type  # Use appropriate pixel type based on bit depth
            )
            
            # Set EXR-specific attributes
            spec.attribute("compression", exr_compression)
            spec.attribute("Orientation", 1)
            spec.attribute("DateTime", datetime.now().isoformat())
            spec.attribute("Software", "COCO Tools")
            
            # Create and write image buffer
            buf = oiio.ImageBuf(spec)
            buf.set_pixels(oiio.ROI(), exr_data)
            
            if not buf.write(out_path):
                raise RuntimeError(f"Failed to write EXR: {oiio.geterror()}")
            
        except Exception as e:
            raise RuntimeError(f"OpenImageIO EXR save failed: {str(e)}")

    def _save_png(self, img_np: np.ndarray, out_path: str, bit_depth: int) -> None:
        """Save image data as PNG file using OpenImageIO."""
        try:
            # Determine data type based on bit depth
            if bit_depth == 8:
                dtype = "uint8"
                pixel_type = np.uint8
            elif bit_depth == 16:
                dtype = "uint16"
                pixel_type = np.uint16
            else:  # 32-bit
                dtype = "float"
                pixel_type = np.float32
            
            # Convert and validate array
            png_data = np.ascontiguousarray(img_np.astype(pixel_type))
            
            # Create image spec
            channels = 1 if png_data.ndim == 2 else png_data.shape[-1]
            spec = oiio.ImageSpec(
                png_data.shape[1],  # width
                png_data.shape[0],  # height
                channels,  # channels
                dtype
            )
            
            # Set PNG-specific attributes
            spec.attribute("compression", "zip")
            spec.attribute("png:compressionLevel", 9)
            
            if bit_depth == 32:
                spec.attribute("oiio:ColorSpace", "Linear")
            
            # Create and write image buffer
            buf = oiio.ImageBuf(spec)
            buf.set_pixels(oiio.ROI(), png_data)
            
            if not buf.write(out_path):
                raise RuntimeError(f"Failed to write PNG: {oiio.geterror()}")
        
        except Exception as e:
            raise RuntimeError(f"OpenImageIO PNG save error: {str(e)}")

    def _save_jpeg_or_webp(self, img_np: np.ndarray, out_path: str, quality: int) -> None:
        """Save image data as JPEG or WebP file using OpenCV."""
        cv.imwrite(out_path, img_np, [cv.IMWRITE_JPEG_QUALITY, quality])

    def _save_tiff(self, img_np: np.ndarray, out_path: str, bit_depth: int) -> None:
        """Save image data as TIFF file using tifffile."""
        # Convert BGR back to RGB for TIFF saving
        if img_np.shape[-1] >= 3:
            img_np = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)
        
        # Handle TIFF saving with appropriate bit depth
        if bit_depth == 8:
            tifffile.imwrite(out_path, img_np.astype(np.uint8), photometric='rgb')
        elif bit_depth == 16:
            tifffile.imwrite(out_path, img_np.astype(np.uint16), photometric='rgb')
        else:  # 32-bit
            tifffile.imwrite(
                out_path,
                img_np.astype(np.float32),
                photometric='rgb',
                dtype=np.float32
            )

    def save_images(self, images: torch.Tensor, file_path: str, file_type: str, bit_depth: str,
                   quality: int = 95, save_as_grayscale: bool = False,
                   version: int = 1, start_frame: int = 1001, frame_pad: int = 4,
                   prompt=None, extra_pnginfo=None, exr_compression: str = "zips") -> Dict:
        """Save a batch of images in various formats with support for versioning and frame numbering."""
        try:
            bit_depth = int(bit_depth)
            file_type, bit_depth = self._validate_format_bitdepth(file_type.lower(), bit_depth)
            file_ext = f".{file_type}"

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
                # Prepare image data
                img_np = self._prepare_image_data(img_tensor, file_type, bit_depth, save_as_grayscale)
                
                # Generate output filename
                frame_num = f".{str(start_frame + i).zfill(frame_pad)}" if file_type == "exr" else f"_{i:05d}"
                out_path = f"{base_path}{version_str}{frame_num}{file_ext}"
                
                # Use improved filename increment if file exists
                if os.path.exists(out_path):
                    out_path = self.increment_filename(out_path)

                # Save the image based on format
                if file_type == "exr":
                    self._save_exr(img_np, out_path, save_as_grayscale, exr_compression, bit_depth)
                elif file_type == "png":
                    self._save_png(img_np, out_path, bit_depth)
                elif file_type in ["jpg", "jpeg", "webp"]:
                    self._save_jpeg_or_webp(img_np, out_path, quality)
                elif file_type == "tiff":
                    self._save_tiff(img_np, out_path, bit_depth)

            # Return success
            return {"ui": {"images": []}}

        except Exception as e:
            sys.stderr.write(f"Error saving images: {e}\n")
            return {"ui": {"error": str(e)}}

# Register the node
NODE_CLASS_MAPPINGS = {
    "saver": saver,
}

# Optional: Register display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "saver": "Image Saver"
}