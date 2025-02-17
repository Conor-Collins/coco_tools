import os
import torch
import numpy as np
import tifffile
import folder_paths
from typing import Dict, List, Tuple, Union
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
                "sRGB_to_linear": ("BOOLEAN", {"default": True}),
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
    def sRGBtoLinear(np_array: np.ndarray) -> np.ndarray:
        mask = np_array <= 0.0404482362771082
        result = np_array.copy()  # Create a copy to avoid modifying the input
        result[mask] = result[mask] / 12.92
        result[~mask] = np.power((result[~mask] + 0.055) / 1.055, 2.4)
        return result

    @staticmethod
    def linearToSRGB(np_array: np.ndarray) -> np.ndarray:
        mask = np_array <= 0.0031308
        result = np_array.copy()  # Create a copy to avoid modifying the input
        result[mask] = result[mask] * 12.92
        result[~mask] = np.power(result[~mask], 1/2.4) * 1.055 - 0.055
        return result

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

    def _convert_bit_depth(self, img: np.ndarray, bit_depth: int, sRGB_to_linear: bool) -> np.ndarray:
        # For EXR files, skip sRGB conversion and normalization
        if bit_depth == 32:
            return img.astype(np.float32)

        if sRGB_to_linear:
            img = self.sRGBtoLinear(img)

        if bit_depth == 8:
            return (np.clip(img, 0, 1) * 255).astype(np.uint8)
        elif bit_depth == 16:
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
        # Convert to BGR for OpenCV formats only if not saving as grayscale
        elif file_type in ['jpg', 'jpeg', 'webp', 'png'] and img.shape[-1] >= 3:
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        
        return img

    def save_images(self, images: torch.Tensor, file_path: str, file_type: str, bit_depth: str,
                   quality: int = 95, sRGB_to_linear: bool = True, save_as_grayscale: bool = False,
                   version: int = 1, start_frame: int = 1001, frame_pad: int = 4,
                   prompt=None, extra_pnginfo=None, exr_compression: str = "zips") -> Dict:

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
                # Convert tensor to numpy without any scaling
                if file_type == "exr":
                    # Skip sRGB conversion for EXR to preserve original values
                    sRGB_to_linear = False
                    
                    # Convert from torch tensor if needed
                    if isinstance(img_tensor, torch.Tensor):
                        # Remove batch dimension if present
                        if img_tensor.ndim == 4 and img_tensor.shape[0] == 1:
                            img_np = img_tensor.squeeze(0).cpu().numpy()
                        else:
                            img_np = img_tensor.cpu().numpy()
                    else:
                        img_np = img_tensor
                    
                    sys.stderr.write(f"Initial EXR data - Shape: {img_np.shape}, Min: {img_np.min()}, Max: {img_np.max()}\n")
                    
                    # Prepare image (handles channel conversion if needed)
                    img_np = self._prepare_image_for_saving(img_np, file_type, save_as_grayscale)
                    
                    # For EXR, we want to preserve the original values without normalization
                    img_np = img_np.astype(np.float32)
                else:
                    # For non-EXR files, use standard processing
                    if isinstance(img_tensor, torch.Tensor):
                        img_np = img_tensor.cpu().numpy()
                    else:
                        img_np = img_tensor
                    
                    sys.stderr.write(f"Initial numpy min: {img_np.min()}, max: {img_np.max()}\n")
                    
                    # Convert bit depth first for non-EXR files
                    img_np = self._convert_bit_depth(img_np, bit_depth, sRGB_to_linear)
                    sys.stderr.write(f"After bit_depth conversion min: {img_np.min()}, max: {img_np.max()}\n")
                    
                    # Then prepare image
                    img_np = self._prepare_image_for_saving(img_np, file_type, save_as_grayscale)
                
                # Generate output filename
                frame_num = f".{str(start_frame + i).zfill(frame_pad)}" if file_type == "exr" else f"_{i:05d}"
                out_path = f"{base_path}{version_str}{frame_num}{file_ext}"
                
                # Use improved filename increment if file exists
                if os.path.exists(out_path):
                    out_path = self.increment_filename(out_path)

                # Save the image based on format
                if file_type == "exr":
                    try:
                        # For EXR files, handle the data directly
                        if img_np.ndim == 2 or img_np.shape[-1] == 1 or save_as_grayscale:
                            channels = 1
                            if img_np.ndim == 3 and img_np.shape[-1] > 1:
                                # Extract first channel for grayscale, preserving values
                                exr_data = img_np[..., 0:1]
                            else:
                                exr_data = img_np[..., np.newaxis] if img_np.ndim == 2 else img_np
                        else:
                            channels = img_np.shape[-1]
                            exr_data = img_np

                        sys.stderr.write(f"\nSaving EXR - Shape: {exr_data.shape}, Channels: {channels}\n")
                        sys.stderr.write(f"Value range - Min: {exr_data.min()}, Max: {exr_data.max()}\n")
                        
                        # Ensure data is float32 and contiguous
                        exr_data = np.ascontiguousarray(exr_data.astype(np.float32))
                        
                        # Create spec with detected channels
                        spec = oiio.ImageSpec(
                            exr_data.shape[1],  # width
                            exr_data.shape[0],  # height
                            channels,  # channels
                            oiio.FLOAT  # Always use FLOAT for EXR
                        )
                        
                        # Set compression level (0=none, 9=max)
                        spec.attribute("compression", exr_compression)
                        spec.attribute("Orientation", 1)
                        
                        # Add creation time and software metadata
                        spec.attribute("DateTime", datetime.now().isoformat())
                        spec.attribute("Software", "COCO Tools")
                        
                        # Create image buffer and write
                        buf = oiio.ImageBuf(spec)
                        buf.set_pixels(oiio.ROI(), exr_data)
                        
                        if not buf.write(out_path):
                            raise RuntimeError(f"Failed to write EXR: {oiio.geterror()}")
                        
                    except Exception as e:
                        raise RuntimeError(f"OpenImageIO EXR save failed: {str(e)}")

                elif file_type == "png":
                    try:
                        # Determine data type and format based on bit depth
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
                        spec.attribute("compression", "zip")  # Use ZIP compression
                        spec.attribute("png:compressionLevel", 9)  # Maximum compression
                        
                        if bit_depth == 32:
                            spec.attribute("oiio:ColorSpace", "Linear")  # Ensure linear color space for float
                        
                        # Create image buffer and write
                        buf = oiio.ImageBuf(spec)
                        buf.set_pixels(oiio.ROI(), png_data)
                        
                        if not buf.write(out_path):
                            raise RuntimeError(f"Failed to write PNG: {oiio.geterror()}")
                    
                    except Exception as e:
                        raise RuntimeError(f"OpenImageIO PNG save error: {str(e)}")

                elif file_type in ["jpg", "jpeg", "webp"]:
                    cv.imwrite(out_path, img_np, [cv.IMWRITE_JPEG_QUALITY, quality])

                elif file_type == "tiff":
                        # Convert BGR back to RGB for TIFF saving
                    if img_np.shape[-1] >= 3:
                        img_np = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)
                    # Handle TIFF saving with appropriate bit depth
                    if bit_depth == 8:
                        tifffile.imwrite(out_path, img_np.astype(np.uint8), photometric='rgb')
                    elif bit_depth == 16:
                        tifffile.imwrite(out_path, img_np.astype(np.uint16), photometric='rgb')
                    else:  # 32-bit
                        # For 32-bit, we keep the float values and save with appropriate metadata
                        tifffile.imwrite(
                            out_path,
                            img_np.astype(np.float32),
                            photometric='rgb',
                            dtype=np.float32,
                            compression='none',
                            metadata={'bit_depth': 32}
                        )

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