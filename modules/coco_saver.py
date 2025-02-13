import os
import torch
import numpy as np
import tifffile
import folder_paths
from typing import Dict, List, Tuple, Union
import OpenImageIO as oiio
from datetime import datetime

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

        if image.shape[2] == 3:
            return np.allclose(image[..., 0], image[..., 1]) and np.allclose(image[..., 1], image[..., 2])
        return False

    def _validate_format_bitdepth(self, file_type: str, bit_depth: int) -> Tuple[str, int]:

        valid_combinations = {
            "exr": [16, 32],  # OpenEXR supports half and full float
            "png": [8, 16],
            "jpg": [8],
            "jpeg": [8],
            "webp": [8],
            "tiff": [8, 16, 32]
        }
        
        if bit_depth not in valid_combinations[file_type]:
            print(f"Warning: {file_type} format only supports {valid_combinations[file_type]} bit depth. Adjusting to {valid_combinations[file_type][0]} bit.")
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

        if sRGB_to_linear:
            img = self.sRGBtoLinear(img)

        if bit_depth == 8:
            return (np.clip(img, 0, 1) * 255).astype(np.uint8)
        elif bit_depth == 16:
            return (np.clip(img, 0, 1) * 65535).astype(np.uint16)
        else:  # 32-bit
            return img.astype(np.float32)

    def _prepare_image_for_saving(self, img: np.ndarray, file_type: str) -> np.ndarray:

        if self.is_grayscale(img) and img.shape[2] == 3:
            # Convert RGB grayscale to actual grayscale
            img = img[..., 0:1]
            
        if img.shape[-1] >= 3:
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        
        return img

    def save_images(self, images: torch.Tensor, file_path: str, file_type: str, bit_depth: str,
                   quality: int = 95, sRGB_to_linear: bool = True, version: int = 1,
                   start_frame: int = 1001, frame_pad: int = 4, prompt=None, extra_pnginfo=None, exr_compression: str = "zips") -> Dict:

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
                # Convert tensor to numpy
                img_np = img_tensor.cpu().numpy() if isinstance(img_tensor, torch.Tensor) else img_tensor
                
                # Handle grayscale images
                if len(img_np.shape) == 2 or (len(img_np.shape) == 3 and img_np.shape[-1] == 1):
                    img_np = np.expand_dims(img_np, axis=-1)

                # Generate output filename
                frame_num = f".{str(start_frame + i).zfill(frame_pad)}" if file_type == "exr" else f"_{i:05d}"
                out_path = f"{base_path}{version_str}{frame_num}{file_ext}"
                
                # Use improved filename increment if file exists
                if os.path.exists(out_path):
                    out_path = self.increment_filename(out_path)

                # Convert bit depth and prepare for saving
                img_np = self._convert_bit_depth(img_np, bit_depth, sRGB_to_linear)
                img_np = self._prepare_image_for_saving(img_np, file_type)

                # Save the image based on format
                if file_type == "exr":

                    
                    try:
                        # EXR-specific channel handling
                        if file_type == "exr":
                            # Use grayscale detection to handle single-channel
                            channels = 1 if img_np.ndim == 2 else img_np.shape[2]
                            
                            # Ensure float32 data type
                            exr_data = np.ascontiguousarray(img_np.astype(np.float32))
                            
                            # Create spec with detected channels
                            spec = oiio.ImageSpec(
                                exr_data.shape[1],  # width
                                exr_data.shape[0],  # height
                                channels,  # channels
                                "float"
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
                        # Determine data type from bit depth
                        dtype = np.uint16 if bit_depth == 16 else np.uint8
                        
                        # Convert and validate array
                        png_data = np.ascontiguousarray(img_np.astype(dtype))
                        
                        # Create image spec
                        channels = 1 if png_data.ndim == 2 else png_data.shape[2]
                        spec = oiio.ImageSpec(
                            png_data.shape[1],
                            png_data.shape[0],
                            channels,
                            "uint16" if bit_depth == 16 else "uint8"
                        )
                        
                        # Set compression level (0=none, 9=max)
                        spec.attribute("compression", "zip")
                        spec.attribute("png:compressionLevel", 9)
                        
                        # Write image
                        buf = oiio.ImageBuf(spec)
                        buf.set_pixels(oiio.ROI(), png_data)
                        
                        if not buf.write(out_path):
                            raise RuntimeError(f"PNG save failed: {oiio.geterror()}")
                    
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
            print(f"Error saving images: {e}")
            return {"ui": {"error": str(e)}}

# Register the node
NODE_CLASS_MAPPINGS = {
    "saver": saver,
}

# Optional: Register display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "saver": "Image Saver"
}