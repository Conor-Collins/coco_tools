import os
import cv2 as cv
import torch
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import folder_paths
from comfy.cli_args import args

def sRGBtoLinear(np_array):
    mask = np_array <= 0.0404482362771082
    np_array[mask] = np_array[mask] / 12.92
    np_array[~mask] = np.power((np_array[~mask] + 0.055) / 1.055, 2.4)
    return np_array

def linearToSRGB(np_array):
    mask = np_array <= 0.0031308
    np_array[mask] = np_array[mask] * 12.92
    np_array[~mask] = np.power(np_array[~mask], 1/2.4) * 1.055 - 0.055
    return np_array

def is_grayscale(image):
    """Check if an RGB image is effectively grayscale."""
    if image.shape[2] == 3:
        return np.all(image[:, :, 0] == image[:, :, 1]) and np.all(image[:, :, 1] == image[:, :, 2])
    return False

class SAVEALL:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "file_path": ("STRING", {"default": "ComfyUI"}),
                "file_type": (["png", "exr"],),
                "bit_depth": (["8", "16", "32"],),
                "sRGB_to_linear": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "COCO Tools"

    def save_images(self, images, file_path, file_type, bit_depth, sRGB_to_linear, prompt=None, extra_pnginfo=None):
        try:
            bit_depth = int(bit_depth)

            # Determine file extension and adjust bit depth limits based on file type
            if file_type == "png":
                file_ext = ".png"
                bit_depth = min(max(bit_depth, 8), 16)
            elif file_type == "exr":
                file_ext = ".exr"
                bit_depth = min(max(bit_depth, 8), 32)

            # Directly use file_path to determine the output path, assuming it always includes the intended filename.
            if not os.path.isabs(file_path):
                # If file_path is not an absolute path, derive the full path using a standard method.
                full_output_folder, filename, _, subfolder, _ = folder_paths.get_save_image_path(file_path, self.output_dir, images.shape[2], images.shape[3])
                writepath = os.path.join(full_output_folder, filename + file_ext)
            else:
                # If file_path is an absolute path, it's directly used.
                writepath = file_path + file_ext

            def increment_filename(filepath):
                base, ext = os.path.splitext(filepath)
                counter = 1
                new_filepath = f"{base}_{counter:05d}{ext}"
                while os.path.exists(new_filepath):
                    counter += 1
                    new_filepath = f"{base}_{counter:05d}{ext}"
                return new_filepath

            writepath = increment_filename(writepath)

            # Assuming 'images' is a batch of images; iterate and save each.
            for i, img_tensor in enumerate(images):
                # Convert tensor to numpy if it's not already a numpy array
                img_np = img_tensor.cpu().numpy() if isinstance(img_tensor, torch.Tensor) else img_tensor

                if img_np.ndim == 3 and img_np.shape[2] == 1:  # If grayscale, add a dummy channel for compatibility
                    img_np = np.expand_dims(img_np, axis=-1)

                # Convert shape [B, H, W, C] to [H, W, C] for saving
                img_np = img_np[0] if img_np.ndim == 4 else img_np

                # For multiple images, append an index to the filename to avoid overwrites.
                indexed_writepath = f"{writepath[:-4]}_{i}{file_ext}" if len(images) > 1 else writepath

                # Handle bit depth conversion
                if bit_depth == 8:
                    img_np = (img_np * 255).astype(np.uint8)
                elif bit_depth == 16:
                    img_np = (img_np * 65535).astype(np.uint16)
                elif bit_depth == 32 and file_type == "exr":
                    img_np = img_np.astype(np.float32)

                # Convert from RGB to BGR if using OpenCV
                if file_type == "png":
                    img_np = cv.cvtColor(img_np, cv.COLOR_RGB2BGR)

                # Save the image using OpenCV
                cv.imwrite(indexed_writepath, img_np)

            return {"ui": {"images": []}}

        except Exception as e:
            print(f"Error saving images: {e}")
            return {"ui": {"error": str(e)}}

NODE_CLASS_MAPPINGS = {
    "SAVEALL": SAVEALL,
}
