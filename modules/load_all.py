import os
import logging
import imageio
import numpy as np
from PIL import Image, ImageOps
import torch
from typing import Set

# Configure logging
logging.basicConfig(level=logging.INFO)

# Supported image file extensions
supported_image_extensions: Set[str] = {'.png', '.jpg', '.jpeg', '.webp', '.avif', '.tif', '.tiff', '.exr'}

class LOADALL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"forceInput": True}),  # Path input widget
                "image_file": ("STRING", {"forceInput": True, "options": []}),  # Dropdown for image selection
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "load_images"
    CATEGORY = "COCO Tools"

    def pil2tensor(self, image, bit_depth):
        """Convert a PIL Image to a PyTorch tensor, normalized based on bit depth."""
        image_np = np.array(image)
        if bit_depth == 8:
            image_tensor = torch.from_numpy(image_np.astype(np.float32) / 255.0).unsqueeze(0)
        elif bit_depth == 16:
            image_tensor = torch.from_numpy(image_np.astype(np.float32) / 65535.0).unsqueeze(0)
        elif bit_depth == 32:
            image_tensor = torch.from_numpy(image_np.astype(np.float32)).unsqueeze(0)
        else:
            logging.warning(f"Unsupported bit depth: {bit_depth}. Defaulting to 8-bit normalization.")
            image_tensor = torch.from_numpy(image_np.astype(np.float32) / 255.0).unsqueeze(0)
        return image_tensor

    def detect_bit_depth_pil(self, image):
        """Detect the bit depth of a PIL Image."""
        mode_to_bit_depth = {
            "1": 1,     # 1-bit pixels, black and white
            "L": 8,     # 8-bit pixels, grayscale
            "P": 8,     # 8-bit pixels, mapped to any other mode using a palette
            "RGB": 8,   # 8-bit pixels, true color
            "RGBA": 8,  # 8-bit pixels, true color with transparency
            "I;16": 16, # 16-bit integer pixels (greyscale)
            "I": 32,    # 32-bit signed integer pixels
            "F": 32     # 32-bit floating point pixels
        }
        return mode_to_bit_depth.get(image.mode, 8)  # Default to 8-bit if unknown

    def load_single_image(self, path):
        """Load a single image from the given path and return it as a PyTorch tensor."""
        if not os.path.exists(path):
            logging.error(f"File not found: {path}")
            return None

        try:
            # Open the image, applying EXIF orientation correction
            image = Image.open(path)
            image = ImageOps.exif_transpose(image)

            # Detect the bit depth based on the image mode
            bit_depth = self.detect_bit_depth_pil(image)
            logging.info(f"Detected bit depth for {path}: {bit_depth}-bit")

            # Convert to tensor based on bit depth
            image_tensor = self.pil2tensor(image, bit_depth)

            logging.info(f"Successfully loaded image: {path}")
            return image_tensor
        except IOError as e:
            logging.error(f"Error loading image: {e}")
            return None

    def load_exr_image(self, exr_path):
        """Load an EXR image, normalize it, and return it as a PyTorch tensor."""
        try:
            # Ensure the FreeImage plugin is installed
            imageio.plugins.freeimage.download()

            # Load the EXR file into a NumPy array
            exr_data = imageio.imread(exr_path, format='EXR-FI')

            # Normalize the EXR data to [0, 1]
            img_array = (exr_data - np.min(exr_data)) / (np.max(exr_data) - np.min(exr_data))

            # Convert the normalized EXR data to a tensor
            img_tensor = torch.from_numpy(img_array.astype(np.float32)).unsqueeze(0)  # Shape: [1, H, W, C]
            logging.info(f"Successfully loaded EXR image: {exr_path}")
            return img_tensor
        except Exception as e:
            logging.error(f"Error loading EXR file: {e}")
            return None

    def load_images(self, directory_path, image_file):
        """Load image based on file extension (EXR or other formats) and return it as a PyTorch tensor."""
        # Verify the directory exists
        if not os.path.exists(directory_path):
            logging.error(f"Directory not found: {directory_path}")
            return None

        # Full path to the selected image file
        file_path = os.path.join(directory_path, image_file)

        # Load EXR files using `load_exr_image` or other image files using `load_single_image`
        if file_path.lower().endswith('.exr'):
            return [self.load_exr_image(file_path)]
        else:
            return [self.load_single_image(file_path)]

    @classmethod
    def on_input_changed(cls, directory_path):
        """Update the available options for image files when the directory path changes."""
        try:
            # Check if the directory exists
            if not os.path.exists(directory_path):
                logging.error(f"Directory not found: {directory_path}")
                return {"image_file": {"options": []}}

            # List all files in the directory and filter for supported image formats
            files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
            image_files = [f for f in files if os.path.splitext(f)[-1].lower() in supported_image_extensions]

            return {"image_file": {"options": sorted(image_files)}}
        except Exception as e:
            logging.error(f"Error updating dropdown: {e}")
            return {"image_file": {"options": []}}

NODE_CLASS_MAPPINGS = {
    "LOADALL": LOADALL,
}
