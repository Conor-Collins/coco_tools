import torch
import numpy as np

class colorspace:
    def __init__(self):
        self.type = "colorspace"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "conversion": (["sRGB_to_Linear", "Linear_to_sRGB"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_colorspace"
    CATEGORY = "COCO Tools/Processing"

    @staticmethod
    def sRGBtoLinear(np_array: np.ndarray) -> np.ndarray:
        """Convert from sRGB to Linear colorspace."""
        mask = np_array <= 0.0404482362771082
        result = np_array.copy()  # Create a copy to avoid modifying the input
        result[mask] = result[mask] / 12.92
        result[~mask] = np.power((result[~mask] + 0.055) / 1.055, 2.4)
        return result

    @staticmethod
    def linearToSRGB(np_array: np.ndarray) -> np.ndarray:
        """Convert from Linear to sRGB colorspace."""
        mask = np_array <= 0.0031308
        result = np_array.copy()  # Create a copy to avoid modifying the input
        result[mask] = result[mask] * 12.92
        result[~mask] = np.power(result[~mask], 1/2.4) * 1.055 - 0.055
        return result

    def convert_colorspace(self, images: torch.Tensor, conversion: str) -> tuple[torch.Tensor]:
        """Convert images between sRGB and Linear colorspaces."""
        # Convert torch tensor to numpy array
        img_np = images.cpu().numpy()

        # Apply the selected conversion
        if conversion == "sRGB_to_Linear":
            result = self.sRGBtoLinear(img_np)
        else:  # Linear_to_sRGB
            result = self.linearToSRGB(img_np)

        # Convert back to torch tensor
        return (torch.from_numpy(result),)

# Register the node
NODE_CLASS_MAPPINGS = {
    "colorspace": colorspace,
}

# Specify the display name for the node
NODE_DISPLAY_NAME_MAPPINGS = {
    "colorspace": "Colorspace Converter",
}
