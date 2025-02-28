import numpy as np
import torch

class zdepth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "exr_path": ("STRING", {"forceInput": True}),
                "min_depth": ("FLOAT", {"default": 0.0, "description": "Minimum depth value for normalization."}),
                "max_depth": ("FLOAT", {"default": 1.0, "description": "Maximum depth value for normalization."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Normalized_Depth_Image",)
    FUNCTION = "load_exr_image"
    CATEGORY = "COCO Tools/Loaders"

    def load_exr_image(self, exr_path, min_depth, max_depth):
        try:
            # Load the EXR image using utility function
            depth_tensor = image_utils.load_exr_image(exr_path)

            if depth_tensor is not None:
                # Normalize depth using configurable min/max depth values and ensure efficient data handling
                ch_data_normalized = (depth_tensor - min_depth) / (max_depth - min_depth)
                ch_data_normalized = np.clip(ch_data_normalized, 0.0, 1.0)  # Ensure values are within [0, 1]

                # Scale to 0-255 range directly in float32 for efficiency
                ch_data_rgb = ch_data_normalized * 255.0

                # Duplicate single-channel depth data across RGB channels
                ch_data_rgb = np.stack([ch_data_rgb] * 3, axis=-1)  # Shape [H, W, 3]
                
                # Convert to tensor and add batch dimension directly from float32 data
                depth_tensor = torch.tensor(ch_data_rgb, dtype=torch.float32).unsqueeze(0)  # Shape [B=1, H, W, C=3]
                print("Loaded and normalized depth image as RGB tensor.")

            else:
                depth_tensor = None
                print("Expected channel 'Z.R' not found; cannot load depth image.")

            return (depth_tensor,)

        except Exception as e:
            print(f"Error loading EXR file with OpenEXR: {e}")
            return (None,)
