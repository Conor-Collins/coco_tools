import torch
import torch.nn.functional as F

class frequency_combine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),  # First image 
                "image2": ("IMAGE",),  # Second image
                "operation": (["add", "multiply"],), # COMBO selection for operation
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "combine_images"
    CATEGORY = "COCO Tools/Image Tools"

    def combine_images(self, image1, image2, operation):
        # Get dimensions [B,H,W,C]
        h1, w1 = image1.shape[1:3]
        h2, w2 = image2.shape[1:3]

        # Calculate differences
        h_diff = abs(h1 - h2)
        w_diff = abs(w1 - w2)

        # If difference is more than 3 pixels in either dimension
        if h_diff > 3 or w_diff > 3:
            print(f"Images size mismatch too large to process. Image1: {h1}x{w1}, Image2: {h2}x{w2}")
            raise ValueError(f"Images size mismatch too large. Difference of {h_diff}px in height and {w_diff}px in width exceeds 3px limit.")

        # If there is a small difference (3 pixels or less), resize image2 to match image1
        if h_diff > 0 or w_diff > 0:
            print(f"Small size mismatch detected. Resizing Image2 from {h2}x{w2} to {h1}x{w1}")
            # Convert from BWHC to BCHW for F.interpolate
            image2_temp = image2.permute(0, 3, 1, 2)
            # Resize
            image2_temp = F.interpolate(image2_temp, size=(h1, w1), mode='bilinear', align_corners=False)
            # Convert back to BWHC
            image2 = image2_temp.permute(0, 2, 3, 1)

        # Process the images
        if operation == "add":
            # Add images and clamp to [0,1]
            result = torch.clamp(image1 + image2, 0.0, 1.0)
        else:  # multiply
            # Multiply images (will naturally stay in [0,1] range since inputs are [0,1])
            result = image1 * image2

        return (result,)