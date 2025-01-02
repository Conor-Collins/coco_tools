import torch

class znormalize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Changed to accept IMAGE tensor
                "min_depth": ("FLOAT", {
                    "default": 0.0, 
                    "min": -10000.0,
                    "max": 10000.0,
                    "step": 0.01,
                    "description": "Minimum depth value for normalization"
                }),
                "max_depth": ("FLOAT", {
                    "default": 1.0,
                    "min": -10000.0,
                    "max": 10000.0,
                    "step": 0.01,
                    "description": "Maximum depth value for normalization"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normalized_depth_image",)
    FUNCTION = "normalize_depth"
    CATEGORY = "COCO Tools/Processing"

    def normalize_depth(self, image, min_depth, max_depth):
        """
        Normalize depth image tensor.
        
        Args:
            image: Input tensor in [B,H,W,C] format
            min_depth: Minimum depth value for normalization
            max_depth: Maximum depth value for normalization
            
        Returns:
            Normalized tensor in [B,H,W,C] format
        """
        try:
            # Create a copy to avoid modifying the input
            normalized = image.clone()
            
            # Normalize depth values
            normalized = (normalized - min_depth) / (max_depth - min_depth)
            
            # Clip values to [0,1] range
            normalized = torch.clamp(normalized, 0.0, 1.0)
            
            # If the input is single channel, repeat it across RGB
            if normalized.shape[-1] == 1:
                normalized = normalized.repeat(1, 1, 1, 3)
                
            return (normalized,)

        except Exception as e:
            print(f"Error normalizing depth image: {e}")
            return (None,)