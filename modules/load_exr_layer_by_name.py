import torch
import logging
from typing import Dict, List, Union, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class load_exr_layer_by_name:
    """
    The Load EXR Layer by Name node allows selecting a specific layer from an EXR layer dictionary.
    It works like Nuke's Shuffle node, allowing users to pick a specific layer to output.
    """
    
    # Class variables to store available layer names
    available_layers = ["none"]
    
    def __init__(self):
        logger.info("load_exr_layer_by_name class initialized")

    @classmethod
    def INPUT_TYPES(cls):
        logger.info(f"INPUT_TYPES called - available layers: {cls.available_layers}")
        return {
            "required": {
                "layers": ("LAYERS",),
                "layer_name": ("STRING", {
                    "default": "none",
                    "multiline": False,
                    "description": "Name of the layer to extract from the EXR. You can find layer names in the metadata output of the Load EXR node."
                })
            },
            "optional": {
                "conversion": (["Auto", "To RGB", "To Mask"], {
                    "default": "Auto"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process_layer"
    CATEGORY = "Image/EXR"
    
    def process_layer(self, layers: Dict[str, torch.Tensor], layer_name: str, 
                     conversion: str = "Auto") -> List[Union[torch.Tensor, None]]:
        """
        Extract a specific layer from the layers dictionary.
        
        Args:
            layers: Dictionary of layer names to tensors
            layer_name: Name of the layer to extract
            conversion: How to convert the layer (Auto, To RGB, To Mask)
            
        Returns:
            List containing [image, mask] tensors
        """
        # Check if we have any layers at all
        if not layers or len(layers) == 0:
            logger.warning("No layers available in the input")
            return [torch.zeros((1, 1, 1, 3)), torch.zeros((1, 1, 1))]
            
        # If the layer doesn't exist, try to find a close match
        if layer_name not in layers and layer_name != "none":
            # Try to find a partial match
            matches = [l for l in layers.keys() if layer_name.lower() in l.lower()]
            if matches:
                layer_name = matches[0]
                logger.info(f"Layer name '{layer_name}' not found exactly, using closest match: '{layer_name}'")
            else:
                logger.warning(f"Layer '{layer_name}' not found and no close matches")
                # Use the first available layer as fallback
                if len(layers) > 0:
                    layer_name = list(layers.keys())[0]
                    logger.info(f"Using first available layer: {layer_name}")
                else:
                    return [torch.zeros((1, 1, 1, 3)), torch.zeros((1, 1, 1))]
        
        # If no layer is specified or "none" is selected, return empty tensors
        if not layer_name or layer_name == "none":
            logger.warning("No layer specified, returning empty tensors")
            return [torch.zeros((1, 1, 1, 3)), torch.zeros((1, 1, 1))]
        
        # Get the requested layer
        layer_tensor = layers[layer_name]
        
        # Check tensor shape to determine its type
        if len(layer_tensor.shape) == 4 and layer_tensor.shape[3] == 3:
            # It's an RGB tensor [1, H, W, 3]
            if conversion == "To Mask":
                # Convert RGB to mask by taking the mean across channels
                mask_output = layer_tensor.mean(dim=3, keepdim=False)
                image_output = None
            else:
                # Keep as an image
                image_output = layer_tensor
                mask_output = None
        elif len(layer_tensor.shape) == 3:
            # It's a mask/single-channel tensor [1, H, W]
            if conversion == "To RGB":
                # Convert mask to RGB by replicating to 3 channels
                image_output = torch.cat([layer_tensor.unsqueeze(3)] * 3, dim=3)
                mask_output = None
            else:
                # Keep as a mask
                mask_output = layer_tensor
                image_output = None
        else:
            # Unknown format, log error
            logger.error(f"Layer '{layer_name}' has an unsupported tensor shape: {layer_tensor.shape}")
            return [torch.zeros((1, 1, 1, 3)), torch.zeros((1, 1, 1))]
        
        # Set placeholder for any None outputs
        if image_output is None:
            image_output = torch.zeros((1, 1, 1, 3))
        if mask_output is None:
            mask_output = torch.zeros((1, 1, 1))
        
        return [image_output, mask_output]

# Define a copy of the main class for cryptomatte layers
class shamble_cryptomatte(load_exr_layer_by_name):
    """
    The Cryptomatte Shamble node allows selecting a specific cryptomatte layer from an EXR dictionary.
    It is identical to the Load EXR Layer by Name node but filters for cryptomatte layers only.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cryptomatte": ("CRYPTOMATTE",),
                "layer_name": ("STRING", {
                    "default": "none",
                    "multiline": False,
                    "description": "Name of the cryptomatte layer to extract. Look for names starting with 'crypto' in the metadata."
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_cryptomatte"
    CATEGORY = "Image/EXR"
    
    def process_cryptomatte(self, cryptomatte: Dict[str, torch.Tensor], layer_name: str) -> List[torch.Tensor]:
        """
        Extract a specific cryptomatte layer.
        
        Args:
            cryptomatte: Dictionary of cryptomatte layer names to tensors
            layer_name: Name of the cryptomatte layer to extract
            
        Returns:
            List containing the cryptomatte image tensor
        """
        # Check if we have any layers at all
        if not cryptomatte or len(cryptomatte) == 0:
            logger.warning("No cryptomatte layers available in the input")
            return [torch.zeros((1, 1, 1, 3))]
            
        # If the layer doesn't exist, try to find a close match
        if layer_name not in cryptomatte and layer_name != "none":
            # Try to find a partial match
            matches = [l for l in cryptomatte.keys() if layer_name.lower() in l.lower()]
            if matches:
                layer_name = matches[0]
                logger.info(f"Cryptomatte layer name '{layer_name}' not found exactly, using closest match: '{layer_name}'")
            else:
                logger.warning(f"Cryptomatte layer '{layer_name}' not found and no close matches")
                # Use the first available layer as fallback
                if len(cryptomatte) > 0:
                    layer_name = list(cryptomatte.keys())[0]
                    logger.info(f"Using first available cryptomatte layer: {layer_name}")
                else:
                    return [torch.zeros((1, 1, 1, 3))]
        
        # If no layer is specified or "none" is selected, return an empty tensor
        if not layer_name or layer_name == "none":
            logger.warning("No cryptomatte layer specified, returning empty tensor")
            return [torch.zeros((1, 1, 1, 3))]
        
        # Return the requested cryptomatte layer
        return [cryptomatte[layer_name]]

NODE_CLASS_MAPPINGS = {
    "load_exr_layer_by_name": load_exr_layer_by_name,
    "shamble_cryptomatte": shamble_cryptomatte
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "load_exr_layer_by_name": "Load EXR Layer by Name",
    "shamble_cryptomatte": "Cryptomatte Layer"
}