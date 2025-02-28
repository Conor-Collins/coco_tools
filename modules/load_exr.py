import os
import logging
import numpy as np
import torch
import json
from typing import Tuple, Dict, List, Optional, Union, Any

try:
    import OpenImageIO as oiio
    OIIO_AVAILABLE = True
except ImportError:
    OIIO_AVAILABLE = False

# Import Shamble class
from .load_exr_layer_by_name import load_exr_layer_by_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class load_exr:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {
                    "default": "path/to/image.exr",
                    "description": "Full path to the EXR file"
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "description": "Normalize image values to the 0-1 range"
                })
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
                "layer_data": "DICT"
            }
        }

    # Return types: RGB image, Alpha mask, Metadata string, Layer dictionary, Cryptomatte dictionary
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "LAYERS", "CRYPTOMATTE")
    RETURN_NAMES = ("image", "alpha", "metadata", "layers", "cryptomatte")
    
    FUNCTION = "load_image"
    CATEGORY = "Image/EXR"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")  # Always execute the node

    def load_image(self, image_path: str, normalize: bool = True, 
                       node_id: str = None, layer_data: Dict = None) -> List:
        """
        Load an EXR image with support for multiple layers/channel groups.
        Returns the base RGB/alpha, metadata, and dictionaries containing all layers.
        """
        
        # Check for OIIO availability
        if not OIIO_AVAILABLE:
            raise ImportError("OpenImageIO is required for EXR loading but not available")
            
        # Validate image path
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Scan the EXR metadata if not already provided
            metadata = layer_data if layer_data else self.scan_exr_metadata(image_path)
            
            # Load all pixel data
            all_data = self.load_all_data(image_path)
            
            # Extract channel groups from metadata for the main subimage
            channel_names = metadata["subimages"][0]["channel_names"]
            channel_groups = self._get_channel_groups(channel_names)
            metadata["channel_groups"] = channel_groups
            
            # Prepare outputs
            height, width, channels = all_data.shape
            
            # Dictionary to store all non-cryptomatte layers
            layers_dict = {}
            
            # Dictionary to store all cryptomatte layers
            cryptomatte_dict = {}
            
            # Default RGB output (first 3 channels if available)
            rgb_tensor = None
            if channels >= 3 and 'R' in channel_names and 'G' in channel_names and 'B' in channel_names:
                r_idx = channel_names.index('R')
                g_idx = channel_names.index('G')
                b_idx = channel_names.index('B')
                
                rgb_array = np.stack([
                    all_data[:, :, r_idx],
                    all_data[:, :, g_idx],
                    all_data[:, :, b_idx]
                ], axis=2)
                
                rgb_tensor = torch.from_numpy(rgb_array).float()
                rgb_tensor = rgb_tensor.unsqueeze(0)  # [1, H, W, 3]
                
                if normalize:
                    rgb_range = rgb_tensor.max() - rgb_tensor.min()
                    if rgb_range > 0:
                        rgb_tensor = (rgb_tensor - rgb_tensor.min()) / rgb_range
            else:
                # If no RGB channels, use first 3 channels or create placeholder
                if channels >= 3:
                    rgb_array = all_data[:, :, :3]
                else:
                    rgb_array = np.stack([all_data[:, :, 0]] * 3, axis=2)
                
                rgb_tensor = torch.from_numpy(rgb_array).float()
                rgb_tensor = rgb_tensor.unsqueeze(0)  # [1, H, W, 3]
                
                if normalize:
                    rgb_range = rgb_tensor.max() - rgb_tensor.min()
                    if rgb_range > 0:
                        rgb_tensor = (rgb_tensor - rgb_tensor.min()) / rgb_range
            
            # Default Alpha channel if available
            alpha_tensor = None
            if 'A' in channel_names:
                a_idx = channel_names.index('A')
                alpha_array = all_data[:, :, a_idx]
                
                alpha_tensor = torch.from_numpy(alpha_array).float()
                alpha_tensor = alpha_tensor.unsqueeze(0)  # [1, H, W]
                
                if normalize:
                    alpha_tensor = alpha_tensor.clamp(0, 1)
            else:
                # If no alpha, create a tensor of ones
                alpha_tensor = torch.ones((1, height, width))
            
            # Process each channel group
            for group_name, suffixes in channel_groups.items():
                # Skip the default RGB/A which are already handled separately
                if group_name in ('R', 'G', 'B', 'A'):
                    continue
                
                # Check if this is a cryptomatte layer
                is_cryptomatte = "cryptomatte" in group_name.lower() or group_name.lower().startswith("crypto")
                
                # Find all channel indices for this group
                group_indices = []
                for i, channel in enumerate(channel_names):
                    if (channel == group_name) or (channel.startswith(f"{group_name}.")):
                        group_indices.append(i)
                
                if not group_indices:
                    continue
                
                # Determine layer type and process accordingly
                if set(suffixes) >= {'R', 'G', 'B'}:  # RGB layer
                    # Find the RGB indices
                    r_idx = channel_names.index(f"{group_name}.R")
                    g_idx = channel_names.index(f"{group_name}.G")
                    b_idx = channel_names.index(f"{group_name}.B")
                    
                    # Stack RGB channels
                    rgb_array = np.stack([
                        all_data[:, :, r_idx],
                        all_data[:, :, g_idx],
                        all_data[:, :, b_idx]
                    ], axis=2)
                    
                    # Convert to torch tensor
                    rgb_tensor_layer = torch.from_numpy(rgb_array).float()
                    rgb_tensor_layer = rgb_tensor_layer.unsqueeze(0)  # [1, H, W, 3]
                    
                    # Normalize if requested
                    if normalize:
                        rgb_range = rgb_tensor_layer.max() - rgb_tensor_layer.min()
                        if rgb_range > 0:
                            rgb_tensor_layer = (rgb_tensor_layer - rgb_tensor_layer.min()) / rgb_range
                    
                    # Store in the appropriate dictionary
                    if is_cryptomatte:
                        cryptomatte_dict[group_name] = rgb_tensor_layer
                    else:
                        layers_dict[group_name] = rgb_tensor_layer
                
                # Handle XYZ vector channels (often used for normals, positions, velocity)
                elif set(suffixes) >= {'X', 'Y', 'Z'}:
                    # Find the XYZ indices
                    x_idx = channel_names.index(f"{group_name}.X")
                    y_idx = channel_names.index(f"{group_name}.Y")
                    z_idx = channel_names.index(f"{group_name}.Z")
                    
                    # Stack XYZ channels as RGB
                    xyz_array = np.stack([
                        all_data[:, :, x_idx],
                        all_data[:, :, y_idx],
                        all_data[:, :, z_idx]
                    ], axis=2)
                    
                    xyz_tensor = torch.from_numpy(xyz_array).float()
                    xyz_tensor = xyz_tensor.unsqueeze(0)  # [1, H, W, 3]
                    
                    # For vector data, normalize differently if requested
                    if normalize:
                        # Normalize based on the maximum absolute value to preserve vector directions
                        max_abs = xyz_tensor.abs().max()
                        if max_abs > 0:
                            xyz_tensor = xyz_tensor / max_abs
                    
                    # Store in the layers dictionary (vector data won't be cryptomatte)
                    layers_dict[group_name] = xyz_tensor
                
                # Handle single-channel data (like depth maps)
                elif len(group_indices) == 1:
                    # Extract the single channel
                    idx = group_indices[0]
                    channel_array = all_data[:, :, idx]
                    
                    # Check if it's likely to be a mask/depth type channel
                    is_mask_type = any(keyword in group_name.lower() 
                                      for keyword in ['depth', 'mask', 'matte', 'alpha', 'id'])
                    
                    if is_mask_type:
                        # Store as a mask tensor
                        mask_tensor = torch.from_numpy(channel_array).float().unsqueeze(0)  # [1, H, W]
                        
                        if normalize:
                            mask_range = mask_tensor.max() - mask_tensor.min()
                            if mask_range > 0:
                                mask_tensor = (mask_tensor - mask_tensor.min()) / mask_range
                        
                        layers_dict[group_name] = mask_tensor
                    else:
                        # Replicate to 3 channels for RGB visualization
                        rgb_array = np.stack([channel_array] * 3, axis=2)
                        
                        channel_tensor = torch.from_numpy(rgb_array).float()
                        channel_tensor = channel_tensor.unsqueeze(0)  # [1, H, W, 3]
                        
                        if normalize:
                            channel_range = channel_tensor.max() - channel_tensor.min()
                            if channel_range > 0:
                                channel_tensor = (channel_tensor - channel_tensor.min()) / channel_range
                        
                        layers_dict[group_name] = channel_tensor
                
                # Other multi-channel data
                else:
                    # Create a representation based on available channels (up to 3)
                    channels_to_use = min(3, len(group_indices))
                    array_channels = []
                    
                    for i in range(channels_to_use):
                        array_channels.append(all_data[:, :, group_indices[i]])
                    
                    # If we have fewer than 3 channels, duplicate the last one
                    while len(array_channels) < 3:
                        array_channels.append(array_channels[-1])
                    
                    # Stack the channels
                    multi_array = np.stack(array_channels, axis=2)
                    
                    multi_tensor = torch.from_numpy(multi_array).float()
                    multi_tensor = multi_tensor.unsqueeze(0)  # [1, H, W, 3]
                    
                    if normalize:
                        multi_range = multi_tensor.max() - multi_tensor.min()
                        if multi_range > 0:
                            multi_tensor = (multi_tensor - multi_tensor.min()) / multi_range
                    
                    # Store in the appropriate dictionary
                    if is_cryptomatte:
                        cryptomatte_dict[group_name] = multi_tensor
                    else:
                        layers_dict[group_name] = multi_tensor
            
            # Store layer type information in metadata
            layer_types = {}
            for layer_name, tensor in layers_dict.items():
                if tensor.shape[3:] == (3,):  # It has 3 channels
                    layer_types[layer_name] = "IMAGE"
                else:
                    layer_types[layer_name] = "MASK"
            
            metadata["layer_types"] = layer_types
            
            # Add metadata as JSON string
            metadata_json = json.dumps(metadata)
            
            # Log the available layers
            logger.info(f"Available EXR layers: {list(layers_dict.keys())}")
            if cryptomatte_dict:
                logger.info(f"Available cryptomatte layers: {list(cryptomatte_dict.keys())}")
            
            # Return the results
            return [rgb_tensor, alpha_tensor, metadata_json, layers_dict, cryptomatte_dict]
            
        except Exception as e:
            logger.error(f"Error loading EXR file {image_path}: {str(e)}")
            raise

    def _get_channel_groups(self, channel_names: List[str]) -> Dict[str, List[str]]:
        """
        Group channel names by their prefix (before the dot).
        Returns a dictionary of groups with their respective channel suffixes.
        """
        groups = {}
        
        for channel in channel_names:
            # Handle channels with dots (indicating a group)
            if '.' in channel:
                prefix, suffix = channel.split('.', 1)
                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append(suffix)
            else:
                # For channels without dots, use them as their own group
                if channel not in groups:
                    groups[channel] = []
                groups[channel].append(None)
                
        return groups

    def load_all_data(self, image_path: str) -> np.ndarray:
        """
        Load all pixel data from the EXR file.
        Returns a numpy array of shape (height, width, channels).
        """
        input_file = None
        try:
            input_file = oiio.ImageInput.open(image_path)
            if not input_file:
                raise IOError(f"Could not open {image_path}")
                
            # Get basic specs
            spec = input_file.spec()
            width = spec.width
            height = spec.height
            channels = spec.nchannels
            
            # Read all pixel data
            pixels = input_file.read_image()
            if pixels is None:
                raise IOError(f"Failed to read image data from {image_path}")
                
            # Convert to numpy array with correct shape
            return np.array(pixels, dtype=np.float32).reshape(height, width, channels)
            
        finally:
            if input_file:
                input_file.close()

    def scan_exr_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        Scan the EXR file to extract metadata about available subimages without loading pixel data.
        Returns a dictionary of subimage information including names, channels, dimensions, etc.
        """
        if not OIIO_AVAILABLE:
            raise ImportError("OpenImageIO not installed. Cannot load EXR files.")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"EXR file not found: {image_path}")
            
        input_file = None
        try:
            # Open the image
            input_file = oiio.ImageInput.open(image_path)
            if not input_file:
                raise IOError(f"Could not open {image_path}")
                
            metadata = {}
            subimages = []
            
            # Iterate through all subimages (layers) in the EXR
            current_subimage = 0
            more_subimages = True
            
            while more_subimages:
                # Read the spec for current subimage
                spec = input_file.spec()
                
                # Extract basic information
                width = spec.width
                height = spec.height
                channels = spec.nchannels
                channel_names = [spec.channel_name(i) for i in range(channels)]
                
                # Get subimage name if available
                subimage_name = "default"
                if "name" in spec.extra_attribs:
                    subimage_name = spec.getattribute("name")
                
                # Store subimage information
                subimage_info = {
                    "index": current_subimage,
                    "name": subimage_name,
                    "width": width,
                    "height": height,
                    "channels": channels,
                    "channel_names": channel_names
                }
                
                # Extract any additional metadata
                extra_attribs = {}
                for i in range(len(spec.extra_attribs)):
                    name = spec.extra_attribs[i].name
                    value = spec.extra_attribs[i].value
                    extra_attribs[name] = value
                
                subimage_info["extra_attributes"] = extra_attribs
                subimages.append(subimage_info)
                
                # Move to next subimage if available
                more_subimages = input_file.seek_subimage(current_subimage + 1, 0)
                current_subimage += 1
            
            metadata["subimages"] = subimages
            metadata["is_multipart"] = len(subimages) > 1
            metadata["subimage_count"] = len(subimages)
            metadata["file_path"] = image_path
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error scanning EXR metadata from {image_path}: {str(e)}")
            raise
            
        finally:
            if input_file:
                input_file.close()

NODE_CLASS_MAPPINGS = {
    "load_exr": load_exr
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "load_exr": "Load EXR"
}