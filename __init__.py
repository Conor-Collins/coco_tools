import sys
import inspect
from .modules.zdepth import zdepth
from .modules.noise import noise
from .modules.split_threebands import split_threebands
from .modules.image_loader import coco_loader
from .modules.load_exr import load_exr
from .modules.walk_folder import walk_folder
from .modules.saver import saver
from .modules.json_reader import JSON_SPEC_READER
from .modules.json_value import json_value
from .modules.json import json
from .modules.rand_int import rand_int
from .modules.regex_find import regex_find
from .modules.frequency_separation import frequency_separation
from .modules.frequency_combine import frequency_combine
from .modules.znormalize import znormalize
from .modules.colorspace import colorspace
from .modules.load_exr_layer_by_name import load_exr_layer_by_name, shamble_cryptomatte

# Print deprecation notice on module import
print("\n⚠️  [DEPRECATED] ComfyUI-CoCoTools nodes are deprecated. Please use ComfyUI-CoCoTools_IO instead: https://github.com/Conor-Collins/ComfyUI-CoCoTools_IO\n", file=sys.stderr)

# Function to wrap node classes with deprecation warnings
def add_deprecation_warning(node_class, node_name):
    # Store the original FUNCTION_INPUTS dictionary if it exists
    original_inputs = getattr(node_class, "FUNCTION_INPUTS", {})
    
    # Store the original process method
    original_process = node_class.process if hasattr(node_class, "process") else None
    
    # Define a new process method that prints a deprecation warning
    def process_with_warning(self, *args, **kwargs):
        print(f"⚠️  [DEPRECATED] Using deprecated node: {node_name}. Please use ComfyUI-CoCoTools_IO instead.", file=sys.stderr)
        # Call the original process method
        if original_process:
            return original_process(self, *args, **kwargs)
        return None
    
    # Replace the process method with our wrapped version
    if original_process:
        node_class.process = process_with_warning
    
    # Return the modified class
    return node_class

# Initialize node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Explicitly set the web directory path relative to this file
import os
NODE_DIR = os.path.dirname(os.path.realpath(__file__))
WEB_DIRECTORY = os.path.join(NODE_DIR, "js")

# Wrap each node class with deprecation warning
zdepth = add_deprecation_warning(zdepth, "Z Depth")
znormalize = add_deprecation_warning(znormalize, "Z Normalize")
noise = add_deprecation_warning(noise, "Noise Generator")
split_threebands = add_deprecation_warning(split_threebands, "Split 3 Bands")
coco_loader = add_deprecation_warning(coco_loader, "Image Loader")
load_exr = add_deprecation_warning(load_exr, "Load EXR")
walk_folder = add_deprecation_warning(walk_folder, "Walk Folder")
saver = add_deprecation_warning(saver, "Save Image")
JSON_SPEC_READER = add_deprecation_warning(JSON_SPEC_READER, "JSON Reader")
json_value = add_deprecation_warning(json_value, "JSON Value Finder")
json = add_deprecation_warning(json, "JSON")
rand_int = add_deprecation_warning(rand_int, "Random Int")
regex_find = add_deprecation_warning(regex_find, "Regex Find")
frequency_separation = add_deprecation_warning(frequency_separation, "Frequency Separation")
frequency_combine = add_deprecation_warning(frequency_combine, "Frequency Combine")
colorspace = add_deprecation_warning(colorspace, "Colorspace")
load_exr_layer_by_name = add_deprecation_warning(load_exr_layer_by_name, "Load EXR Layer by Name")
shamble_cryptomatte = add_deprecation_warning(shamble_cryptomatte, "Cryptomatte Layer")

# Add all node classes to the mappings
NODE_CLASS_MAPPINGS.update({
    "ZDepthNode": zdepth,
    "ZNormalizeNode": znormalize,
    "NoiseNode": noise,
    "SplitThreeBandsNode": split_threebands,
    "ImageLoader": coco_loader,
    "LoadExr": load_exr,  
    "WalkFolderNode": walk_folder,
    "SaverNode": saver,
    "JSONReaderNode": JSON_SPEC_READER,
    "JSONValueFinderNode": json_value,
    "JSONNode": json,
    "RandomIntNode": rand_int,
    "RegexFindNode": regex_find,
    "FrequencySeparation": frequency_separation,
    "FrequencyCombine": frequency_combine,
    "ColorspaceNode": colorspace,
    "LoadExrLayerByName": load_exr_layer_by_name,
    "CryptomatteLayer": shamble_cryptomatte
})

# Add display names for better UI presentation (all deprecated)
NODE_DISPLAY_NAME_MAPPINGS.update({
    "ZDepthNode": "(DEPRECATED) Z Depth",
    "ZNormalizeNode": "(DEPRECATED) Z Normalize",
    "NoiseNode": "(DEPRECATED) Noise Generator",
    "SplitThreeBandsNode": "(DEPRECATED) Split 3 Bands",
    "ImageLoader": "(DEPRECATED) Image Loader",
    "LoadExr": "(DEPRECATED) Load EXR", 
    "WalkFolderNode": "(DEPRECATED) Walk Folder",
    "SaverNode": "(DEPRECATED) Save Image",
    "JSONReaderNode": "(DEPRECATED) JSON Reader",
    "JSONValueFinderNode": "(DEPRECATED) JSON Value Finder",
    "JSONNode": "(DEPRECATED) JSON",
    "RandomIntNode": "(DEPRECATED) Random Int",
    "RegexFindNode": "(DEPRECATED) Regex Find",
    "FrequencySeparation": "(DEPRECATED) Frequency Separation",
    "FrequencyCombine": "(DEPRECATED) Frequency Combine",
    "ColorspaceNode": "(DEPRECATED) Colorspace",
    "LoadExrLayerByName": "(DEPRECATED) Load EXR Layer by Name",
    "CryptomatteLayer": "(DEPRECATED) Cryptomatte Layer"
})

# Expose what ComfyUI needs
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
