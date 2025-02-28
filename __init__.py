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

# Initialize node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Explicitly set the web directory path relative to this file
import os
NODE_DIR = os.path.dirname(os.path.realpath(__file__))
WEB_DIRECTORY = os.path.join(NODE_DIR, "js")

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

# Add display names for better UI presentation
NODE_DISPLAY_NAME_MAPPINGS.update({
    "ZDepthNode": "Z Depth",
    "ZNormalizeNode": "Z Normalize",
    "NoiseNode": "Noise Generator",
    "SplitThreeBandsNode": "Split 3 Bands",
    "ImageLoader": "Image Loader",
    "LoadExr": "Load EXR", 
    "WalkFolderNode": "Walk Folder",
    "SaverNode": "Save Image",
    "JSONReaderNode": "JSON Reader",
    "JSONValueFinderNode": "JSON Value Finder",
    "JSONNode": "JSON",
    "RandomIntNode": "Random Int",
    "RegexFindNode": "Regex Find",
    "FrequencySeparation": "Frequency Separation",
    "FrequencyCombine": "Frequency Combine",
    "ColorspaceNode": "Colorspace",
    "LoadExrLayerByName": "Load EXR Layer by Name",
    "CryptomatteLayer": "Cryptomatte Layer"
})

# Expose what ComfyUI needs
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]