from .modules.coco_zdepth import zdepth
from .modules.coco_json_reader import JSON_SPEC_READER
from .modules.coco_json_val_finder import json_value
from .modules.coco_json import json
from .modules.coco_noise import noise
from .modules.coco_rand_int import rand_int
from .modules.coco_regex_find import regex_find
from .modules.coco_saver import saver
from .modules.coco_split_frequencies import split_frequency_tools
from .modules.coco_walk_folder import walk_folder
from .modules.coco_load_exr import load_exr
from .modules.coco_image_loader import coco_loader

# Node Categories for better organization
NODE_CATEGORIES = {
    "Image Processing": [
        "ZDepthNode",
        "NoiseNode",
        "SplitFrequenciesNode"
    ],
    "File & Path Management": [
        "CocoImageLoader",
        "LoadEXRNode", 
        "WalkFolderNode",
        "SaverNode"
    ],
    "JSON Specs": [
        "JSONReaderNode",
        "JSONValueFinderNode",
        "JSONNode"
    ],
    "Utility Nodes": [
        "RandomIntNode",
        "RegexFindNode"
    ]
}

# Initialize node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Add all node classes to the mappings
NODE_CLASS_MAPPINGS.update({
    "ZDepthNode": zdepth,
    "NoiseNode": noise,
    "SplitFrequenciesNode": split_frequency_tools,
    "CocoImageLoader": coco_loader,
    "LoadEXRNode": load_exr,  
    "WalkFolderNode": walk_folder,
    "SaverNode": saver,
    "JSONReaderNode": JSON_SPEC_READER,
    "JSONValueFinderNode": json_value,
    "JSONNode": json,
    "RandomIntNode": rand_int,
    "RegexFindNode": regex_find
})

# Add display names for better UI presentation
NODE_DISPLAY_NAME_MAPPINGS.update({
    "ZDepthNode": "Z-Depth Reader",
    "NoiseNode": "Generate Noise",
    "SplitFrequenciesNode": "Split Frequencies",
    "CocoImageLoader": "Image Loader (Multi-Format)",
    "LoadEXRNode": "Load EXR Image",  
    "WalkFolderNode": "Walk Folder",
    "SaverNode": "Save Image",
    "JSONReaderNode": "JSON Reader",
    "JSONValueFinderNode": "JSON Value Finder",
    "JSONNode": "JSON Operations",
    "RandomIntNode": "Random Integer",
    "RegexFindNode": "Regex Find"
})

# Expose what ComfyUI needs
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "NODE_CATEGORIES"
]