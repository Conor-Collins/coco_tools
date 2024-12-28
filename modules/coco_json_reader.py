import os
import json

class JSON_SPEC_READER:
    def __init__(self, folder_path="D:/Default_Folder/"):
        self.folder_path = folder_path

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "D:/Default_Folder/"})
            }
        }

    RETURN_TYPES = ("INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("Seed", "Asset Folder", "Negative Prompt", "Positive Prompt")
    FUNCTION = "read_metadata"
    CATEGORY = "Coco Tools"

    def read_metadata(self, folder_path):
        metadata_path = os.path.join(folder_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.json not found in {folder_path}")

        with open(metadata_path, 'r') as file:
            metadata = json.load(file)

        seed = metadata.get("seed", 0)
        asset_folder = metadata.get("folders", {}).get("asset_folder", "")
        negative_prompt = metadata.get("prompts", {}).get("negative", "")
        positive_prompt = metadata.get("prompts", {}).get("positive", "")

        return (seed, asset_folder, negative_prompt, positive_prompt)
