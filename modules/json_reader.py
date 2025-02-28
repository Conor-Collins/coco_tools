import os
import json

class JSON_SPEC_READER:
    def __init__(self, folder_path="D:/Default_Folder/"):
        self.folder_path = folder_path

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "D:/Default_Folder/"}),
                "name": ("STRING", {"default": "metadata"})
            }
        }

    RETURN_TYPES = ("INT", "STRING", "STRING")
    RETURN_NAMES = ("Seed", "Positive Prompt", "Negative Prompt")
    FUNCTION = "read_metadata"
    CATEGORY = "COCO Tools/JSON Tools"

    def read_metadata(self, folder_path, name):
        # Construct the full path using the custom name
        json_path = os.path.join(folder_path, f"{name}.json")
        
        # Update error message to include custom filename
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{name}.json not found in {folder_path}")

        with open(json_path, 'r') as file:
            metadata = json.load(file)

        seed = metadata.get("seed", 0)
        negative_prompt = metadata.get("negative", "MISSING KEY")
        positive_prompt = metadata.get("positive", "MISSING KEY")

        return (seed, negative_prompt, positive_prompt)