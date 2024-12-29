import os
import json
from typing import Tuple


class json_value:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_directory": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "key_to_find": ("STRING", {
                    "multiline": False,
                    "default": ""
                })
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    FUNCTION = "find_key_in_json"
    CATEGORY = "COCO Tools/JSON Tools"

    def find_key_in_json(self, json_directory: str, key_to_find: str) -> Tuple[str]:
        try:
            # Ensure the directory exists
            if not os.path.exists(json_directory) or not os.path.isdir(json_directory):
                return (f"The provided directory path '{json_directory}' is not valid.",)

            # Find all JSON files in the directory
            json_files = [f for f in os.listdir(json_directory) if f.endswith(".json")]

            # Ensure there's only one JSON file in the directory
            if len(json_files) != 1:
                return (f"Expected exactly one JSON file in directory '{json_directory}', found {len(json_files)}.",)

            # Load the JSON file
            json_file_path = os.path.join(json_directory, json_files[0])
            with open(json_file_path, "r") as file:
                parsed_data = json.load(file)

            # Check if the key exists in the JSON structure
            if key_to_find in parsed_data:
                return (str(parsed_data[key_to_find]),)
            else:
                return (f"Key '{key_to_find}' not found",)
        
        except json.JSONDecodeError:
            return (f"Invalid JSON format in file '{json_file_path}'.",)
        except Exception as e:
            return (f"Error: {str(e)}",)