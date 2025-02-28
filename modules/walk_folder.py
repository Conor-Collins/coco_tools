import os
from typing import Tuple

class walk_folder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0}),
                "file_extension": ("STRING", {"default": "*", "description": "File extension filter (e.g., 'png', 'jpg'). Use * for all files"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "file_info")
    FUNCTION = "walk_files"
    CATEGORY = "COCO Tools/File Logistics"

    def walk_files(self, folder_path: str, seed: int, file_extension: str) -> Tuple[str, str]:
        # Ensure the folder path exists
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            raise ValueError(f"The provided folder path '{folder_path}' is not valid.")

        # Get all files in the directory
        files = []
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path):
                # Filter by extension if specified
                if file_extension == "*" or file.lower().endswith(f".{file_extension.lower()}"):
                    files.append(file)

        # Sort files alphabetically
        files.sort()

        # Check if there are any files
        if not files:
            raise ValueError(f"No files found in '{folder_path}' matching extension '.{file_extension}'")

        # Ensure seed is within valid range
        if seed >= len(files):
            raise ValueError(f"Seed value {seed} is out of range. There are only {len(files)} files.")

        # Select the file based on the seed
        selected_file = files[seed]
        selected_file_path = os.path.join(folder_path, selected_file)

        # Create the file info string
        file_info = (
            f"Total files: {len(files)}, "
            f"First file: {files[0]}, "
            f"Last file: {files[-1]}, "
            f"Current file: {selected_file}"
        )

        return (selected_file_path, file_info)