import json
import os

class json:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
                "id_number": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"forceInput": True}),
                "output_path": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_file_path",)
    FUNCTION = "export_to_json"
    CATEGORY = "Coco Tools"
    
    def export_to_json(self, positive_prompt, negative_prompt, id_number, seed, output_path):
        try:
            # Data to be written into JSON
            data = {
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "id_number": id_number,
                "seed": seed
            }
            
            # Zero pad the ID number to ensure it is 3 digits
            padded_id = str(id_number).zfill(3)

            # Construct the directory path for the JSON file
            json_directory = os.path.join(output_path, f"{padded_id}_{seed}")
            
            # Create the directory if it does not exist
            if not os.path.exists(json_directory):
                os.makedirs(json_directory)
            
            # Define the full path for the JSON file
            json_path = os.path.join(json_directory, 'metadata.json')
            
            # Write data into JSON file
            with open(json_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            
            print(f"JSON file created successfully at: {json_path}")
            return (json_path,)
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return (None,)  # Return None or handle the error as needed

