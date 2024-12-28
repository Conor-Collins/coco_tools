import re
from typing import Tuple

class regex_find:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "regex_pattern": ("STRING", {
                    "multiline": False,
                    "default": "(?<=in\\s)(.*?)(?=,)"  # Default regex for matching text between "in" and ","
                })
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("found_match",)
    FUNCTION = "find_regex_match"
    CATEGORY = "Coco Tools"

    def find_regex_match(self, input_string: str, regex_pattern: str) -> Tuple[str]:
        try:
            # Compile the regex pattern
            pattern = re.compile(regex_pattern)
            
            # Search for the first match in the input string
            match = pattern.search(input_string)
            
            # Return the first match or an appropriate message if no matches are found
            if match:
                return (match.group(0),)
            else:
                return ("No match found",)
        
        except re.error as e:
            return (f"Regex error: {str(e)}",)

