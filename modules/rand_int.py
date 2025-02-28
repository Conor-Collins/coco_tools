import random

class rand_int:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "low": ("INT", {"default": 1, "min": -2147483648, "max": 2147483647}),
                "high": ("INT", {"default": 745, "min": -2147483648, "max": 2147483647}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "use_padding": ("BOOLEAN", {"default": False}),  # Toggle for padding
                "padding_length": ("INT", {"default": 4, "min": 1, "max": 20}),  # Amount of padding
            }
        }

    RETURN_TYPES = ("STRING", "INT",)  
    RETURN_NAMES = ("random_integer_str", "random_integer",) 
    FUNCTION = "generate_random_integer"
    CATEGORY = "COCO Tools/File Logistics"

    def generate_random_integer(self, low, high, seed, use_padding, padding_length):
        # Input validation
        if high <= low:
            raise ValueError("'high' must be greater than 'low'")
        
        # Set the random seed for reproducibility
        random.seed(seed) 
        
        # Generate the random integer
        random_int = random.randint(low, high)
        
        # Handle the string output with optional padding
        if use_padding:
            # The zfill() method adds zeros to the left of the string until it reaches the specified length
            # We convert the int to string first, then pad it
            random_int_str = str(random_int).zfill(padding_length)
        else:
            random_int_str = str(random_int)

        # Return both the string (possibly padded) and the original integer
        return (random_int_str, random_int,)