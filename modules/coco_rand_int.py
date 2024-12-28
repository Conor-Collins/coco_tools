import random

class rand_int:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "low": ("INT", {"default": 1, "min": -2147483648, "max": 2147483647}),
                "high": ("INT", {"default": 745, "min": -2147483648, "max": 2147483647}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),  # Seed input
            }
        }

    RETURN_TYPES = ("STRING", "INT",)  # Updated to return a string
    RETURN_NAMES = ("random_integer_str", "random_integer",)  # Updated name to reflect it's now a string
    FUNCTION = "generate_random_integer"
    CATEGORY = "Coco Tools"

    def generate_random_integer(self, low, high, seed):
        random.seed(seed)  # Initialize the random number generator with the provided seed
        
        if high <= low:
            raise ValueError("'high' must be greater than 'low'")
        
        # Generate a random integer then convert it to a string
        random_int = random.randint(low, high)
        return (str(random_int), random_int,)  # Convert the integer to a string before returning