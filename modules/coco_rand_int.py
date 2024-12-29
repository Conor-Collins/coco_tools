import random

class rand_int:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "low": ("INT", {"default": 1, "min": -2147483648, "max": 2147483647}),
                "high": ("INT", {"default": 745, "min": -2147483648, "max": 2147483647}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING", "INT",)  
    RETURN_NAMES = ("random_integer_str", "random_integer",) 
    FUNCTION = "generate_random_integer"
    CATEGORY = "COCO Tools/File Logistics"

    def generate_random_integer(self, low, high, seed):
        random.seed(seed) 
        
        if high <= low:
            raise ValueError("'high' must be greater than 'low'")
        
        random_int = random.randint(low, high)

        return (str(random_int), random_int,) 