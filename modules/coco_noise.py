import torch
import numpy as np
from opensimplex import OpenSimplex

class noise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "blend_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_type": (["perlin", "simplex", "cellular"], {"default": "perlin"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_noise"
    CATEGORY = "COCO Tools/Image Tools"

    def exclusion_blend(self, img1, img2):
        return img1 + img2 - 2 * torch.mul(img1, img2)

    def generate_channel_seeds(self, seed):
        seed_r = (seed * 1234567) % 0xffffffffffffffff
        seed_g = (seed * 7654321) % 0xffffffffffffffff
        seed_b = (seed * 1357924) % 0xffffffffffffffff
        return seed_r, seed_g, seed_b

    def generate_noise_layer(self, height, width, scale, seed, noise_type):
        """Generate a single noise layer efficiently."""
        simplex = OpenSimplex(seed)
        noise = np.zeros((height, width), dtype=np.float32)
        
        # Pre-calculate coordinates
        x_coords = np.linspace(0, width * scale, width)
        y_coords = np.linspace(0, height * scale, height)
        
        if noise_type == "perlin":
            for i, y in enumerate(y_coords):
                for j, x in enumerate(x_coords):
                    noise[i, j] = (simplex.noise2(x, y) + 1) / 2
        elif noise_type == "simplex":
            for i, y in enumerate(y_coords):
                for j, x in enumerate(x_coords):
                    noise[i, j] = (simplex.noise3(x, y, 0) + 1) / 2
        else:  # cellular
            for i, y in enumerate(y_coords):
                for j, x in enumerate(x_coords):
                    noise[i, j] = simplex.noise2(x, y)
                    
        return torch.from_numpy(noise).float()

    def apply_noise(self, image, scale=0.05, seed=0, blend_amount=0.5, noise_type="perlin"):
        # ComfyUI uses [B,H,W,C] format, need to convert
        batch_size, height, width, channels = image.shape
        device = image.device
        
        # Generate noise for each channel
        seed_r, seed_g, seed_b = self.generate_channel_seeds(seed)
        
        with torch.no_grad():
            # Generate noise for each channel
            noise_r = self.generate_noise_layer(height, width, scale, seed_r, noise_type)
            noise_g = self.generate_noise_layer(height, width, scale, seed_g, noise_type)
            noise_b = self.generate_noise_layer(height, width, scale, seed_b, noise_type)
            
            # Stack channels in ComfyUI's [B,H,W,C] format
            noise_tensor = torch.stack([noise_r, noise_g, noise_b], dim=-1)  # [H,W,C]
            noise_tensor = noise_tensor.unsqueeze(0)  # [1,H,W,C]
            noise_tensor = noise_tensor.repeat(batch_size, 1, 1, 1)  # [B,H,W,C]
            noise_tensor = noise_tensor.to(device)
            
            # Apply blending
            blended = self.exclusion_blend(image, noise_tensor)
            result = blend_amount * blended + (1 - blend_amount) * image
            
            # Ensure proper value range
            result = torch.clamp(result, 0, 1)
            
        return (result,)