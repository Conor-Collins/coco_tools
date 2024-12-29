import torch
import torch.nn.functional as F
import numpy as np

class split_frequency_tools:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI will handle input forcing
                "low_freq_radius": ("INT", {"default": 5, "min": 1, "max": 100}),
                "medium_freq_radius": ("INT", {"default": 10, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("Low Frequency Image", "Medium Frequency Image", "High Frequency Image")
    FUNCTION = "split_frequencies"
    CATEGORY = "COCO Tools/Image Tools"

    def gaussian_kernel(self, kernel_size, sigma):
        """Create 2D Gaussian kernel."""
        x = torch.linspace(-sigma, sigma, kernel_size)
        x = x.view(1, -1).repeat(kernel_size, 1)
        y = x.t()
        gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        return gaussian / gaussian.sum()

    def gaussian_blur(self, image, kernel_size, sigma):
        """Apply Gaussian blur to image tensor."""
        # Create Gaussian kernel
        kernel = self.gaussian_kernel(kernel_size, sigma)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.to(image.device)

        # Convert from [B,H,W,C] to [B,C,H,W]
        image = image.permute(0, 3, 1, 2)
        
        # Pad the image
        pad_size = kernel_size // 2
        pad_size = min(pad_size, min(image.shape[2:]) - 1)
        
        # Process each batch
        output = []
        for b in range(image.shape[0]):
            channels = []
            for c in range(image.shape[1]):
                channel = image[b:b+1, c:c+1]
                channel = F.pad(channel, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
                blurred = F.conv2d(channel, kernel, padding=0, groups=1)
                channels.append(blurred)
            output.append(torch.cat(channels, dim=1))
        
        output = torch.cat(output, dim=0)
        
        # Convert back to [B,H,W,C]
        output = output.permute(0, 2, 3, 1)
        return output

    def split_frequencies(self, image, low_freq_radius, medium_freq_radius):
        """Split image into frequency components."""
        # Input validation
        if low_freq_radius >= medium_freq_radius:
            raise ValueError("Low frequency radius must be smaller than medium frequency radius")
            
        # Ensure reasonable kernel sizes
        low_kernel_size = min(2 * low_freq_radius + 1, min(image.shape[1:3]))
        medium_kernel_size = min(2 * medium_freq_radius + 1, min(image.shape[1:3]))
        
        # Ensure odd kernel sizes
        low_kernel_size = low_kernel_size if low_kernel_size % 2 == 1 else low_kernel_size - 1
        medium_kernel_size = medium_kernel_size if medium_kernel_size % 2 == 1 else medium_kernel_size - 1
        
        # Apply Gaussian blur for different frequencies
        low_freq = self.gaussian_blur(image, low_kernel_size, low_freq_radius)
        medium_freq_blur = self.gaussian_blur(image, medium_kernel_size, medium_freq_radius)
        
        # Calculate frequency components
        high_freq = image - low_freq
        medium_freq = medium_freq_blur - low_freq
        
        # Ensure proper value range
        low_freq = torch.clamp(low_freq, 0, 1)
        medium_freq = torch.clamp(medium_freq, 0, 1)
        high_freq = torch.clamp(high_freq, 0, 1)
        
        return (low_freq, medium_freq, high_freq)