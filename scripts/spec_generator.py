import json
import random
import os
import argparse
from typing import Dict, List

# Global configuration
DEFAULT_BASE_PATH = "spec_folder"  # Default path if none specified
SEED_MIN = 1
SEED_MAX = 999999999
NEGATIVE_ELEMENTS_PER_PROMPT = 3

# Word banks for randomization
LIGHTING_STYLES = [
    "soft diffused lighting",
    "dramatic rim light",
    "moody low key",
    "cinematic split lighting",
    "golden hour glow",
    "harsh direct sunlight",
    "ethereal backlight",
    "ambient occlusion lighting",
    "volumetric god rays",
    "natural overcast lighting",
    "dappled dramatic lighting",
    "golden hour with plant gobo shadows"
]

TYPE = [
    "80's leather and steel studio",
    "Jungle setting",
    "Art Deco speakeasy with amber lighting",
    "Blade Runner-esque neon-soaked megacity",
    "Venetian palazzo at golden hour",
    "Brutalist concrete fortress in rain",
    "quirky symmetrical library",
    "Kubrickian white minimalist gallery",
    "Gothic cathedral with dust particles",
    "1970s NASA control room",
    "Japanese zen garden in morning mist",
    "Steam locomotive interior with brass details",
    "Moroccan riad with filtered sunlight",
    "Mid-century Palm Springs poolside",
    "Ancient Roman bathhouse ruins",
    "Arctic research station at twilight",
    "1920s Paris café terrace",
    "Underwater bioluminescent cave",
    "Soviet Constructivist factory",
    "Renaissance artist's workshop"
]

NAME = [
    "Apollo",
    "Hades",
    "Pluto",
    "Artemis",
    "Chronos",
    "Nyx",
    "Helios",
    "Selene",
    "Atlas"
]

STYLE = [
    "high fashion",
    "dark moody",
    "minimalist studio",
    "vintage film grain",
    "street photography",
    "glamour and neon",
    "normcore",
    "punk aesthetic",
    "high key",
    "film noir style",
    "dramatic black and white",
    "soft focus",
    "subtle low contrast",
    "low key dramatic"
]

NEGATIVE_ELEMENTS = [
    "blur, blurry", 
    "deformed, distorted",
    "low quality, jpeg artifacts",
    "oversaturated, overexposed"
]

def create_base_prompt() -> str:
    """Creates a base prompt structure with random elements."""
    lighting = random.choice(LIGHTING_STYLES)
    type = random.choice(TYPE)
    name = random.choice(NAME)
    style = random.choice(STYLE)
    
    prompt = f"A {style} product photo in {lighting} of a beauty product container with the word {name} on the label in a {type}, highly detailed, 8k resolution, conde nast, lvmh"
    return prompt

def create_negative_prompt() -> str:
    """Creates a negative prompt from random negative elements."""
    selected_negatives = random.sample(NEGATIVE_ELEMENTS, NEGATIVE_ELEMENTS_PER_PROMPT)
    return ", ".join(selected_negatives)

def create_spec(index: int) -> Dict:
    """Creates a complete specification with prompts and seed."""
    return {
        "positive": create_base_prompt(),
        "negative": create_negative_prompt(),
        "seed": random.randint(SEED_MIN, SEED_MAX)
    }

def generate_specs(num_specs: int, base_path: str) -> None:
    """
    Generates multiple specification files in numbered folders.
    
    Creates a structure like:
    base_path/
        spec_0001/
            spec.json
        spec_0002/
            spec.json
        ...
    """
    # Create base directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    for i in range(num_specs):
        # Create folder name with padded index (e.g., spec_0001)
        folder_name = f"spec_{str(i+1).zfill(4)}"
        folder_path = os.path.join(base_path, folder_name)
        
        # Create numbered folder
        os.makedirs(folder_path, exist_ok=True)
        
        # Create spec file inside the numbered folder
        spec = create_spec(i)
        spec_path = os.path.join(folder_path, "spec.json")
        
        # Write spec to file with pretty printing
        with open(spec_path, 'w') as f:
            json.dump(spec, f, indent=2)
        
        print(f"Created specification in {folder_path}/spec.json")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate prompt specification files for ComfyUI.')
    parser.add_argument('num_specs', type=int, help='Number of specifications to generate')
    parser.add_argument('--path', '-p', type=str, default=DEFAULT_BASE_PATH, 
                       help=f'Base path for generating spec folders (default: {DEFAULT_BASE_PATH})')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Generate the specs
    generate_specs(args.num_specs, args.path)
    
    print(f"\nGenerated {args.num_specs} specifications in {args.path}/")
