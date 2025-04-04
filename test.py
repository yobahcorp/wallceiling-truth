import os
import numpy as np
import torch
import requests
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import StableDiffusionInpaintPipeline
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import argparse

def download_file(url, save_path):
    """Download a file from a URL if it doesn't exist locally."""
    if not os.path.exists(save_path):
        print(f"Downloading {url} to {save_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete!")
    else:
        print(f"File already exists at {save_path}")

def setup_sam(checkpoint_path="sam_vit_h_4b8939.pth", model_type="vit_h"):
    """Download and set up the Segment Anything Model."""
    # Download SAM checkpoint if it doesn't exist
    if not os.path.exists(checkpoint_path):
        sam_url = f"https://dl.fbaipublicfiles.com/segment_anything/{checkpoint_path}"
        download_file(sam_url, checkpoint_path)
    
    # Set up SAM model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    return sam

def find_wall_segments(image, sam):
    """Use SAM to generate segments and filter for likely wall segments."""
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    
    # Sort masks by area (descending)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    # Filter for likely wall segments (typically large, rectangular segments)
    wall_masks = []
    for mask in masks:
        # Skip very small segments
        if mask['area'] < (image.shape[0] * image.shape[1]) * 0.05:
            continue
            
        # Calculate aspect ratio of the bounding box
        bbox = mask['bbox']  # [x, y, w, h]
        aspect_ratio = bbox[2] / bbox[3] if bbox[3] > 0 else 0
        
        # Skip segments that are too square (furniture) or too narrow (objects)
        if 0.2 < aspect_ratio < 5.0:
            # Check if the segment touches the edge of the image (walls often do)
            binary_mask = mask['segmentation']
            edge_contact = (
                np.any(binary_mask[0, :]) or  # Top edge
                np.any(binary_mask[-1, :]) or  # Bottom edge
                np.any(binary_mask[:, 0]) or  # Left edge
                np.any(binary_mask[:, -1])  # Right edge
            )
            
            if edge_contact:
                wall_masks.append(mask)
    
    return wall_masks

def modify_walls(image_path, prompt, output_path="modified_room.png"):
    """Main function to segment walls and modify them with Stable Diffusion."""
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Set up SAM
    sam = setup_sam()
    
    # Find wall segments
    print("Finding wall segments...")
    wall_masks = find_wall_segments(image_rgb, sam)
    
    if not wall_masks:
        print("No wall segments found!")
        return
    
    print(f"Found {len(wall_masks)} potential wall segments")
    
    # Combine wall masks into a single mask
    combined_mask = np.zeros_like(wall_masks[0]['segmentation'], dtype=bool)
    for mask in wall_masks:
        combined_mask = combined_mask | mask['segmentation']
    
    # Convert combined mask to image format for inpainting
    mask_image = Image.fromarray((~combined_mask).astype(np.uint8) * 255)
    
    # Convert original image to PIL format
    original_image = Image.fromarray(image_rgb)
    
    # Load Stable Diffusion inpainting pipeline
    print("Loading Stable Diffusion inpainting model...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Modify walls based on prompt
    print(f"Applying prompt: '{prompt}'")
    images = pipe(
        prompt=prompt,
        image=original_image,
        mask_image=mask_image,
        guidance_scale=7.5,
        num_inference_steps=30
    ).images
    
    # Save the result
    images[0].save(output_path)
    print(f"Modified image saved to {output_path}")
    
    # Also show a visualization of the wall mask
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(image_rgb)
    plt.imshow(combined_mask, alpha=0.5)
    plt.title("Detected Walls")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(images[0])
    plt.title("Modified Walls")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("wall_modification_process.png")
    plt.close()
    
    return images[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify room walls based on a prompt")
    parser.add_argument("--image", required=True, help="Path to the input room image")
    parser.add_argument("--prompt", required=True, help="Text prompt describing the desired wall modification")
    parser.add_argument("--output", default="modified_room.png", help="Path to save the output image")
    
    args = parser.parse_args()
    
    modify_walls(args.image, args.prompt, args.output)
    print("Done!")