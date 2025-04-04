import numpy as np
import cv2
import torch
from torchvision import transforms
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import segment_anything.utils.transforms as sam_transforms
original_apply_coords = sam_transforms.ResizeLongestSide.apply_coords

def patched_apply_coords(self, coords, original_size):
    # Force float32 instead of float64
    coords = coords.astype(np.float32)
    result = original_apply_coords(self, coords, original_size)
    return result.astype(np.float32)  # Ensure output is also float32

# Apply the monkey patch
sam_transforms.ResizeLongestSide.apply_coords = patched_apply_coords

def describe_image_segments(image, sam):
    """
    Use SAM to generate segments and describe them based on size, shape, and position.
    
    Parameters:
        image (np.ndarray): Input image.
        sam (SamPredictor or SAM object): Initialized SAM model.
    
    Returns:
        List[str]: Descriptions of image segments.
    """
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92
    )
    
    # Ensure the image is in float32 format
    # if isinstance(image, np.ndarray):
    #     image = image.astype(np.float32)
    
    print(f"Image type: {type(image)}, dtype: {image.dtype}")  # Debugging output
    
    # Generate masks
    masks = mask_generator.generate(image)
    
    # Sort masks by area (descending)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    print(masks)
    descriptions = []
    
    for i, mask in enumerate(masks):
        area_ratio = mask['area'] / (image.shape[0] * image.shape[1])
        bbox = mask['bbox']  # [x, y, w, h]
        aspect_ratio = bbox[2] / bbox[3] if bbox[3] > 0 else 0
        binary_mask = mask['segmentation']
        
        edge_contact = (
            np.any(binary_mask[0, :]) or
            np.any(binary_mask[-1, :]) or
            np.any(binary_mask[:, 0]) or
            np.any(binary_mask[:, -1])
        )

        # Build textual description
        description = f"Segment {i+1}: "
        print(description)
        # Size
        if area_ratio > 0.3:
            description += "very large, "
        elif area_ratio > 0.1:
            description += "large, "
        elif area_ratio > 0.02:
            description += "medium, "
        else:
            description += "small, "
        
        # Shape
        if aspect_ratio > 5.0:
            description += "very wide, "
        elif aspect_ratio > 2.0:
            description += "wide, "
        elif aspect_ratio < 0.5:
            description += "tall, "
        else:
            description += "moderate shape, "
        
        # Position
        if edge_contact:
            description += "touching image edge (possible wall or boundary)."
        else:
            description += "located internally (possible object or furniture)."
        
        descriptions.append(description)
    
    return descriptions

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

image = cv2.imread("test.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transform = transforms.Compose([
        transforms.ToTensor()
    ])
image = transform(image).to(torch.float32)
image = image.to(device)

# Load SAM model
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam = sam.to(device)

# Describe image
print('Describing image')
descriptions = describe_image_segments(image, sam)

# Print results
for desc in descriptions:
    print(desc)