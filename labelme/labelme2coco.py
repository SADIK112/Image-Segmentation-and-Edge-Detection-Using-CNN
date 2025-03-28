import os
import numpy as np
import cv2
from pathlib import Path
from pycocotools.coco import COCO

# Define input and output directories
img_dir = "labelme/labelme_annot/input/"
annFile = "labelme/labelme_annot/runs/labelme2coco/dataset.json"

# Load COCO annotations
coco = COCO(annFile)

# Get category and annotation IDs
catIds = coco.getCatIds()
annsIds = coco.getAnnIds()

# Create folders for each category
output_base = "labelme/labelme_annot/output/"
for cat in catIds:
    category_name = coco.loadCats(cat)[0]['name']
    Path(os.path.join(output_base, category_name)).mkdir(parents=True, exist_ok=True)

# Loop over each image and accumulate masks
for img_id in coco.getImgIds():  # Loop over each image ID
    # Initialize a blank mask for this image
    mask = np.zeros((coco.loadImgs(img_id)[0]['height'], coco.loadImgs(img_id)[0]['width']), dtype=np.uint8)
    
    # Get the annotations for this image
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
    
    for ann in anns:
        # Add the mask for each annotation
        mask += coco.annToMask(ann)  # Accumulate the mask

    # Convert binary mask to uint8 for OpenCV processing
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Ensure mask is 2D grayscale
    if len(mask_uint8.shape) == 3:
        mask_uint8 = cv2.cvtColor(mask_uint8, cv2.COLOR_BGR2GRAY)

    # Find contours for boundary detection
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty black mask for boundaries (ensure it has correct shape and type)
    boundary_mask = np.zeros(mask_uint8.shape, dtype=np.uint8)

    # Draw contours with gray boundary (gray boundary instead of white)
    cv2.drawContours(boundary_mask, contours, -1, 128, thickness=2)  # Adjust thickness as needed (gray boundary)

    # Get the category name and image details for file saving
    image_info = coco.loadImgs([img_id])[0]
    
    # Define output file path for saving boundary masks
    # We'll save the boundary mask in the category folder of the image
    for ann in anns:
        category_name = coco.loadCats(ann['category_id'])[0]['name']
        file_path = os.path.join(output_base, category_name, os.path.splitext(image_info['file_name'])[0] + ".png")
        
        # Ensure directory exists for saving boundary mask
        Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)

        # Save the boundary mask to the designated path
        cv2.imwrite(file_path, boundary_mask)
        print(f"Saved boundary mask for image: {image_info['file_name']} at {file_path}")
