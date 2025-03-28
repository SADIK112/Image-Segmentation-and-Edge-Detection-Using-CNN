import numpy as np
import matplotlib.pyplot as plt
import cv2
from UNET.model import Unet
from UNET.config import IMAGE_SIZE, PRETRAINED_WEIGHTS

def preprocess_image(image):
    """
    Reads and preprocesses an image for model prediction.
    
    Steps:
    2. Resizes the image to (256, 256).
    3. Normalizes pixel values to [0,1].
    4. Adds channel and batch dimensions.
    
    Returns:
        numpy array: Preprocessed image with shape (1, 256, 256, 1).
    """
    # Read image in grayscale mode
    # Resize and normalize
    img = cv2.resize(image, IMAGE_SIZE) / 255.0  
    img = np.expand_dims(img, -1)  
    img = np.expand_dims(img, 0)  

    return img


def detect_edges(mask):
    thresholds = [0.5, 0.7, 0.8, 0.9]
    binary_masks = [(mask.squeeze() > threshold).astype(np.uint8) * 255 
                    for threshold in thresholds]
    
    # Calculate edges from the best threshold (adjust based on visual inspection)
    best_threshold_idx = 3  # Using 0.85 threshold initially, adjust if needed
    edges = cv2.Canny(binary_masks[best_threshold_idx], 100, 200)
    
    return edges

def modelPredictionOutcome():
    # Create model with single-channel input shape for grayscale
    model = Unet(pretrained_weights = PRETRAINED_WEIGHTS)
    image_path = "data/test/1.png"
    # Read in greyscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    # raise exception if image is not found
    if image is None:
        raise ValueError(f"Error: Unable to load image from {image_path}")
    # Preprocess image for prediction
    proc_img = preprocess_image(image)
    # Predict segmentation mask
    mask = model.predict(proc_img)
    # Detect edges
    edges = detect_edges(mask)
    # Visualize results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(proc_img.squeeze())
    plt.title('Original Grayscale Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze())
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(edges.squeeze())
    plt.title('Edges')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    return mask

if __name__ == "__main__":
    # Predict the model
    modelPredictionOutcome()