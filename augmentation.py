
import numpy as np
import os
from skimage import transform as trans
from skimage import exposure, color, io


def apply_all_augmentations(image, mask, 
                           images_dir='augmented/images', 
                           masks_dir='augmented/masks'):

    # Create directories if they don't exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Count existing files to determine starting index
    existing_files = len(os.listdir(images_dir))
    start_idx = existing_files
    
    augmented_pairs = []
    
    # Original image (no augmentation)
    aug_image = image.copy()
    aug_mask = mask.copy()
    augmented_pairs.append((aug_image, aug_mask))

    # 1. Rotations (90°, 180°, 270°)
    for angle in [90, 180, 270]:
        aug_image = trans.rotate(image, angle, preserve_range=True)
        aug_mask = trans.rotate(mask, angle, preserve_range=True)
        augmented_pairs.append((aug_image, aug_mask))

    # 2. Horizontal flip
    aug_image = np.fliplr(image)
    aug_mask = np.fliplr(mask)
    augmented_pairs.append((aug_image, aug_mask))

    # 3. Vertical flip
    aug_image = np.flipud(image)
    aug_mask = np.flipud(mask)
    augmented_pairs.append((aug_image, aug_mask))

    # 6. Brightness adjustment (only for image, mask remains unchanged)
    for factor in np.arange(0.5, 1, 0.1):
        aug_image = exposure.adjust_gamma(image, factor)
        aug_mask = mask.copy()
        augmented_pairs.append((aug_image, aug_mask))

    # 7. Contrast adjustment (only for image, mask remains unchanged)
    for factor in np.arange(0.5, 1, 0.1):
        aug_image = exposure.adjust_gamma(image, gamma=factor)
        aug_mask = mask.copy()
        augmented_pairs.append((aug_image, aug_mask))

    # 8. Hue adjustment if image is RGB (only for image, mask remains unchanged)
    if image.ndim == 3 and image.shape[2] >= 3:
        # Convert to HSV, adjust hue, convert back to RGB
        hsv_image = color.rgb2hsv(image)
        
        # Shift hue positively
        hsv_shifted = hsv_image.copy()
        hsv_shifted[:, :, 0] = (hsv_image[:, :, 0] + 0.2) % 1.0
        aug_image = color.hsv2rgb(hsv_shifted)
        aug_mask = mask.copy()
        augmented_pairs.append((aug_image, aug_mask))
        
        # Shift hue negatively
        hsv_shifted = hsv_image.copy()
        hsv_shifted[:, :, 0] = (hsv_image[:, :, 0] - 0.2) % 1.0
        aug_image = color.hsv2rgb(hsv_shifted)
        aug_mask = mask.copy()
        augmented_pairs.append((aug_image, aug_mask))

    # 9. Saturation adjustment if image is RGB (only for image, mask remains unchanged)
    if image.ndim == 3 and image.shape[2] >= 3:
        hsv_image = color.rgb2hsv(image)
        
        # Increase saturation
        hsv_shifted = hsv_image.copy()
        hsv_shifted[:, :, 1] = np.clip(hsv_image[:, :, 1] * 1.5, 0, 1)
        aug_image = color.hsv2rgb(hsv_shifted)
        aug_mask = mask.copy()
        augmented_pairs.append((aug_image, aug_mask))
        
        # Decrease saturation
        hsv_shifted = hsv_image.copy()
        hsv_shifted[:, :, 1] = np.clip(hsv_image[:, :, 1] * 0.5, 0, 1)
        aug_image = color.hsv2rgb(hsv_shifted)
        aug_mask = mask.copy()
        augmented_pairs.append((aug_image, aug_mask))

    # Save all augmented pairs with sequential filenames
    for i, (aug_image, aug_mask) in enumerate(augmented_pairs):
        idx = start_idx + i + 1
        image_path = os.path.join(images_dir, f"{idx}.png")
        mask_path = os.path.join(masks_dir, f"{idx}.png")
        
        # Ensure correct data type for saving
        if aug_image.dtype != np.uint8 and aug_image.max() > 1:
            aug_image = (aug_image).astype(np.uint8)
        elif aug_image.dtype != np.uint8 and aug_image.max() <= 1:
            aug_image = (aug_image * 255).astype(np.uint8)
            
        if aug_mask.dtype != np.uint8 and aug_mask.max() > 1:
            aug_mask = (aug_mask).astype(np.uint8)
        elif aug_mask.dtype != np.uint8 and aug_mask.max() <= 1:
            aug_mask = (aug_mask * 255).astype(np.uint8)
        
        # Save the images
        io.imsave(image_path, aug_image)
        io.imsave(mask_path, aug_mask)
    
    return len(augmented_pairs)  # Return number of augmented pairs created

def process_all_images():
    # Create output directories if they don't exist
    os.makedirs('augmented/images', exist_ok=True)
    os.makedirs('augmented/masks', exist_ok=True)
    
    # Get all image files from the directory, filtering out hidden files
    image_files = [f for f in sorted(os.listdir('data/train/images')) 
                  if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    mask_files = [f for f in sorted(os.listdir('data/train/masks')) 
                 if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    total_augmented = 0
    processed_count = 0
    
    # Process each image-mask pair
    for img_file in image_files:
        # Get the corresponding mask file (assuming same name)
        base_name = os.path.splitext(img_file)[0]
        mask_file = None
        
        # Find the matching mask file
        for mf in mask_files:
            if os.path.splitext(mf)[0] == base_name:
                mask_file = mf
                break
        
        if mask_file is None:
            print(f"Warning: No matching mask found for {img_file}, skipping.")
            continue
        
        try:
            # Load the image and mask
            image_path = os.path.join('data/train/images', img_file)
            mask_path = os.path.join('data/train/masks', mask_file)
            
            image = io.imread(image_path)
            mask = io.imread(mask_path)
            
            # Preprocess image
            if image.ndim == 2:  # If the image is grayscale (1 channel)
                image = np.stack([image] * 3, axis=-1)  # Convert to 3-channel RGB
            elif image.shape[-1] == 4:  # If the image has 4 channels (RGBA)
                image = color.rgba2rgb(image)  # Convert RGBA to RGB
            
            # Ensure the mask has 1 channel (grayscale)
            if mask.ndim == 3 and mask.shape[-1] > 1:
                mask = mask[..., 0]  # Take only one channel
            
            # Apply all augmentations and save them
            num_augmented = apply_all_augmentations(
                image, 
                mask,
                images_dir='augmented/images',
                masks_dir='augmented/masks'
            )
            
            total_augmented += num_augmented
            processed_count += 1
            
            # Print progress
            print(f"Processed image {processed_count}/{len(image_files)}: {img_file} - Created {num_augmented} augmentations")
        
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    print(f"Total: Created {total_augmented} augmented image-mask pairs from {processed_count} original pairs")

if __name__=='__main__':
    process_all_images()