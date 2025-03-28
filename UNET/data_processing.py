import os
import numpy as np
import skimage.io as io
import skimage.transform as trans
from UNET.config import BATCH_SIZE, TRAIN_IMG_NUM, MASK_FOLDER, IMAGE_FOLDER, TRAIN_DIR, VALIDATION_DIR, VALIDATION_IMG_NUM, IMAGE_SIZE, IS_GRAY_IMAGE

def mask_proccess(mask):
    """Process a mask by thresholding and inverting values."""
    for i in range(256):
        for j in range(256):
            if mask[i, j] > 0.001:
                mask[i, j] = 0
            else:
                mask[i, j] = 1
    return mask

def trainGenerator(batch_size = BATCH_SIZE, train_path= TRAIN_DIR, image_folder = IMAGE_FOLDER,
                   mask_folder = MASK_FOLDER, image_num = TRAIN_IMG_NUM, target_size = IMAGE_SIZE):
    """Generate batches of images and masks for training or validation."""
    image_dir = os.path.join(train_path, image_folder)
    mask_dir = os.path.join(train_path, mask_folder)
    i = 0
    while True:
        image_batch = []
        mask_batch = []
        image_count = i
        for _ in range(batch_size):
            image = io.imread(os.path.join(image_dir, "%s.png" % image_count), as_gray = IS_GRAY_IMAGE)
            image = trans.resize(image, target_size)
            image_batch.append(image)
            
            # Load mask as grayscale (1 channel)
            mask = io.imread(os.path.join(mask_dir, "%s.png" % image_count), as_gray = IS_GRAY_IMAGE)
            mask = trans.resize(mask, target_size)
            mask = mask_proccess(mask)
            mask = np.reshape(mask, (*target_size, 1))
            mask_batch.append(mask)
            
            image_count += 1
            if image_count >= image_num:
                image_count = 0

        # Convert lists to NumPy arrays directly
        image_batch = np.array(image_batch)  # Shape: (batch_size, 256, 256, 1)
        mask_batch = np.array(mask_batch)    # Shape: (batch_size, 256, 256, 1)
        
        i += 1
        if i >= image_num:
            i = 0
        yield (image_batch, mask_batch)

def get_train_dataset():
    """Return a training data generator."""
    return trainGenerator(
        batch_size = BATCH_SIZE,
        train_path = TRAIN_DIR,
        image_folder = IMAGE_FOLDER,
        mask_folder = MASK_FOLDER,
        image_num = TRAIN_IMG_NUM,
    )

def get_val_dataset():
    """Return a validation data generator."""
    return trainGenerator(
        batch_size = BATCH_SIZE,
        train_path = VALIDATION_DIR,
        image_folder = IMAGE_FOLDER,
        mask_folder = MASK_FOLDER,
        image_num = VALIDATION_IMG_NUM,
    )