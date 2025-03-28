import cv2
from data import *
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import pandas as pd
import streamlit as st
from UNET.model import Unet
from UNET.config import PRETRAINED_WEIGHTS, IMAGE_SIZE


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

    # Add channel dimension (grayscale -> 3D)
    img = np.expand_dims(img, -1)

    # Add batch dimension (for model input)
    img = np.expand_dims(img, 0)

    return img


def detect_edges(mask):
    thresholds = [0.5, 0.7, 0.8, 0.9]
    binary_masks = [
        (mask.squeeze() > threshold).astype(np.uint8) * 255 for threshold in thresholds
    ]

    # Calculate edges from the best threshold (adjust based on visual inspection)
    best_threshold_idx = 3  # Using 0.85 threshold initially, adjust if needed
    edges = cv2.Canny(binary_masks[best_threshold_idx], 100, 200)

    return edges


def findParameters(image, mask):
    (H, W) = image.shape[:2]

    if mask.shape[-1] > 1:
        output = np.argmax(mask[0], axis=-1)
    else:
        output = (mask[0] > 0.5).astype(np.uint8) * 255

    resize_output = cv2.resize(output, (W, H), interpolation=cv2.INTER_NEAREST)
    blur = cv2.GaussianBlur(resize_output, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, connectivity=4
    )

    colors = np.random.randint(0, 255, size=(n_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Set the background to black
    false_colors = colors[labels]

    # Step 10: Draw centroids on the false color image
    false_colors_centroid = false_colors.copy()
    for centroid in centroids:
        cv2.drawMarker(
            false_colors_centroid,
            (int(centroid[0]), int(centroid[1])),
            color=(255, 255, 255),
            markerType=cv2.MARKER_CROSS,
        )

    # Step 11: Filter out small objects
    MIN_AREA = 50  # Minimum area threshold
    false_colors_area_filtered = false_colors.copy()
    for i, centroid in enumerate(centroids[1:], start=1):
        area = stats[i, 4]  # Area of the object
        if area > MIN_AREA:
            cv2.drawMarker(
                false_colors_area_filtered,
                (int(centroid[0]), int(centroid[1])),
                color=(255, 255, 255),
                markerType=cv2.MARKER_CROSS,
            )

    # Extract region properties
    props = measure.regionprops_table(
        labels,
        intensity_image=image,
        properties=[
            "label",
            "area",
            "equivalent_diameter",
            "mean_intensity",
            "solidity",
        ],
    )

    # Convert to a DataFrame
    df = pd.DataFrame(props)
    return df  # Return the DataFrame instead of printing it


def run():
    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            # Read the uploaded file directly as grayscale
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

            # Check if image loaded successfully
            if image is None:
                raise ValueError("Error: Unable to load image")

            model = Unet(pretrained_weights=PRETRAINED_WEIGHTS)

            # Preprocess image for prediction
            proc_img = preprocess_image(image)

            # Predict segmentation mask
            mask = model.predict(proc_img)

            # Detect edges
            edges = detect_edges(mask)

            # Get the DataFrame from findParameters
            df = findParameters(image, mask)

            # Visualize results using Streamlit
            st.subheader("Analysis Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("Original Grayscale Image")
                st.image(proc_img.squeeze(), use_container_width=True, clamp=True)

            with col2:
                st.write("Predicted Mask")
                st.image(mask.squeeze(), use_container_width=True, clamp=True)

            with col3:
                st.write("Edges")
                st.image(edges.squeeze(), use_container_width=True, clamp=True)

            # Display image information
            st.subheader("Image Information")
            height, width = image.shape
            st.write(f"Width: {width}px, Height: {height}px")

            # Display the pandas DataFrame
            st.subheader("Region Properties")
            st.dataframe(df)  # Display DataFrame in Streamlit

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Run the Streamlit app
if __name__ == "__main__":
    run()
