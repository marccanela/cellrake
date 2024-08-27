"""
@author: Marc Canela
"""

import math
import pickle as pkl
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from csbdeep.utils import normalize
from PIL import Image
from scipy.ndimage import distance_transform_edt, zoom
from skimage import color, draw, feature, filters, measure, morphology, segmentation
from tqdm import tqdm

from cellrake.utils import convert_to_roi, crop


def iterate_segmentation(
    image_folder: Path,
) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray]]:
    """
    This function iterates over all `.tif` files in the given `image_folder`, applies a pre-trained StarDist model
    to segment the images, and extracts ROIs. The segmented layers and corresponding ROI data are stored in dictionaries
    with the image filename (without extension) as the key.

    Parameters:
    ----------
    image_folder : pathlib.Path
        A Path object pointing to the folder containing the `.tif` images to be segmented.

    Returns:
    -------
    tuple[dict[str, dict], dict[str, numpy.ndarray]]
        A tuple containing:
        - `rois`: A dictionary where keys are image filenames and values are dictionaries of ROI data.
        - `layers`: A dictionary where keys are image filenames and values are the corresponding segmented layers as NumPy arrays.
    """
    rois = {}
    layers = {}

    # Load the pre-trained StarDist model
    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    # Get a list of all .tif files in the folder
    tif_paths = list(image_folder.glob("*.tif"))

    # Iterate over each .tif file and segment the image
    for tif_path in tqdm(tif_paths, desc="Segmenting images", unit="image"):
        tag = tif_path.stem

        # Segment the image and extract ROIs
        polygon, layer = segment_image(tif_path, model)
        rois_dict = convert_to_roi(polygon, layer)

        # Store the results in the dictionaries
        rois[tag] = rois_dict
        layers[tag] = layer

    return rois, layers


def export_rois(project_folder: Path, rois: Dict[str, Dict]) -> None:
    """
    This function saves the ROIs for each image into a separate `.pkl` file within the `rois_raw` directory
    inside the specified `project_folder`. Each file is named according to the image's tag (filename without extension).

    Parameters:
    ----------
    project_folder : pathlib.Path
        A Path object pointing to the project directory where the ROIs will be saved.

    rois : dict[str, dict]
        A dictionary where keys are image tags (filenames without extension) and values are dictionaries of ROI data.

    Returns:
    -------
    None
    """
    # Define the path to the 'rois_raw' folder
    raw_folder = project_folder / "rois_raw"

    # Export each ROI dictionary to a .pkl file
    for tag, rois_dict in rois.items():
        pkl_path = raw_folder / f"{tag}.pkl"
        with open(str(pkl_path), "wb") as file:
            pkl.dump(rois_dict, file)


def segment_image(tif_path: Path, model: StarDist2D) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function reads a TIFF image from the specified path, processes it by compressing, extracting the relevant layer,
    removing empty rows and columns, and normalizing the image. The processed image layer is then segmented using a
    StarDist2D model to identify polygonal ROIs.

    Parameters:
    ----------
    tif_path : pathlib.Path
        A Path object pointing to the TIFF image to be segmented.

    model : StarDist2D
        A pre-trained StarDist2D model used to segment the image.

    Returns:
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - `polygon`: An array of coordinates representing the segmented polygonal ROIs.
        - `layer`: The processed 2D image layer from which the polygons were extracted.
    """
    # Read the image in its original form (unchanged)
    layer = np.asarray(Image.open(tif_path))

    # Eliminate rows and columns that are entirely zeros
    layer = crop(layer)

    # Compress the image
    # layer = zoom(layer, zoom=1.2, order=2)

    # Normalize the image layer
    norm_layer = normalize(layer)

    # Apply the StarDist2D model to predict instances and extract polygons
    _, polygon = model.predict_instances(
        norm_layer, prob_thresh=0.05, nms_thresh=0.4, verbose=False
    )

    return polygon, layer


blobs_log = feature.blob_log(layer, max_sigma=15, num_sigma=10, threshold=0.05)
blobs_log[:, 2] = blobs_log[:, 2] * 1.5 * math.sqrt(2)
# fig, ax = plt.subplots(1, 2, figsize=(20, 10))
# ax[0].imshow(layer)
# ax[1].imshow(layer)
# for blob in blobs_log:
#    y, x, r = blob
#    c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
#    ax[1].add_patch(c)
# plt.tight_layout()
# plt.show()

binaries = []
for blob in blobs_log:
    copy_layer = np.copy(layer)

    y, x, r = blob
    y = min(max(y, r), copy_layer.shape[0] - r)
    x = min(max(x, r), copy_layer.shape[1] - r)

    rr, cc = draw.disk((y, x), r, shape=copy_layer.shape)
    mask = np.zeros(copy_layer.shape)
    mask[rr, cc] = 1

    mask = mask == 0
    copy_layer[mask] = 0
    # plt.imshow(copy_layer);

    # Step 1: Mask out zero regions
    non_zero_mask = copy_layer > 0
    non_zero_values = copy_layer[non_zero_mask]

    # Step 2: Apply Otsu's thresholding only to the non-zero regions
    threshold = filters.threshold_otsu(non_zero_values)
    # threshold = filters.threshold_sauvola(layer, window_size=25)

    # Step 3: Apply the threshold to the entire image
    binary_image = copy_layer > threshold
    binaries.append(binary_image)

binary_image = binaries[0]
for mask in binaries[1:]:
    binary_image = np.logical_or(binary_image, mask)

# plt.imshow(binary_image)

# Step 4: Morphological operations to clean up the image
remove_holes = morphology.remove_small_holes(binary_image, area_threshold=100)
remove_particles = morphology.remove_small_objects(remove_holes, min_size=25)
circular = morphology.binary_closing(remove_particles, morphology.disk(3))
cleaned = circular
# plt.imshow(cleaned)

markers_raw = measure.label(cleaned)

# Step 6: Distance map
distance = distance_transform_edt(cleaned)
cell_radius = int(np.max(distance)) // 2

# Step 7: Identify local maxima for refined markers
coords = feature.peak_local_max(
    distance,
    min_distance=cell_radius,
    # threshold_abs=cell_radius,
    # exclude_border=cell_radius,
    footprint=morphology.disk(cell_radius),
    labels=markers_raw,
    num_peaks_per_label=5,
)

# Step 8: Create a mask from the local maxima
mask = np.zeros(layer.shape, dtype=bool)
mask[tuple(coords.T)] = True

# Step 9: Create markers for the refined watershed
markers, _ = measure.label(mask, return_num=True)

# Step 10: Apply watershed segmentation with refined markers
labels = segmentation.watershed(
    -distance, markers, mask=cleaned, watershed_line=True, compactness=1
)

# Plot the results
plt.figure(figsize=(30, 30))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(layer, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# Binary Image
plt.subplot(1, 3, 2)
plt.imshow(cleaned, cmap="gray")
plt.title("Binary Image")
plt.axis("off")

# Watershed Labels
plt.subplot(1, 3, 3)
# Overlay the labels on the original image
plt.imshow(color.label2rgb(labels, image=layer, bg_label=0))
plt.title("Watershed Segmentation")
plt.axis("off")

plt.show()
