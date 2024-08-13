"""
@author: Marc Canela
"""

import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
from csbdeep.utils import normalize
from stardist import export_imagej_rois
from stardist.models import StarDist2D
from tqdm import tqdm
from utils import compress, convert_to_roi, crop, get_layer


def iterate_segmentation(
    image_folder: Path,
) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
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


def export_rois(project_folder: Path, rois: dict[str, dict]) -> None:
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


def segment_image(tif_path: Path, model: StarDist2D) -> tuple[np.ndarray, np.ndarray]:
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
    image = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)

    # Compress the image (optional)
    image = compress(image)

    # Extract the relevant layer from the image
    layer = get_layer(image)

    # Eliminate rows and columns that are entirely zeros
    layer = crop(layer)

    # Normalize the image layer
    norm_layer = normalize(layer)

    # Apply the StarDist2D model to predict instances and extract polygons
    _, polygon = model.predict_instances(
        norm_layer, prob_thresh=0.5, nms_thresh=0.4, verbose=False
    )

    return polygon, layer
