"""
@author: Marc Canela
"""

import pickle as pkl
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tqdm import tqdm

from cellrake.predicting import iterate_predicting
from cellrake.segmentation import iterate_segmentation
from cellrake.training import active_learning, create_subset_df
from cellrake.utils import build_project, crop


def look_for_segmentation(project_folder, image_folder, threshold_rel):
    """
    Check for existing segmentation results or perform segmentation on images.

    This function checks if a segmentation file already exists for the specified
    `project_folder`. If it exists, it loads the file and returns the segmentation data.
    If not, it performs segmentation on the images in `image_folder` based on a given
    threshold, saves the results, and returns them.

    Parameters
    ----------
    project_folder : Path
        The directory path where the project files are stored, including any
        existing segmentation data.
    image_folder : Path
        The directory path containing the images to be segmented.
    threshold_rel : float
        The relative threshold used for segmentation in `iterate_segmentation`.

    Returns
    -------
    rois : dict
        A dictionary containing segmented regions of interest (ROIs) for each image.
    layers : dict
        A dictionary containing the processed layers for each image in `image_folder`.
    """
    # Check if there is an existing segmentation
    rois_path = project_folder / f"{image_folder.stem}_segmentation.pkl"
    if rois_path.exists():
        print(f"Existing segmentation detected.")

        # Open segmentation
        try:
            with open(rois_path, "rb") as file:
                rois = pkl.load(file)
        except Exception as e:
            raise RuntimeError(f"Error loading existing segmentation file: {e}")

        # Create layers dictionary
        layers = {}
        for tif_path in tqdm(
            list(image_folder.glob("*.tif")), desc="Openining images", unit="image"
        ):
            try:
                layer = np.asarray(Image.open(tif_path))
                layer = crop(layer)
                layer = layer.astype(np.uint8)
                tag = tif_path.stem
                layers[tag] = layer
            except Exception as e:
                raise RuntimeError(f"Error processing image {tif_path.stem}: {e}")

    else:
        # Segment images to obtain two dictionaries: 'rois' and 'layers'
        try:
            rois, layers = iterate_segmentation(image_folder, threshold_rel)
        except Exception as e:
            raise RuntimeError(f"Error during segmentation: {e}")

        # Save the rois
        try:
            with open(rois_path, "wb") as file:
                pkl.dump(rois, file)
        except Exception as e:
            raise RuntimeError(f"Error saving segmentation results: {e}")

    return rois, layers


def train(
    image_folder: Path, threshold_rel: float, model_type: str = "svm"
) -> Union[Pipeline, SVC, RandomForestClassifier, LogisticRegression]:
    """
    This function trains a machine learning model using segmented images from the specified folder
    and an active learning approach.

    Parameters:
    ----------
    image_folder : Path
        The folder containing TIFF images to be segmented and used for training.
    threshold_rel : float
        Minimum intensity of peaks of Laplacian-of-Gaussian (LoG).
        This should have a value between 0 and 1.
    model_type : str, optional
        The type of model to train. Options are 'svm', 'rf' (Random Forest), or 'logreg' (Logistic Regression).
        Default is 'svm'.

    Returns:
    -------
    sklearn Pipeline or RandomForestClassifier
        The best estimator found by the active learning.
    """
    # Validate model type
    if model_type not in {"svm", "rf", "logreg"}:
        raise ValueError(
            f"Invalid model type '{model_type}'. Choose from 'svm', 'rf', or 'logreg'."
        )

    # Create the base project folder with "_training" suffix
    project_folder = image_folder.parent / f"{image_folder.stem}_training"
    project_folder.mkdir(parents=True, exist_ok=True)

    # Check if there is an existing segmentation
    try:
        rois, layers = look_for_segmentation(
            project_folder, image_folder, threshold_rel
        )
    except Exception as e:
        raise RuntimeError(f"Error segmenting the images: {e}")

    # Extract features and labels from ROIs
    try:
        subset_df = create_subset_df(rois, layers)
    except Exception as e:
        raise RuntimeError(f"Error extracting features and labels: {e}")

    # Perform active learning
    try:
        model, performance_df = active_learning(subset_df, rois, layers, model_type)
    except Exception as e:
        raise RuntimeError(f"Error during active learning: {e}")

    # Save the trained model
    try:
        with open(
            project_folder / f"{image_folder.stem}_model_{model_type}.pkl", "wb"
        ) as file:
            pkl.dump(model, file)
    except Exception as e:
        raise RuntimeError(f"Error saving the model: {e}")

    # Save the metrics
    try:
        performance_df.to_csv(
            project_folder / "training_performance_metrics.csv", index=False
        )
    except Exception as e:
        raise RuntimeError(f"Error exporting training performance metrics: {e}")

    return model


def analyze(
    image_folder: Path, model: BaseEstimator, threshold_rel: float, cmap: str = "Reds"
) -> None:
    """
    This function processes TIFF images located in the `image_folder` by:
    1. Building a project directory.
    2. Segmenting the images to identify regions of interest (ROIs).
    3. Exporting the segmented ROIs to the project folder.
    4. Applying a prediction model (optional) to the segmented ROIs.

    Parameters:
    ----------
    image_folder : Path
        A `Path` object representing the folder containing TIFF image files to analyze.
    model : BaseEstimator
        A scikit-learn pipeline object used for predictions. This model should be previously obtained
        through functions like `cellrake.main.train`.
    threshold_rel : float
        Minimum intensity of peaks of Laplacian-of-Gaussian (LoG).
        This should have a value between 0 and 1.
    cmap : str, optional
        The color map to use for visualization when plotting results using matplotlib. Default is "Reds".
        It should be one of the available color maps in matplotlib, such as 'Reds', 'Greens', etc.

    Returns:
    -------
    None
    """

    # Ensure the provided color map is valid
    if cmap not in plt.colormaps():
        raise ValueError(
            f"Invalid colormap '{cmap}'. Available options are: {', '.join(plt.colormaps())}"
        )

    # Create a project folder for organizing results
    project_folder = build_project(image_folder)

    # Check if there is an existing segmentation
    try:
        rois, layers = look_for_segmentation(
            project_folder, image_folder, threshold_rel
        )
    except Exception as e:
        raise RuntimeError(f"Error segmenting the images: {e}")

    # Apply the prediction model to the layers and ROIs
    try:
        iterate_predicting(layers, rois, cmap, project_folder, model)
    except Exception as e:
        raise RuntimeError(f"Error during prediction iteration: {e}")
