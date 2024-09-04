"""
@author: Marc Canela
"""

import pickle as pkl
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from cellrake.predicting import iterate_predicting
from cellrake.segmentation import export_rois, iterate_segmentation
from cellrake.training import active_learning, create_subset_df
from cellrake.utils import build_project

# with open(
#     Path(
#         "//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/2_CellRake/model_train_data/"
#     )
#     / "tdt_model_rf.pkl",
#     "rb",
# ) as file:
#     model = pkl.load(file)


def train(
    image_folder: Path, threshold_rel: float, model_type: str = "svm"
) -> Union[Pipeline, RandomForestClassifier]:
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

    # Segment images to obtain ROIs and layers
    rois, layers = iterate_segmentation(image_folder, threshold_rel)

    # Extract features and labels from ROIs
    subset_df = create_subset_df(rois, layers)

    # Perform active learning
    model = active_learning(subset_df, rois, layers, model_type)

    # Save the trained model
    model_path = image_folder.parent / f"model_{model_type}.pkl"
    with open(model_path, "wb") as file:
        pkl.dump(model, file)

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
    model : BaseEstimator, optional
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

    # Segment images to obtain two dictionaries: 'rois' and 'layers'
    rois, layers = iterate_segmentation(image_folder, threshold_rel)

    # Export segmented ROIs to the project folder
    export_rois(project_folder, rois)

    # Apply the prediction model to the layers and ROIs
    iterate_predicting(layers, rois, cmap, project_folder, model)


# project_folder = Path('//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/2_CellRake/jose_bla_data/tdt_sample_analysis_trainotherdata')
