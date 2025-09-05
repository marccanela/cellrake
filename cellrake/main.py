"""
@author: Marc Canela
"""

import pickle as pkl
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.base import BaseEstimator

from cellrake.predicting import iterate_predicting
from cellrake.segmentation import iterate_segmentation
from cellrake.training import create_subset_df, train_classifier


class CellRake:

    def __init__(
        self,
        image_folder: Path,
        project_dir: Path,
        segmented_data: dict = None,
    ):
        """
        Initialize a CellRake object.

        Parameters
        ----------
        image_folder : Path
            Folder containing TIFF images.
        project_dir : Path
            Directory to save models and results.
        segmented_data : dict, optional
            Already segmented ROIs and layers (from a saved segmentation).
        """
        self.image_folder: Path = image_folder
        self.segmented_data: Dict = segmented_data
        self.project_dir: Path = project_dir
        self.model: BaseEstimator = None
        self.metrics: Dict = None
        self.counts: pd.DataFrame = None
        self.features: pd.DataFrame = None

    def segment_images(self, threshold_rel: float):
        """
        Run segmentation on the images in `image_folder`.
        If `segmented_data` exists, reuse it.
        """
        if self.segmented_data is not None:
            print("Using pre-segmented data.")
            return self.segmented_data

        else:
            if self.image_folder is None:
                raise ValueError("Please, define image_folder before segment_images.")

            rois, layers = iterate_segmentation(self.image_folder, threshold_rel)
            self.segmented_data = {"rois": rois, "layers": layers}
            return self.segmented_data

    def train(
        self, threshold_rel: float = 0.1, model_type: str = "rf", samples: int = 10
    ):
        """
        Train a model using active learning on the segmented images.
        """
        seg = self.segment_images(threshold_rel)
        subset_df = create_subset_df(seg["rois"], seg["layers"], samples)

        model, metrics_combined = train_classifier(
            subset_df, seg["rois"], seg["layers"], samples, model_type, self.project_dir
        )
        self.model = model
        self.metrics = metrics_combined
        print("Model trained successfully.")

    def save_model(self, filename: str):
        """Save the trained model with a custom name."""
        if self.model is None:
            raise ValueError("No trained model to save.")
        if self.project_dir is None:
            raise ValueError("Please, define project_dir before save_model.")

        with open(self.project_dir / f"{filename}.pkl", "wb") as file:
            pkl.dump(self.model, file)
        print(f"Model saved as {filename}.pkl")

    def load_model(self, filename: str):
        """Load a trained model from disk."""
        if self.project_dir is None:
            raise ValueError("Please, define project_dir before load_model.")
        filepath = self.project_dir / f"{filename}.pkl"
        if not filepath.exists():
            raise FileNotFoundError(f"Model file {filepath} not found.")

        with open(filepath, "rb") as file:
            self.model = pkl.load(file)
        print(f"Model {filename} loaded successfully. Use .model to access it.")

    def save_segmentation(self, filename: str):
        """Save the segmented data with a custom name."""
        if self.segmented_data is None:
            raise ValueError("No segmented data to save.")
        if self.project_dir is None:
            raise ValueError("Please, define project_dir before save_segmentation.")

        with open(self.project_dir / f"{filename}.pkl", "wb") as file:
            pkl.dump(self.segmented_data, file)
        print(f"Segmentation saved as {filename}.pkl")

    def load_segmentation(self, filename: str):
        """Load segmented data from disk."""
        if self.project_dir is None:
            raise ValueError("Please, define project_dir before load_segmentation.")
        filepath = self.project_dir / f"{filename}.pkl"
        if not filepath.exists():
            raise FileNotFoundError(f"Segmentation file {filepath} not found.")

        with open(filepath, "rb") as file:
            self.segmented_data = pkl.load(file)
        print(
            f"Segmentation {filename} loaded successfully. Use .segmented_data to access it."
        )

    def analyze(self, threshold_rel: float = 0.5, cmap: str = "Reds"):
        """
        Apply the trained model to new images or a segmented object.
        """
        if self.model is None:
            raise ValueError(
                """
                No trained model available:
                - Load a previously trained model using the load_model method.
                - Train a new model using the train method.
                """
            )

        if self.project_dir is None:
            raise ValueError("Please, define project_dir before analyze.")

        seg = self.segment_images(threshold_rel)
        counts, features = iterate_predicting(
            seg["layers"], seg["rois"], cmap, self.model, self.project_dir
        )
        self.counts = counts
        self.features = features

        # Save counts and features to CSV files
        counts.to_csv(self.project_dir / "cell_counts.csv", index=False)
        features.to_csv(self.project_dir / "cell_features.csv", index=False)
        print("\nAnalysis complete. Results saved to project directory.")
