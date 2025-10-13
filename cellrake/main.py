# Created by: Marc Canela

import math
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
        # Segmentation hyperparameters
        max_sigma: int = 15,
        num_sigma: int = 10,
        overlap: float = 0,
        radius_expansion: float = 1.5 * math.sqrt(2),
        min_area: int = 60,
        max_area: int = 2000,
        hole_fill_ratio: float = 0.8,
        contour_level: float = 0.5,
        # Training hyperparameters
        clusters: int = 2,
        pca_variance_ratio: float = 0.95,
        entropy_threshold: float = 0.025,
        max_train_samples: int = 1000,
        max_test_samples: int = 1000,
        dataset_size_threshold: int = 2000,
        default_train_ratio: float = 0.8,
        smote_strategy: str = "minority",
        label_spreading_kernel: str = "knn",
        plot_entropy_threshold: float = 0.2,
        plot_dpi: int = 300,
        random_state: int = 42,
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

        Segmentation Parameters
        ----------------------
        max_sigma : int, default=15
            Maximum standard deviation for LoG filter.
        num_sigma : int, default=10
            Number of intermediate values for LoG filter.
        overlap : float, default=0
            Minimum distance between blobs as fraction of radius.
        radius_expansion : float, default=1.5*sqrt(2)
            Factor to expand blob radius for mask creation.
        min_area : int, default=60
            Minimum object area for cleaning.
        max_area : int, default=2000
            Maximum object area for cleaning.
        hole_fill_ratio : float, default=0.8
            Ratio for hole filling threshold.
        contour_level : float, default=0.5
            Level for contour extraction.

        Training Parameters
        ------------------
        clusters : int, default=2
            Number of clusters for grouping features.
        pca_variance_ratio : float, default=0.95
            Proportion of variance to retain in PCA dimensionality reduction.
        entropy_threshold : float, default=0.025
            Confidence threshold for pseudo-label selection based on entropy.
        max_train_samples : int, default=1000
            Maximum number of training samples when dataset is large.
        max_test_samples : int, default=1000
            Maximum number of testing samples when dataset is large.
        dataset_size_threshold : int, default=2000
            Dataset size above which to use fixed sample limits.
        default_train_ratio : float, default=0.8
            Training ratio for standard train/test split on smaller datasets.
        smote_strategy : str, default="minority"
            SMOTE sampling strategy for handling class imbalance.
        label_spreading_kernel : str, default="knn"
            Kernel type for label spreading algorithm.
        plot_entropy_threshold : float, default=0.2
            Entropy threshold for plot visualization confidence.
        plot_dpi : int, default=300
            DPI resolution for saved plots.
        random_state : int, default=42
            Random seed for reproducibility.
        """
        self.image_folder: Path = image_folder
        self.segmented_data: Dict = segmented_data
        self.project_dir: Path = project_dir
        self.model: BaseEstimator = None
        self.metrics: Dict = None
        self.counts: pd.DataFrame = None
        self.features: pd.DataFrame = None

        # Segmentation parameters
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.overlap = overlap
        self.radius_expansion = radius_expansion
        self.min_area = min_area
        self.max_area = max_area
        self.hole_fill_ratio = hole_fill_ratio
        self.contour_level = contour_level

        # Training parameters
        self.clusters = clusters
        self.pca_variance_ratio = pca_variance_ratio
        self.entropy_threshold = entropy_threshold
        self.max_train_samples = max_train_samples
        self.max_test_samples = max_test_samples
        self.dataset_size_threshold = dataset_size_threshold
        self.default_train_ratio = default_train_ratio
        self.smote_strategy = smote_strategy
        self.label_spreading_kernel = label_spreading_kernel
        self.plot_entropy_threshold = plot_entropy_threshold
        self.plot_dpi = plot_dpi
        self.random_state = random_state

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

            rois, layers = iterate_segmentation(
                self.image_folder,
                threshold_rel,
                self.max_sigma,
                self.num_sigma,
                self.overlap,
                self.radius_expansion,
                self.min_area,
                self.max_area,
                self.hole_fill_ratio,
                self.contour_level,
            )
            self.segmented_data = {"rois": rois, "layers": layers}
            return self.segmented_data

    def train(
        self,
        threshold_rel: float = 0.1,
        model_type: str = "rf",
        samples: int = 10,
        # Optional parameter overrides
        clusters: int = None,
        pca_variance_ratio: float = None,
        entropy_threshold: float = None,
        max_train_samples: int = None,
        max_test_samples: int = None,
        dataset_size_threshold: int = None,
        default_train_ratio: float = None,
        smote_strategy: str = None,
        label_spreading_kernel: str = None,
        plot_entropy_threshold: float = None,
        plot_dpi: int = None,
        random_state: int = None,
    ):
        """
        Train a model using semi-supervised learning on the segmented images.

        Parameters
        ----------
        threshold_rel : float, default=0.1
            Threshold for segmentation.
        model_type : str, default="rf"
            Type of model to train ('svm', 'rf', 'et', or 'logreg').
        samples : int, default=10
            Number of samples to draw from each cluster for initial labeling.

        All other parameters are optional overrides of the instance defaults.
        If provided, they will update the instance values permanently.
        If not provided, the values from __init__ will be used.
        """
        # Update instance hyperparameters with any provided overrides
        if clusters is not None:
            self.clusters = clusters
        if pca_variance_ratio is not None:
            self.pca_variance_ratio = pca_variance_ratio
        if entropy_threshold is not None:
            self.entropy_threshold = entropy_threshold
        if max_train_samples is not None:
            self.max_train_samples = max_train_samples
        if max_test_samples is not None:
            self.max_test_samples = max_test_samples
        if dataset_size_threshold is not None:
            self.dataset_size_threshold = dataset_size_threshold
        if default_train_ratio is not None:
            self.default_train_ratio = default_train_ratio
        if smote_strategy is not None:
            self.smote_strategy = smote_strategy
        if label_spreading_kernel is not None:
            self.label_spreading_kernel = label_spreading_kernel
        if plot_entropy_threshold is not None:
            self.plot_entropy_threshold = plot_entropy_threshold
        if plot_dpi is not None:
            self.plot_dpi = plot_dpi
        if random_state is not None:
            self.random_state = random_state

        seg = self.segment_images(threshold_rel)

        # Create subset with feature extraction and clustering
        subset_df = create_subset_df(
            seg["rois"],
            seg["layers"],
            clusters=self.clusters,
            pca_variance_ratio=self.pca_variance_ratio,
            random_state=self.random_state,
        )

        # Train classifier with current hyperparameters
        model, metrics_combined = train_classifier(
            subset_df,
            seg["rois"],
            seg["layers"],
            samples,
            model_type,
            self.project_dir,
            entropy_threshold=self.entropy_threshold,
            max_train_samples=self.max_train_samples,
            max_test_samples=self.max_test_samples,
            dataset_size_threshold=self.dataset_size_threshold,
            default_train_ratio=self.default_train_ratio,
            smote_strategy=self.smote_strategy,
            label_spreading_kernel=self.label_spreading_kernel,
            plot_entropy_threshold=self.plot_entropy_threshold,
            plot_dpi=self.plot_dpi,
            random_state=self.random_state,
        )

        self.model = model
        self.metrics = metrics_combined
        print("Model trained successfully.")

    def get_hyperparameters(self) -> Dict:
        """
        Get current hyperparameter values for training.

        Returns
        -------
        Dict
            Dictionary containing all current hyperparameter values.
        """
        return {
            # Feature extraction parameters
            "clusters": self.clusters,
            "pca_variance_ratio": self.pca_variance_ratio,
            "random_state": self.random_state,
            # Training parameters
            "entropy_threshold": self.entropy_threshold,
            "max_train_samples": self.max_train_samples,
            "max_test_samples": self.max_test_samples,
            "dataset_size_threshold": self.dataset_size_threshold,
            "default_train_ratio": self.default_train_ratio,
            # Semi-supervised learning parameters
            "smote_strategy": self.smote_strategy,
            "label_spreading_kernel": self.label_spreading_kernel,
            # Visualization parameters
            "plot_entropy_threshold": self.plot_entropy_threshold,
            "plot_dpi": self.plot_dpi,
        }

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
