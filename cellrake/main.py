# Created by: Marc Canela

import pickle as pkl
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sklearn.base import BaseEstimator

from cellrake.predicting import iterate_predicting
from cellrake.segmentation import iterate_segmentation
from cellrake.training import create_subset_df, train_classifier


class CellRake:
    """
    A cell analysis pipeline for image segmentation, training, and analysis.
    """

    def __init__(
        self,
        project_dir: Path,
        image_folder: Optional[Path] = None,
    ):
        """
        Initialize a CellRake object.

        Parameters
        ----------
        project_dir : Path
            Directory to save models and results.
        image_folder : Path, optional
            Folder containing TIFF images.
        """
        # Validate and setup directories
        project_dir = Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)

        if image_folder is not None:
            image_folder = Path(image_folder)
            if not image_folder.exists():
                raise FileNotFoundError(f"Image folder does not exist: {image_folder}")

        # Core instance variables
        self.image_folder: Optional[Path] = image_folder
        self.project_dir: Path = project_dir
        self.model: Optional[BaseEstimator] = None
        self.metrics: Optional[Dict] = None
        self.counts: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.segmented_data: Optional[Dict] = None

    def segment_images(
        self,
        threshold_rel: float,
        # Segmentation parameters - all optional, will use function defaults if not provided
        max_sigma: Optional[int] = None,
        num_sigma: Optional[int] = None,
        overlap: Optional[float] = None,
        radius_expansion: Optional[float] = None,
        min_area: Optional[int] = None,
        max_area: Optional[int] = None,
        hole_fill_ratio: Optional[float] = None,
        contour_level: Optional[float] = None,
    ) -> dict:
        """
        Run segmentation on images in image_folder.

        Parameters
        ----------
        threshold_rel : float
            Threshold for blob detection (0-1).
        max_sigma : int, optional
            Maximum standard deviation for LoG filter.
        num_sigma : int, optional
            Number of intermediate values for LoG filter.
        overlap : float, optional
            Minimum distance between blobs as fraction of radius.
        radius_expansion : float, optional
            Factor to expand blob radius for mask creation.
        min_area : int, optional
            Minimum object area for cleaning.
        max_area : int, optional
            Maximum object area for cleaning.
        hole_fill_ratio : float, optional
            Ratio for hole filling threshold.
        contour_level : float, optional
            Level for contour extraction.

        Returns
        -------
        dict
            Dictionary containing 'rois' and 'layers' data.
        """
        if self.segmented_data is not None:
            print("Using pre-segmented data.")
            return self.segmented_data

        if self.image_folder is None:
            raise ValueError("image_folder must be defined before segmentation.")

        # Build segmentation arguments - only include non-None parameters
        seg_args = {
            "image_folder": self.image_folder,
            "threshold_rel": threshold_rel,
        }

        # Add optional parameters if provided
        if max_sigma is not None:
            seg_args["max_sigma"] = max_sigma
        if num_sigma is not None:
            seg_args["num_sigma"] = num_sigma
        if overlap is not None:
            seg_args["overlap"] = overlap
        if radius_expansion is not None:
            seg_args["radius_expansion"] = radius_expansion
        if min_area is not None:
            seg_args["min_area"] = min_area
        if max_area is not None:
            seg_args["max_area"] = max_area
        if hole_fill_ratio is not None:
            seg_args["hole_fill_ratio"] = hole_fill_ratio
        if contour_level is not None:
            seg_args["contour_level"] = contour_level

        rois, layers = iterate_segmentation(**seg_args)
        self.segmented_data = {"rois": rois, "layers": layers}
        return self.segmented_data

    def train(
        self,
        threshold_rel: float = 0.1,
        model_type: str = "rf",
        samples: int = 25, # 2 clusters x "samples" to label in each
        # Segmentation parameters (if you want to use different segmentation for training)
        max_sigma: Optional[int] = None,
        num_sigma: Optional[int] = None,
        overlap: Optional[float] = None,
        radius_expansion: Optional[float] = None,
        min_area: Optional[int] = None,
        max_area: Optional[int] = None,
        hole_fill_ratio: Optional[float] = None,
        contour_level: Optional[float] = None,
        # Training parameters
        pca_variance_ratio: Optional[float] = None,
        entropy_threshold: Optional[float] = None,
        max_test_samples: Optional[int] = None,
        default_train_ratio: Optional[float] = None,
        label_spreading_kernel: Optional[str] = None,
        plot_entropy_threshold: Optional[float] = None,
        plot_dpi: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Train a classification model using semi-supervised learning.

        Parameters
        ----------
        threshold_rel : float, default=0.1
            Threshold for segmentation.
        model_type : str, default="rf"
            Type of model to train ('svm', 'rf', 'et', or 'logreg').
        samples : int, default=10
            Number of samples to draw from each cluster for initial labeling.
        max_sigma : int, optional
            Maximum standard deviation for LoG filter.
        num_sigma : int, optional
            Number of intermediate values for LoG filter.
        overlap : float, optional
            Minimum distance between blobs as fraction of radius.
        radius_expansion : float, optional
            Factor to expand blob radius for mask creation.
        min_area : int, optional
            Minimum object area for cleaning.
        max_area : int, optional
            Maximum object area for cleaning.
        hole_fill_ratio : float, optional
            Ratio for hole filling threshold.
        contour_level : float, optional
            Level for contour extraction.
        pca_variance_ratio : float, optional
            Proportion of variance to retain in PCA.
        entropy_threshold : float, optional
            Confidence threshold for pseudo-label selection.
        max_test_samples : int, optional
            Maximum number of testing samples.
        default_train_ratio : float, optional
            Training ratio for train/test split.
        label_spreading_kernel : str, optional
            Kernel type for label spreading algorithm.
        plot_entropy_threshold : float, optional
            Entropy threshold for plot visualization.
        plot_dpi : int, optional
            DPI resolution for saved plots.
        random_state : int, optional
            Random seed for reproducibility.
        """
        # Get segmentation with optional parameters
        seg = self.segment_images(
            threshold_rel,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            overlap=overlap,
            radius_expansion=radius_expansion,
            min_area=min_area,
            max_area=max_area,
            hole_fill_ratio=hole_fill_ratio,
            contour_level=contour_level,
        )

        # Create subset with feature extraction and clustering
        subset_args = {
            "rois": seg["rois"],
            "layers": seg["layers"],
        }
        if pca_variance_ratio is not None:
            subset_args["pca_variance_ratio"] = pca_variance_ratio
        if random_state is not None:
            subset_args["random_state"] = random_state

        df_train, df_val, df_test = create_subset_df(**subset_args)

        # Train classifier with hyperparameters
        train_args = {
            "df_train": df_train,
            "df_val": df_val,
            "df_test": df_test,
            "rois": seg["rois"],
            "layers": seg["layers"],
            "samples": samples,
            "model_type": model_type,
            "project_folder": self.project_dir,
        }

        # Add training parameters if provided
        if entropy_threshold is not None:
            train_args["entropy_threshold"] = entropy_threshold
        if max_test_samples is not None:
            train_args["max_test_samples"] = max_test_samples
        if default_train_ratio is not None:
            train_args["default_train_ratio"] = default_train_ratio
        if label_spreading_kernel is not None:
            train_args["label_spreading_kernel"] = label_spreading_kernel
        if plot_entropy_threshold is not None:
            train_args["plot_entropy_threshold"] = plot_entropy_threshold
        if plot_dpi is not None:
            train_args["plot_dpi"] = plot_dpi
        if random_state is not None:
            train_args["random_state"] = random_state

        model, metrics_combined = train_classifier(**train_args)

        self.model = model
        self.metrics = metrics_combined
        print("Model trained successfully.")

    def save_model(self, filename: str) -> Path:
        """
        Save the trained model to disk.

        Parameters
        ----------
        filename : str
            Name for the saved model file (without .pkl extension).

        Returns
        -------
        Path
            Path to the saved model file.
        """
        if self.model is None:
            raise ValueError(
                "No trained model to save. Train a model first using .train()."
            )

        filepath = self.project_dir / f"{filename}.pkl"
        with open(filepath, "wb") as file:
            pkl.dump(self.model, file)
        print(f"Model saved as {filepath}")

    def load_model(self, filename: str) -> BaseEstimator:
        """
        Load a trained model from disk.

        Parameters
        ----------
        filename : str
            Name of the model file to load (without .pkl extension).

        Returns
        -------
        BaseEstimator
            The loaded model.
        """
        filepath = self.project_dir / f"{filename}.pkl"
        if not filepath.exists():
            raise FileNotFoundError(f"Model file {filepath} not found.")

        with open(filepath, "rb") as file:
            self.model = pkl.load(file)
        print(f"Model loaded successfully from {filepath}")

    def save_segmentation(self, filename: str) -> Path:
        """
        Save the segmented data to disk.

        Parameters
        ----------
        filename : str
            Name for the saved segmentation file (without .pkl extension).

        Returns
        -------
        Path
            Path to the saved segmentation file.
        """
        if self.segmented_data is None:
            raise ValueError("No segmented data to save. Run segmentation first.")

        filepath = self.project_dir / f"{filename}.pkl"
        with open(filepath, "wb") as file:
            pkl.dump(self.segmented_data, file)
        print(f"Segmentation saved as {filepath}")

    def load_segmentation(self, filename: str) -> dict:
        """
        Load segmented data from disk.

        Parameters
        ----------
        filename : str
            Name of the segmentation file to load (without .pkl extension).

        Returns
        -------
        dict
            The loaded segmentation data.
        """
        filepath = self.project_dir / f"{filename}.pkl"
        if not filepath.exists():
            raise FileNotFoundError(f"Segmentation file {filepath} not found.")

        with open(filepath, "rb") as file:
            self.segmented_data = pkl.load(file)
        print(f"Segmentation loaded successfully from {filepath}")

    def analyze(
        self,
        threshold_rel: float = 0.1,
        cmap: str = "Reds",
        # Segmentation parameters (optional, for analysis-specific segmentation)
        max_sigma: Optional[int] = None,
        num_sigma: Optional[int] = None,
        overlap: Optional[float] = None,
        radius_expansion: Optional[float] = None,
        min_area: Optional[int] = None,
        max_area: Optional[int] = None,
        hole_fill_ratio: Optional[float] = None,
        contour_level: Optional[float] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply the trained model to analyze images.

        Parameters
        ----------
        threshold_rel : float, default=0.5
            Threshold for segmentation.
        cmap : str, default="Reds"
            Colormap for visualization plots.
        max_sigma : int, optional
            Maximum standard deviation for LoG filter.
        num_sigma : int, optional
            Number of intermediate values for LoG filter.
        overlap : float, optional
            Minimum distance between blobs as fraction of radius.
        radius_expansion : float, optional
            Factor to expand blob radius for mask creation.
        min_area : int, optional
            Minimum object area for cleaning.
        max_area : int, optional
            Maximum object area for cleaning.
        hole_fill_ratio : float, optional
            Ratio for hole filling threshold.
        contour_level : float, optional
            Level for contour extraction.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Tuple containing (counts, features) DataFrames.
        """
        if self.model is None:
            raise ValueError(
                "No trained model available. Either:\n"
                "- Train a new model using the train() method, or\n"
                "- Load a previously trained model using load_model()"
            )

        # Get segmentation with optional parameters
        seg = self.segment_images(
            threshold_rel,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            overlap=overlap,
            radius_expansion=radius_expansion,
            min_area=min_area,
            max_area=max_area,
            hole_fill_ratio=hole_fill_ratio,
            contour_level=contour_level,
        )
        counts, features = iterate_predicting(
            seg["layers"], seg["rois"], cmap, self.model, self.project_dir
        )
        self.counts = counts
        self.features = features

        # Save results to CSV files
        counts_path = self.project_dir / "cell_counts.csv"
        features_path = self.project_dir / "cell_features.csv"

        counts.to_csv(counts_path, index=False)
        features.to_csv(features_path, index=False)

        print(f"\nAnalysis complete!")
        print(f"- Cell counts saved to: {counts_path}")
        print(f"- Cell features saved to: {features_path}")

        return counts, features

    def set_image_folder(self, image_folder: Path) -> None:
        """
        Set or change the image folder path.

        Parameters
        ----------
        image_folder : Path
            Path to folder containing TIFF images.
        """
        image_folder = Path(image_folder)
        if not image_folder.exists():
            raise FileNotFoundError(f"Image folder does not exist: {image_folder}")
        self.image_folder = image_folder

        # Clear segmented data since we're using a different image folder
        self.segmented_data = None

    def set_project_dir(self, project_dir: Path) -> None:
        """
        Set or change the project directory path.

        Parameters
        ----------
        project_dir : Path
            Directory to save models and results.
        """
        project_dir = Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        self.project_dir = project_dir

    def __repr__(self) -> str:
        """
        String representation of the CellRake object.

        Returns
        -------
        str
            String describing the current state of the CellRake object.
        """
        status_items = []

        # project_dir is always available now
        project_name = self.project_dir.name
        status_items.append(f"project='{project_name}'")

        if self.model:
            status_items.append("model = trained")
        else:
            status_items.append("model = None")

        if self.segmented_data:
            num_images = len(self.segmented_data.get("layers", {}))
            status_items.append(f"segmented = {num_images}")
        else:
            status_items.append("segmented = 0")

        if self.counts is not None:
            status_items.append(f"analyzed = {len(self.counts)}")
        else:
            status_items.append("analyzed = 0")

        status = ", ".join(status_items) if status_items else "empty"
        return f"CellRake({status})"
