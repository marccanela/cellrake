# Created by: Marc Canela

from pathlib import Path
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from tqdm import tqdm

from cellrake.utils import (
    create_stats_dict,
    crop_cell_large,
    evaluate_model,
    train_model,
)

# ================================================================================
# USER INTERACTION UTILITIES
# ================================================================================


def user_input(
    roi_values: Dict[str, np.ndarray], layer: np.ndarray, padding: int = 120
) -> str:
    """
    Display ROI overlaid on image and prompt user for cell/non-cell classification.

    Parameters
    ----------
    roi_values : Dict[str, np.ndarray]
        Dictionary with 'x' and 'y' keys containing ROI coordinates.
    layer : np.ndarray
        2D image array for ROI visualization.
    padding : int, default=120
        Padding pixels for cropped ROI visualization.

    Returns
    -------
    str
        User classification: '1' for cell, '0' for non-cell.
    """
    x_coords, y_coords = roi_values["x"], roi_values["y"]

    # Set up the plot with four subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    # Full image with ROI highlighted
    axes[0].imshow(layer, cmap="viridis")
    axes[0].plot(x_coords, y_coords, "m-", linewidth=1)
    axes[0].axis("off")  # Hide the axis

    # Full image without ROI highlighted
    axes[1].imshow(layer, cmap="viridis")
    axes[1].axis("off")  # Hide the axis

    # Cropped image with padding, ROI highlighted
    layer_cropped_small, x_coords_cropped, y_coords_cropped = crop_cell_large(
        layer, x_coords, y_coords, padding=padding
    )
    axes[2].imshow(layer_cropped_small, cmap="viridis")
    axes[2].plot(x_coords_cropped, y_coords_cropped, "m-", linewidth=1)
    axes[2].axis("off")  # Hide the axis

    # Cropped image without ROI highlighted
    axes[3].imshow(layer_cropped_small, cmap="viridis")
    axes[3].axis("off")  # Hide the axis

    plt.tight_layout()
    plt.show()
    plt.pause(0.1)

    # Ask for user input
    user_input_value = input("Please enter 1 (cell) or 0 (non-cell): ")
    while user_input_value not in ["1", "0"]:
        user_input_value = input("Invalid input. Please enter 1 or 0: ")

    plt.close(fig)

    return int(user_input_value)


def manual_labeling(
    features_df: pd.DataFrame, rois: Dict[str, dict], layers: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Prompt user to manually label ROIs for training data.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with features where each row is a sample.
    rois : Dict[str, dict]
        Dictionary mapping image tags to ROI dictionaries with coordinates.
    layers : Dict[str, np.ndarray]
        Dictionary mapping image tags to 2D image arrays.

    Returns
    -------
    pd.DataFrame
        DataFrame with manual labels in "label_column".
    """
    if features_df.empty:
        raise ValueError("The features DataFrame is empty. Please provide a valid one.")

    index_list = features_df.index.tolist()

    labels_dict = {}
    n = 1
    for index in index_list:
        print(f"Image {n} out of {len(index_list)}.")
        tag = index.split("_roi")[0]
        roi = f"roi{index.split('_roi')[1]}"
        layer = layers[tag]
        roi_values = rois[tag][roi]
        labels_dict[index] = user_input(roi_values, layer)
        n += 1

    labels_df = pd.DataFrame.from_dict(
        labels_dict, orient="index", columns=["label_column"]
    )

    return labels_df


# ================================================================================
# SEMI-SUPERVISED LEARNING ALGORITHMS
# ================================================================================


def label_speading(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    rois: Dict[str, dict],
    layers: Dict[str, np.ndarray],
    samples: int,
    label_spreading_kernel: str,
    random_state: int,
) -> pd.DataFrame:
    """
    Apply semi-supervised learning using label spreading for pseudo-labeling.

    Parameters
    ----------
    df_train : pd.DataFrame
        DataFrame with ROI features and cluster assignments for training.
    df_val : pd.DataFrame
        DataFrame with ROI features and cluster assignments for validation.
    df_test : pd.DataFrame
        DataFrame with ROI features and cluster assignments for testing.
    rois : Dict[str, dict]
        Dictionary mapping image tags to ROI dictionaries with coordinates.
    layers : Dict[str, np.ndarray]
        Dictionary mapping image tags to 2D image arrays.
    samples : int
        Number of samples to manually label from each cluster.
    label_spreading_kernel : str
        Kernel type for label spreading algorithm.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        DataFrames with ROI features, cluster assignments, and pseudo-labels for training, validation, and testing.
    """
    # Add dummy labels to the dfs
    df_train['labels'] = -1
    df_val['labels'] = -1
    df_test['labels'] = -1
    
    # Explore and label the clusters
    exploratory_dfs = []
    for cluster in df_train["cluster"].unique():
        cluster_df = df_train[df_train["cluster"] == cluster]
        if len(cluster_df) >= samples:
            sampled_df = cluster_df.sample(n=samples, random_state=random_state)
        else:
            sampled_df = cluster_df
        exploratory_dfs.append(sampled_df)

    # Combine exploratory samples and manual labeling
    exploratory_df = pd.concat(exploratory_dfs)
    exploratory_df_labeled = manual_labeling(exploratory_df, rois, layers)
    
    # Add manual labels to df_train
    labeled_series = exploratory_df_labeled['label_column'].astype(int)
    df_train.loc[labeled_series.index, 'labels'] = labeled_series

    # Standarize features
    scaler = StandardScaler() # Always returns a NumPy array
    features_to_scale = [col for col in df_train.columns if col not in ['cluster', 'labels']]
    
    X_train_scaled = scaler.fit_transform(df_train[features_to_scale]) # (Fit on df_train)
    X_val_scaled = scaler.transform(df_val[features_to_scale])
    X_test_scaled = scaler.transform(df_test[features_to_scale])
    
    # Apply SMOTE to balance the labeled data
    labeled_mask = df_train['labels'] != -1
    X_labeled = X_train_scaled[labeled_mask]
    y_labeled = df_train.loc[labeled_mask, 'labels'].astype(int)
    smote = SMOTE(sampling_strategy="minority", random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_labeled, y_labeled)
    
    # Create new training set with resampled labeled data and unlabeled pool
    X_pool = X_train_scaled[~labeled_mask]
    X_train_scaled_smoted = np.vstack([X_resampled, X_pool])
    y_train_smoted = np.hstack([y_resampled, [-1]*len(X_pool)])

    # Fit LabelSpreading (Fit on df_train)
    ls = LabelSpreading(kernel=label_spreading_kernel)
    ls.fit(X_train_scaled_smoted, y_train_smoted)

    # Generate pseudo-labels for all three sets
    data_sets = {
        'train': {
            'df': df_train,
            'X_scaled': X_train_scaled,
        },
        'val': {
            'df': df_val,
            'X_scaled': X_val_scaled,
        },
        'test': {
            'df': df_test,
            'X_scaled': X_test_scaled,
        },
    }
    
    subset_dfs = []
    for subset in data_sets.values():
        df = subset['df']
        X_scaled = subset['X_scaled']
        
        # Predict labels using the trained LabelSpreading model
        predicted_labels = ls.predict(X_scaled)
        
        # Compute probabilities of predictions
        subset_probabilities = ls.predict_proba(X_scaled)
        label_entropy = entropy(subset_probabilities, base=2, axis=1)
        # Entropy = 1.0 is maximum uncertainty (random guess)
        # Entropy = 0.0 is minimum uncertainty (full confidence)
        is_zero_prob = np.all(subset_probabilities == [0, 0], axis=1)
        
        # Create new df
        subset_df = df.copy()
        subset_df["labels"] = predicted_labels
        subset_df["label_entropy"] = label_entropy
        subset_df["is_zero_prob"] = is_zero_prob
        
        subset_dfs.append(subset_df)

    return subset_dfs[0], subset_dfs[1], subset_dfs[2]


# ================================================================================
# VISUALIZATION UTILITIES
# ================================================================================


def plot_pca(
    tag: str,
    total_df: pd.DataFrame,
    project_folder: Union[str, Path],
    entropy_threshold: float,
    plot_dpi: int,
    random_state: int,
) -> None:
    """
    Create 2D PCA visualization of labeled and pseudo-labeled data.

    Parameters
    ----------
    tag : str
        Tag for the dataset (e.g., "Train", "Validation", "Test").
    total_df : pd.DataFrame
        DataFrame with features, labels, and confidence metrics.
    project_folder : Union[str, Path]
        Path to project folder for saving the plot.
    entropy_threshold : float
        Entropy threshold for determining confidence in pseudo-labels.
    plot_dpi : int
        DPI resolution for saved plot.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    None
    """
    # Create colormaps
    pastel1_colors = ["#fbb4ae", "#b3cde3"]  # First two colors of Pastel1
    set1_colors = ["#e41a1c", "#377eb8"]  # First two colors of Set1

    # Reduce the dimensions of the data to 2D using PCA for visualization
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=random_state)),
        ]
    )
    X = total_df.drop(
        columns=["cluster", "labels", "is_zero_prob", "label_entropy"]
    ).values
    X = pipeline.fit_transform(X)

    # Rotated loading scores
    loading_scores = pipeline.named_steps["pca"].components_
    max_contributing_features = np.argmax(np.abs(loading_scores), axis=1)
    feature_names = total_df.drop(
        columns=["cluster", "labels", "is_zero_prob", "label_entropy"]
    ).columns
    most_contributing_features = [feature_names[i] for i in max_contributing_features]

    # Build masks
    label_0 = total_df.labels == 0
    label_1 = total_df.labels == 1
    correct_prob = total_df.is_zero_prob == False
    confident = total_df.label_entropy < entropy_threshold

    # Plot the data points
    plt.figure()

    # Negative pseudo-labeled
    mask_negative_pseudo = label_0 & correct_prob & confident
    X_negative_pseudo = X[mask_negative_pseudo]
    plt.scatter(
        X_negative_pseudo[:, 0],
        X_negative_pseudo[:, 1],
        color=pastel1_colors[1],
        marker="x",
        label="Negative Pseudo-labeled",
    )

    # Positive pseudo-labeled
    mask_positive_pseudo = label_1 & correct_prob & confident
    X_positive_pseudo = X[mask_positive_pseudo]
    plt.scatter(
        X_positive_pseudo[:, 0],
        X_positive_pseudo[:, 1],
        color=pastel1_colors[0],
        marker="x",
        label="Positive Pseudo-labeled",
    )

    # Add legend and title
    plt.legend()
    plt.title(f"2D PCA of Pseudo-labeled Data in the {tag} Set")
    plt.xlabel(f"PC1 ({most_contributing_features[0]})")
    plt.ylabel(f"PC2 ({most_contributing_features[1]})")

    # Save the plot as a PNG file
    output_path = f"{project_folder}/pca_pseudolabels_{tag}.png"
    plt.savefig(output_path, dpi=plot_dpi, bbox_inches="tight")
    plt.close()


# ================================================================================
# WORKFLOW ORCHESTRATION
# ================================================================================


def handle_pseudo_labels(
    project_folder: Union[str, Path],
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    rois: Dict[str, dict],
    layers: Dict[str, np.ndarray],
    samples: int,
    label_spreading_kernel: str,
    plot_entropy_threshold: float,
    plot_dpi: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Generate pseudo-labels using label spreading and create visualization.

    Parameters
    ----------
    project_folder : Union[str, Path]
        Path to project folder for saving plots.
    df_train : pd.DataFrame
        DataFrame with ROI features and cluster assignments for training.
    df_val : pd.DataFrame
        DataFrame with ROI features and cluster assignments for validation.
    df_test : pd.DataFrame
        DataFrame with ROI features and cluster assignments for testing.
    rois : Dict[str, dict]
        Dictionary mapping image tags to ROI dictionaries with coordinates.
    layers : Dict[str, np.ndarray]
        Dictionary mapping image tags to 2D image arrays.
    samples : int
        Number of samples to manually label from each cluster.
    label_spreading_kernel : str
        Kernel type for label spreading algorithm.
    plot_entropy_threshold : float
        Entropy threshold for plot visualization confidence.
    plot_dpi : int
        DPI resolution for saved plots.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        DataFrames with ROI features, cluster assignments, and pseudo-labels for training, validation, and testing.
    """
    df_train_labelled, df_val_labelled, df_test_labelled = label_speading(
        df_train,
        df_val,
        df_test,
        rois,
        layers,
        samples,
        label_spreading_kernel,
        random_state,
    )
    print("Pseudo-labeled dataset generated.")
    
    plot_pca("Train", df_train_labelled, project_folder, plot_entropy_threshold, plot_dpi, random_state)
    plot_pca("Validation", df_val_labelled, project_folder, plot_entropy_threshold, plot_dpi, random_state)
    plot_pca("Test", df_test_labelled, project_folder, plot_entropy_threshold, plot_dpi, random_state)
    
    return df_train_labelled, df_val_labelled, df_test_labelled


# ================================================================================
# MAIN EXPORTED FUNCTIONS
# ================================================================================


def create_subset_df(
    rois: Dict[str, dict],
    layers: Dict[str, np.ndarray],
    pca_variance_ratio: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Extract features from ROIs and cluster them into groups for training.

    Parameters
    ----------
    rois : Dict[str, dict]
        Dictionary mapping image tags to ROI dictionaries with coordinates.
    layers : Dict[str, np.ndarray]
        Dictionary mapping image tags to 2D image arrays.
    pca_variance_ratio : float, default=0.95
        Proportion of variance to retain in PCA dimensionality reduction.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        DataFrames with ROI features and cluster assignments for training, validation, and testing.
    """

    # Extract statistical features from each ROI
    roi_props_dict = {}
    for tag in tqdm(rois.keys(), desc="Extracting input features", unit="image"):
        roi_dict = rois[tag]
        layer = layers[tag]

        # Check if roi_dict is empty
        if not roi_dict:
            continue

        roi_props_dict[tag] = create_stats_dict(roi_dict, layer)

    # Flatten the dictionary structure for input features
    input_features = {}
    for tag, all_rois in roi_props_dict.items():
        for roi_num, stats in all_rois.items():
            input_features[f"{tag}_{roi_num}"] = stats
    features_df = pd.DataFrame.from_dict(input_features, orient="index")

    if features_df.empty:
        raise ValueError("The features DataFrame is empty. Please provide a valid one.")
    
    # Ensure specific columns have the correct data types
    features_df["min_intensity"] = features_df["min_intensity"].astype(int)
    features_df["max_intensity"] = features_df["max_intensity"].astype(int)
    features_df["hog_mean"] = features_df["hog_mean"].astype(float)
    features_df["hog_std"] = features_df["hog_std"].astype(float)
    
    # Split into train-validation-test sets
    n_rows = features_df.shape[0]
    if n_rows < 1e6: # 70-15-15
        test_size = 0.15
        val_size = 0.15 / (1 - test_size)
    else: # 90-5-5
        test_size = 0.05
        val_size = 0.05 / (1 - test_size)
    
    df_train_temp, df_test = train_test_split(features_df, test_size=test_size, random_state=random_state)

    # Cluster the features to aproximate the positive/negative classes (in df_train_temp)
    kmeans_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_variance_ratio, random_state=random_state)),
            ("kmeans", KMeans(n_clusters=2, random_state=random_state)),
        ]
    )
    best_clusters = kmeans_pipeline.fit_predict(df_train_temp)
    df_train_temp["cluster"] = best_clusters
    
    # Split df_train_temp into train and validation sets
    df_train, df_val = train_test_split(df_train_temp, test_size=val_size, random_state=random_state, stratify=df_train_temp["cluster"])
    
    # Expand clustering to the test set
    test_clusters = kmeans_pipeline.predict(df_test)
    df_test["cluster"] = test_clusters

    return df_train, df_val, df_test


def train_classifier(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    rois: Dict[str, dict],
    layers: Dict[str, np.ndarray],
    samples: int,
    model_type: str,
    project_folder: Union[str, Path],
    entropy_threshold: float = 0.036,
    max_test_samples: int = 1000,
    default_train_ratio: float = 0.8,
    label_spreading_kernel: str = "knn",
    plot_entropy_threshold: float = 0.2,
    plot_dpi: int = 300,
    random_state: int = 42,
) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Train a classifier using semi-supervised learning with label spreading.

    Parameters
    ----------
    df_train : pd.DataFrame
        DataFrame with ROI features and cluster assignments for training.
    df_val : pd.DataFrame
        DataFrame with ROI features and cluster assignments for validation.
    df_test : pd.DataFrame
        DataFrame with ROI features and cluster assignments for testing.
    rois : Dict[str, dict]
        Dictionary mapping image tags to ROI dictionaries with coordinates.
    layers : Dict[str, np.ndarray]
        Dictionary mapping image tags to 2D image arrays.
    samples : int
        Number of samples to draw from each cluster for initial labeling.
    model_type : str
        Type of model to train ('svm', 'rf', 'et', or 'logreg').
    project_folder : Union[str, Path]
        Path to project folder for saving results and models.
    entropy_threshold : float, default=0.036
        Confidence threshold for pseudo-label selection based on entropy.
    max_test_samples : int, default=1000
        Maximum number of testing samples when dataset is large.
    default_train_ratio : float, default=0.8
        Training ratio for standard train/test split on smaller datasets.
    label_spreading_kernel : str, default="knn"
        Kernel type for label spreading algorithm.
    plot_entropy_threshold : float, default=0.2
        Entropy threshold for plot visualization confidence.
    plot_dpi : int, default=300
        DPI resolution for saved plots.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[Pipeline, pd.DataFrame]
        Trained classifier pipeline and performance metrics DataFrame.
    """
    # Label spreading
    df_train_labelled, df_val_labelled, df_test_labelled = handle_pseudo_labels(
        project_folder,
        df_train,
        df_val,
        df_test,
        rois,
        layers,
        samples,
        label_spreading_kernel,
        plot_entropy_threshold,
        plot_dpi,
        random_state,
    )
    
    # Create dictionary of dataframes
    dataframes = {
        "train": df_train_labelled,
        "val": df_val_labelled,
        "test": df_test_labelled,
    }

    print("Proceeding with training the model...")
    
    processed_dfs = {}
    
    for tag, total_df in dataframes.items():
        
        # Build mask for confident pseudo-labels
        correct_prob = total_df.is_zero_prob == False
        confident = total_df.label_entropy < entropy_threshold
        mask = correct_prob & confident

        raw_features = total_df.drop(
            columns=["cluster", "labels", "is_zero_prob", "label_entropy"] 
        ).values
        raw_labels = total_df.labels.values.astype(int)

        X = raw_features[mask]
        y = raw_labels[mask]
        
        processed_dfs[tag] = {
            "X": X,
            "y": y,
        }

    # Train classifier
    best_model = train_model(processed_dfs, model_type, random_state)

    # Evaluations
    print(f"Proceeding with evaluating the {model_type} classifier...")
    metrics_train = evaluate_model(best_model, processed_dfs['train'])
    metrics_test = evaluate_model(best_model, processed_dfs['test'])
    metrics_combined = pd.DataFrame({"Train": metrics_train, "Test": metrics_test})
    print(metrics_combined)

    return best_model, metrics_combined
