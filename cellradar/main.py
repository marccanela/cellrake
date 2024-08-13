"""
Created on Sat Aug 10 12:38:20 2024
@author: mcanela
"""

import pickle as pkl
from pathlib import Path

import pandas as pd
from predicting import iterate_predicting
from segmentation import export_rois, iterate_segmentation
from training import (
    label_rois,
    random_train_test_split,
    train_logreg,
    train_rf,
    train_svm,
)
from utils import build_project

image_folder = Path("C:/Users/mcanela/Desktop/sample_marc/tdt")
with open(
    "C:/Users/mcanela/Desktop/sample_jose/best_model/best_model_svm.pkl", "rb"
) as file:
    best_model = pkl.load(file)
cmap = "Reds"


def analyze(image_folder, cmap="Reds", best_model=None):

    project_folder = build_project(image_folder)

    # Segment images
    rois, layers = iterate_segmentation(image_folder)
    export_rois(project_folder, rois)

    # Apply prediction model
    iterate_predicting(layers, rois, cmap, project_folder, best_model)


# analyze(image_folder, cmap, best_model)

# processed_rois_path_1 = Path('C:/Users/mcanela/Desktop/sample_jose/cfos_analysis/rois_processed')
# images_path_1 = Path('C:/Users/mcanela/Desktop/sample_jose/cfos')
# processed_rois_path_2 = Path('C:/Users/mcanela/Desktop/sample_jose/tdt_analysis_with_model/rois_processed')
# images_path_2 = Path('C:/Users/mcanela/Desktop/sample_jose/tdt')
# colocalize(processed_rois_path_1, images_path_1, processed_rois_path_2, images_path_2)


def train(image_folder, model_type="svm"):

    # Segment images
    rois, layers = iterate_segmentation(image_folder)

    # Extract features and labels
    df = label_rois(rois, layers)
    features_path = image_folder.parent / "features.csv"
    df.to_csv(features_path, index=False)

    # Train model
    X_train, y_train, X_test, y_test = random_train_test_split(df)
    if model_type == "svm":
        random_search, best_model = train_svm(X_train, y_train)
    elif model_type == "rf":
        random_search, best_model = train_rf(X_train, y_train)
    elif model_type == "logreg":
        random_search, best_model = train_logreg(X_train, y_train)

    model_path = image_folder.parent / f"best_model_{model_type}.pkl"
    with open(model_path, "wb") as file:
        pkl.dump(best_model, file)

    print("Best parameters found: ", random_search.best_params_)
    print("Best cross-validation score: ", random_search.best_score_)

    return best_model, X_train, y_train, X_test, y_test


def expand_retrain(image_folder, df, model_type="svm"):

    # Segment images
    rois, layers = iterate_segmentation(image_folder)

    # Extract features and labels
    df_2 = label_rois(rois, layers)

    # Combine dataframes
    combined_df = pd.concat([df, df_2], ignore_index=False)
    features_path = image_folder.parent / "expanded_features.csv"
    combined_df.to_csv(features_path, index=True)

    # Train model
    X_train, y_train, X_test, y_test = random_train_test_split(combined_df)
    if model_type == "svm":
        random_search, best_model = train_svm(X_train, y_train)
    elif model_type == "rf":
        random_search, best_model = train_rf(X_train, y_train)
    elif model_type == "logreg":
        random_search, best_model = train_logreg(X_train, y_train)

    model_path = image_folder.parent / f"expanded_best_model_{model_type}.pkl"
    with open(model_path, "wb") as file:
        pkl.dump(best_model, file)

    print("Best parameters found: ", random_search.best_params_)
    print("Best cross-validation score: ", random_search.best_score_)

    return best_model, X_train, y_train, X_test, y_test


# path = image_folder.parent / 'features.csv'
# with open(path, 'rb') as file:
#     best_model = pkl.load(file)
# df = pd.read_csv(path, index_col=None)
# df = df.drop(columns=['Unnamed: 0'])
# X = X_test
# y = y_test


# df = pd.read_csv("C:/Users/mcanela/Desktop/sample_jose/tdt_analysis_with_model/counts.csv")
# split_df = df["file_name"].str.split("_", expand=True)
# df["id"] = split_df[0]
# grouped_df = df.groupby("id")["num_cells"].mean().reset_index()
# grouped_df.to_excel("C:/Users/mcanela/Desktop/sample_jose/counts_grouped.xlsx", index=False)


# split_df = df["file_name"].str.split("_", expand=True)
# df[["group", "id", "brain", "replica"]] = split_df
# grouped_df = df.groupby(["group", "id", "brain"])["cells_per_squared_mm"].mean().reset_index()
# grouped_df.columns = ["group", "id", "brain", "mean_cells_per_squared_mm"]
# grouped_df.to_excel(os.path.join(output_folder, "results_friendly.xlsx"), index=False)
