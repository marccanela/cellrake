"""
Created on Fri Aug  9 13:44:44 2024
@author: mcanela
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


def iterate_segmentation(image_folder):

    rois = {}
    layers = {}

    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    tif_paths = list(image_folder.glob("*.tif"))

    for tif_path in tqdm(tif_paths, desc="Segmenting images", unit="image"):
        tag = tif_path.stem

        polygon, layer = segment_image(tif_path, model)
        rois_dict = convert_to_roi(polygon, layer)

        rois[tag] = rois_dict
        layers[tag] = layer

    return rois, layers


def export_rois(project_folder, rois):

    raw_folder = project_folder / "rois_raw"

    for tag, rois_dict in rois.items():
        pkl_path = raw_folder / f"{tag}.pkl"
        with open(str(pkl_path), "wb") as file:
            pkl.dump(rois_dict, file)


def segment_image(tif_path, model):

    image = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)

    # Compress image (optional)
    image = compress(image)

    # Extract layer from image
    layer = get_layer(image)

    # Eliminate rows and columns that are entirely zeros
    layer = crop(layer)

    # Normalize layer
    norm_layer = normalize(layer)

    # Apply StarDist2D
    _, polygon = model.predict_instances(
        norm_layer, prob_thresh=0.5, nms_thresh=0.4, verbose=False
    )

    return polygon, layer
