---
title: Cell Radar 📡🔬
subject: Tutorial
subtitle: A package for analyzing cells in fluorescent images
short_title: Cell Radar Tutorial
authors:
  - name: Marc Canela
    affiliations:
      - Hospital del Mar Research Institute
    orcid: 0000-0002-6248-4202
    email: mcanela@researchmar.net
license: MIT License
---
## Why this package?

**Cell Radar** is a Python package designed to analyze cells in fluorescent images. It provides tools for image segmentation, model training, and predictions on new data.

## Installation

To install **Cell Radar**, you can clone the repository and install it using `pip`:

```bash
git clone https://github.com/marccanela/cellradar.git
cd cellradar
pip install -i https://test.pypi.org/simple/ cellradar==0.1.0
```

Make sure you have the required dependencies installed. You can find them in `pyproject.toml`.

## Quick Start

Here’s a quick example to get you started:

## Usage
### Utilities

The `utils` module contains helper functions for loading images, preprocessing data, and more.

### Segmentation

The `segmentation` module provides tools for segmenting cells from images using [StarDist](https://github.com/stardist/stardist). See also [](10.1007/978-3-030-00934-2_30), [](10.1109/WACV45572.2020.9093435), and [](10.1109/ISBIC56247.2022.9854534).

### Training

The `training` module allows you to train models on your segmented cell data.

### Predicting

The `predicting` module is used for making predictions on new cell data using trained models.

## Examples

Check out the `examples` directory for more detailed tutorials and use cases.

## License

**Cell Radar** is licensed under the [MIT License](https://opensource.org/license/MIT).