# CellRadar 📡🔬

## Why this package?

**Cell Radar** is a Python package designed to analyze cells in fluorescent images. It provides tools for image segmentation based on [StarDist](https://github.com/stardist/stardist), model training based and prediction based on [Scikit-learn](https://scikit-learn.org/stable/), and colocalization analysis, tailored for complex experiments involving multiple fluorescent markers.

## Installation

To install **Cell Radar**, you can clone the repository and install it using `pip`:

```bash
git clone https://github.com/marccanela/cellradar.git
cd cellradar
pip install -i https://test.pypi.org/simple/ cellradar==0.1.0
```

Make sure you have the required dependencies installed. You can find them in `pyproject.toml`.

## How to use it?

For detailed tutorials and use cases, see the [examples](./examples) directory:

- [Training Models](./examples/1_training_models.ipynb): Learn how to train a machine learning model using your dataset.
- [Analyzing Images](./examples/2_analyzing_images.ipynb): Analyze new images using a pre-trained model.
- [Colocalization Analysis](./examples/2_analyzing_images.ipynb): Perform colocalization analysis on multi-marker images.

## License

**Cell Radar** is licensed under the [MIT License](https://opensource.org/license/MIT).