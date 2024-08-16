# CellRadar 📡🔬

## Why this package?

**Cell Radar** is a Python package designed to analyze cells in fluorescent images. It provides tools for image segmentation based on [StarDist](https://github.com/stardist/stardist), model training based and prediction based on [Scikit-learn](https://scikit-learn.org/stable/), and colocalization analysis, tailored for complex experiments involving multiple fluorescent markers.

## Installation

### Step 1: Install Conda

First, you need to install Conda, a package and environment management system. If you don't have Conda installed, you can download and install it by following these steps:

1. Go to the [Miniconda download page](https://docs.anaconda.com/miniconda/miniconda-install/).
2. Choose the version for your operating system (Windows, macOS, or Linux).
3. Follow the installation instructions provided on the website.

Miniconda is a minimal version of Anaconda that includes only Conda, Python, and a few essential packages, making it lightweight and easy to manage.

### Step 2: Create a Conda Environment

A Conda environment is an isolated space where you can install specific versions of Python and packages, like CellRadar, without affecting other projects or installations. This is important to avoid conflicts between different software packages.

To create a new Conda environment for **CellRadar**, open a terminal and run the following commands:

```console
conda create --name cellradar python=3.9
```

This command creates a new environment named `cellradar` with Python 3.9 installed.

### Step 3: Activate the Conda Environment

Before installing the CellRadar package, you need to activate the Conda environment you just created. This tells your system to use the Python and packages installed in that environment.

To activate the environment, run:

```console
conda activate cellradar
````

After running this command, your terminal prompt should change to indicate that you're now working within the `cellradar` environment.

### Step 4: Install CellRadar

Now that your environment is set up and activated, you can install **CellRadar**. The package is hosted on PyPI.

To install CellRadar, run the following command:

```console
pip install -i https://test.pypi.org/simple/ cellradar
```

This command tells pip to install CellRadar from the PyPI repository. Now you are ready to go!

## How to use it?

For detailed tutorials and use cases, see the [examples](./examples) directory:

- [Training Models](./examples/1_training_models.ipynb): Learn how to train a machine learning model using your dataset.
- [Analyzing Images](./examples/2_analyzing_images.ipynb): Analyze new images using a pre-trained model.
- [Colocalization Analysis](./examples/2_analyzing_images.ipynb): Perform colocalization analysis on multi-marker images.

## License

**Cell Radar** is licensed under the [MIT License](https://opensource.org/license/MIT).