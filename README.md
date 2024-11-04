# CellRake 🔬

<img src="./CellRake.svg" style="width: 100%;" alt="CellRake">

## Why this package?

**CellRake** is a Python package that analyzes cells in fluorescent images. It provides tools for image segmentation based on [Scikit-image](https://scikit-image.org/stable/), model training and prediction based on [Scikit-learn](https://scikit-learn.org/stable/), and colocalization analysis, tailored for complex experiments involving multiple fluorescent markers.

## Installation

### Step 1: Install Conda

First, you need to install Conda, a package and environment management system. If you don't have Conda installed, you can download and install it by following these steps:

1. Go to the [Miniconda download page](https://docs.anaconda.com/miniconda/miniconda-install/).
2. Choose the version for your operating system (Windows, macOS, or Linux).
3. Follow the installation instructions provided on the website.

Miniconda is a minimal version of Anaconda that includes only Conda, Python, and a few essential packages, making it lightweight and easy to manage.

### Step 2: Create a Conda Environment

A Conda environment is an isolated space where you can install specific versions of Python and packages, like CellRake, without affecting other projects or installations. This is important to avoid conflicts between different software packages.

To create a new Conda environment for **CellRake**, open a terminal and run the following commands:

```console
conda create --name cellrake python=3.10
```

This command creates a new environment named `cellrake` with Python 3.9 installed.

### Step 3: Activate the Conda Environment

Before installing the CellRake package, you must activate the Conda environment you just created. This tells your system to use the Python and packages installed in that environment.

To activate the environment, run:

```console
conda activate cellrake
````

After running this command, your terminal prompt should change to indicate that you're now working within the `cellrake` environment.

### Step 4: Install CellRake

Now that your environment is set up and activated, you can install **CellRake**. The package is hosted on PyPI.

To install CellRake, run the following command:

```console
pip install cellrake
```

This command tells pip to install CellRake from the PyPI repository. Now you are ready to go!

**Note:** If you installed the package some time ago, I recommend updating to the newest version to avoid bugs. You can do this by running:

```bash
pip install cellrake --upgrade
```

## How to use it?

For detailed tutorials and use cases, see the [examples](./examples) directory:

- [Training Models](./examples/1_training_models.ipynb): Learn how to train a machine learning model using your dataset.
- [Analyzing Images](./examples/2_analyzing_images.ipynb): Analyze new images using a pre-trained model.
- [Colocalization Analysis](./examples/2_analyzing_images.ipynb): Perform colocalization analysis on multi-marker images.

## License

**CellRake** is licensed under the [MIT License](https://opensource.org/license/MIT).
