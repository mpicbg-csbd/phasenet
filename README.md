# PhaseNet

PhaseNet is a Python package for fast optical aberration estimation from bead images using CNNs trained only on synthetic images. Please see [our preprint](https://arxiv.org/abs/2006.01804) for details.


## Installation

PhaseNet requires Python 3.6 or later.

Install via `pip install git+https://github.com/mpicbg-csbd/phasenet.git`

## Usage

1)  Setup the config file and train the PhaseNet model as shown in the [training](https://github.com/mpicbg-csbd/phasenet/blob/master/notebooks/Training.ipynb) notebook.
2)  Use the trained network to make predictions on acquired bead images from the microscope as shown in the [prediction](https://github.com/mpicbg-csbd/phasenet/blob/master/notebooks/Prediction.ipynb) notebook.

## Troubleshoot

1)  Check if the training images are in well agreement with the observed PSF using the [PSF](https://github.com/mpicbg-csbd/phasenet/blob/master/notebooks/PSF.ipynb) notebook.
2)  Check the generation of wavefront using the [Wavefront](https://github.com/mpicbg-csbd/phasenet/blob/master/notebooks/Wavefront.ipynb) notebook.