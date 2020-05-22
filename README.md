# Phasenet
Phasenet is a python package for fast optical aberration estimation from bead images using CNNs trained only on synthetic images. 


## Installation

Phasenet requires Python 3.6 and above.

Install via `pip install git+https://github.com/mpicbg-csbd/phase_net_code.git`

## Usage

1)  Setup the config file and train the phasenet model as shown in the [training](https://github.com/mpicbg-csbd/phase_net_code/blob/master/notebooks/Training.ipynb) notebook
2)  Use the trained network to make predictions on acquired bead images from the microscope as shown in the [prediction](https://github.com/mpicbg-csbd/phase_net_code/blob/master/notebooks/Prediction.ipynb) notebook

## Troubleshoot

1)  Check if the training images is in well agreement with the observed PSF using the [PSF](https://github.com/mpicbg-csbd/phase_net_code/blob/master/notebooks/PSF.ipynb)notebook
2)  Check the generation of wavefront using the [Wavefront](https://github.com/mpicbg-csbd/phase_net_code/blob/master/notebooks/Wavefront.ipynb) notebook