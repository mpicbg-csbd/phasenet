# Phasenet
Phasenet is a python package for fast aberration estimation from bead images using CNNs trained only on synthetic images.


## Installation:

Phasenet requires Python 3.6 and above.

Install via `pip install git+https://github.com/mpicbg-csbd/phase_net_code.git`

## Usage:

1)  Setup the config file and train the phasenet model
2)  Use the trained network to make predictions on bead images

## Troubleshoot

1)  To see if the training data is well agreement with the observed PSF use the PSF.ipynb notebook
2)  To check if the wavefront, use the Wavefront.ipynb notebook