import numpy
from astropy.io import fits
import matplotlib.pyplot as plt
import argparse

hdul = fits.open('tic0010048801.fits')
times = hdul[1].data['times']
fluxes = hdul[1].data['fluxes']
ferrs = hdul[1].data['ferrs']

