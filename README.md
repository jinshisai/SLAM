## pvfit

pvfit is a python module to determine representative data points from the position-velocity (PV) diagram through fitting of a Gaussian function. The detail of the method is presented in [Sai et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...893...51S/abstract).


## Basic Usage
Download the module and put it at the directory where a script to perform the PV fitting is contained, or any place added to your python path. For an easy test to check if the module works in your python environment, you can run `debug.py` first.

The simplest way to perform the fitting is as the following.

```
from pvfit import PVFit

# ----- input parameters ------
fitsfile = 'path_to_your_fitsfile'
outname  = 'outputfilename'
rms      = 1   # rms noise level
thr      = 6.  # threshold, n sigma
# ------------------------------


# ----------- main ------------
impv = PVFit(fitsfile) # Read the input fits file

# fitting along x-axis (i.e., fitting to the intensity distribution at each velocity channel)
impv.pvfit_xcut(outname, rms, thr)

# fitting along v-axis (i.e., fitting to the spectrum at each offset)
impv.pvfit_vcut(outname, rms, thr)
# -----------------------------
```
What done here is to fit a Gaussian function to the intensity distribution at each velocity or the spectrum at each offset. Peak positions or velocities are output as results. Uncertainties of peak positions and velocities are estimated from the covariance matrix which calculated in the chi-square fitting. The result is recoreded in a file named `outname`_chi2_pv_xfit.txt or `outname`_chi2_pv_vfit.txt.

To overplot fitting results on the PV diagram, you can do as follows.

```
clevels  = np.array([-3,3,6,9,12,15,20,25,30])*rms # Contour level
impv.plotresults_onpvdiagram(outname=outname, clevels=clevels) # Draw a PV diagram with the fitting results
```

`outname`.pdf is the output file.

## More Opitions
#### Specify Fitting Ranges
You can specify the x and v ranges for the fitting with the parameters of `xlim` and `vlim`. The unit will be in arcsec and km s^-1.

```
vlim = [2,6]   # from 2 to 6 km s^-1
xlim = [-5, 5] # from -5 to 5 arcsec
impv.pvfit_xcut(outname, rms, thr, vlim=vlim, xlim=xlim)
```

#### Read Results

To read the fitting results,
```
from pvfit import analysis_tools as at

offset, velocity, offerr, velerr = at.read_pvfitres('yourresults.txt')
```

#### Read Header Information of the PV Diagram
PVFit object contains the data of the fits file. You can look them and call the header information as follows.

```
impv = PVFit(fitsfile) # Read the input fits file
impv.fitsdata.__dict__ # List all contained info. of the fits file

# Call parameters
# E.g.,
impv.fitsdata.xaxis    # Call x-axis of the PV diagram
```

This function is basically a copy of [Imfits](https://github.com/jinshisai/Imfits). Use this to draw your own PV diagram or to look into data in more detail.