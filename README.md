## pvanalysis

pvanalysis is a python package to perform fitting to the rotational curve using the position-velocity (PV) diagram. The detail of the method is presented in [Aso et al. 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...812...27A/abstract), [Sai et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...893...51S/abstract) and reference therein.


## Basic Usage
Download the module and put it at the directory where you run a script to perform the fitting, or any place added to your python path. For an easy test to check if the module works in your python environment, you can run `example.py` first.

The simplest way to perform the fitting is as the following.

```
from pvanalysis import PVAnalysis

# ----- input parameters ------
fitsfile = 'path_to_your_fitsfile'
outname  = 'outputfilename'
rms      = 1    # rms noise level
dist     = 140. # distance to the object (pc)
vsys     = 5.   # systemic velocity of the object (km/s)
pa       = 60.  # P.A. of the PV cut (degree, optional)
# ------------------------------


# ----------- main ------------
impv = PVAnalysis(fitsfile, rms, vsys, dist, pa=pa) # Read the input fits file
impv.get_edgeridge(outname) # get edge/ridge (representative data points) on a PV diagram

# quick check
impv.plotresults_pvdiagram(clevels=np.array([-3,3,6,9,12,15])*rms)
impv.plotresults_rvplane()

# fitting to rotational curve
impv.fit_edgeridge(
	include_vsys=False,
	include_dp=True,
	include_pin=True,
	show_corner=False)
# -----------------------------
```