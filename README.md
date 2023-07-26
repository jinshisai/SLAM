[![Documentation Status](https://readthedocs.org/projects/slam-astro/badge/?version=latest)](https://slam-astro.readthedocs.io/en/latest/?badge=latest)

# SLAM: Spectral Line Analysis/Modeling
SLAM (Spectral Line Analysis/Modeling) is a python library to analyze/model spectral line data especially obtained with radio telescopes. The current main package, pvanalysis, is to derive the rotational velocity as a function of radius and fit the profile with a power-law function. The detail of the method is presented in [Aso et al. 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...812...27A/abstract), [Sai et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...893...51S/abstract) and the reference therein. More analysis tools will be coming in future.


## Demo and Usage
 
The file example.py will help to find how to use pvanalysis.
```bash
git clone https://github.com/jinshisai/SLAM
cd SLAM
python example.py
```
To keep the package updated, type the command below in the directory SLAM, always before you use.
```bash
git pull
```
Also, setting the path in .bashrc (or .zshrc etc.) will be useful.
```bash
export PYTHONPATH=${PYTHONPATH}:/YOUR_PATH_TO/SLAM
```
 
## Features
 
pvanalysis can do the following things.
* Get edge and ridge points in the directions of positional and velocity axes from a FITS file of a position-velocity diagram.
* Write out the derived points in text files.
* Fit the derived points with the Keplerian, single power-law, or double power-law functions, using the MCMC method (emcee).
* Systemic velocity Vsys can also be a free parameter.
* Get the central stellar mass calculated from the best-fit functions, in addition to the fitting parameters.
* Plot the derived points and best-fit functions on the position-velocity diagram in the linear or logR-logV planes.

 
## Requirement

* astropy
* copy
* corner
* emcee
* dynesty
* math
* matplotlib
* mpl_toolkits
* numpy
* scipy

 
## Installation
 
Download from https://github.com/jinshisai/SLAM or git clone.
```bash 
git clone https://github.com/jinshisai/SLAM
```
 
## Note

* Edge/ridge x is derived by xcut in dv steps. Edge/ridge v is derived by vcut in bmaj / 2 steps.
* The derived points are removed (1) before the maximum value, (2) in the opposite quadrant, and (3) before the cross of xcut and vcut results. These removing operations can be switched on/off by nanbeforemax, nanopposite, and nanbeforecross, respectively, in get_edgeridge. Default is True for all the three.
* Each 1D profile of xcut and vcut are also provided as outname_pvfit_xcut.pdf and outname_pvfit_vcut.pdf.
* In outname.edge.dat and outname.ridge.dat files, xcut results have dv=0, and vcut results have dx=0.
* The double power-law fitting, fit_edgeridge, uses chi2 = ((x - Xmodel(v)) / dx)^2 + ((v - Vmodel(x)) / dv)^2. Vmodel = sgn(x) Vb (|x|/Rb)^-p, where p=pin for |x| < Rb and p=pin+dp for |x| > Rb. Xmodel(v) is the inverse function of Vmodel(x).
* dx and dv are the fitting undertainty for 'gaussian' ridge, the uncertainty propagated from the image noise for 'mean' ridge, and the image noise divided by the emission gradient for edge.
* Min, Mout, and Mb are stellar masses calculated from the best-fit model at the innermost, outermost, and Rb radii, respectively. When p is not 0.5, these masses are just for reference.

 
## Authors

* Jinshi Sai
    * Affiliation: Academia Sinica Institute of Astronomy and Astrophysics
    * E-mail: jsai@asiaa.sinica.edu.tw
* Yusuke Aso
    * Affiliation: Korea Astronomy and Space Science Institute
    * E-mail: yaso@kasi.re.kr
 
## License
 
"SLAM" is under [GNU General Public License Version 3](https://www.gnu.org/licenses/gpl-3.0.html).
