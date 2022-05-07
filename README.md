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

* TBD...

 
## Authors

* Name: Jinshi Sai
* Affiliation: Academia Sinica Institute of Astronomy and Astrophysics
* E-mail: jsai@asiaa.sinica.edu.tw
* Name: Yusuke Aso
* Affiliation: Korea Astronomy and Space Science Institute
* E-mail: yaso@kasi.re.kr
 
## License
 
"SLAM" is under [GNU General Public License Version 3](https://www.gnu.org/licenses/gpl-3.0.html).
