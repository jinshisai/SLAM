[![Documentation Status](https://readthedocs.org/projects/slam-astro/badge/?version=latest)](https://slam-astro.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7944158.svg)](https://doi.org/10.5281/zenodo.7944158)


# SLAM: Spectral Line Analysis/Modeling
**SLAM (Spectral Line Analysis/Modeling)** is a Python library for analyzing and modeling spectral line data at (sub)millimeter wavelengths, with a particular focus on rotational motions around (proto)stellar objects. The current release (v2) includes several packages for deriving rotation curves and estimating the dynamical mass of the central object. Details of the methods are presented [Aso & Sai (2024)](https://ui.adsabs.harvard.edu/abs/2024PKAS...39...27A/abstract) and Aso, Sai et al. (2026) in prep.


## Features

- `pvanalysis` extracts rotational curves from position-velocity (PV) diagrams and fits them with a power-law (or double-power-law) function. The detail of the method is presented in [Aso et al. 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...812...27A/abstract), [Sai et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...893...51S/abstract), [Aso & Sai (2024)](https://ui.adsabs.harvard.edu/abs/2024PKAS...39...27A/abstract), and references therein.
- `velgrad` extracts rotation curves using the velocity gradient method, which derives two dimensional mean positions as a function of velocity.
- `pvfitting` fits observed PV diagrams (taken along the disk major and/or minor axes) with model PV diagrams of a protostellar system that consists of a Keplerian disk and a UCM envelope (Ulrich 1976; Cassen & Moosman 1981). This package enables simultaneous estimation of rotational and infalling velocities, from which the dynamical mass of the central object and deceleration factor for infall motion can be derived.
- `channelfit` fits the velocity channel maps with a Keplerian disk model to estimate the dynamical mass of the central object.

For basic usages, please see `example_xxx.py` in this library, where xxx is the name of each package.

 
## Requirement

* numpy
* scipy
* copy
* math
* matplotlib
* mpl_toolkits
* astropy
* emcee
* corner
* dynesty
 
## Installation
 
You can install SLMA with `git clone`.

```bash 
git clone https://github.com/jinshisai/SLAM
```

To keep the library updated, type the command below in the directory SLAM, always before you use.

```bash
git pull
```

Also, setting the path in .bashrc (or .zshrc etc.) will be useful.

```bash
export PYTHONPATH=${PYTHONPATH}:/YOUR_PATH_TO/SLAM
``` 
 
## Authors

* *Jinshi Sai*, Kagoshima University (jinshi.sai@sci.kagoshima-u.ac.jp)
* *Yusuke Aso*, Korea Astronomy and Space Science Institute (yaso@kasi.re.kr)

## Citation

If you use SLAM for publications, please cite Aso, Sai et al. (2026 in prep) after it comes out, as well as [the latest release in Zenodo](https://doi.org/10.5281/zenodo.7944158):

```
@software{aso_2026_7944158,
  author       = {{Aso}, Yusuke and {Sai}, Jinshi},
  title        = {SLAM v2.0.0},
  month        = feb,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v2.0.0},
  doi          = {10.5281/zenodo.7944158},
  url          = {https://doi.org/10.5281/zenodo.7944158},
}
```

For details of the methodology of `pvanalysis`, the manual paper [Aso & Sai (2024)](https://pkas.kas.org/journal/article.php?code=91201&list.php?m=1) can be also referred:

```
@article{2024PKAS...39...2A,
    author      = {{Aso}, Yusuke and {Sai}, Jinshi},
    title       = {{SPECTRAL LINE ANALYSIS/MODELING (SLAM) I: PVANALYSIS}},
    booltitle   = {{Vol.39 No.2}},
    journal     = {{PKAS}},
    volume      = {{39}},
    issue       = {{2}},
    publisher   = {Korean Astronomical Society},
    year        = {2024},
    pages       = {27-38},
    doi         = {10.5303/PKAS.2024.39.2.027},
    URL         = {http://pkas.kas.org/journal/article.php?code=91201}
}
```

## License
 
SLAM is under [GNU General Public License Version 3](https://www.gnu.org/licenses/gpl-3.0.html).
