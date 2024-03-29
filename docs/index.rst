.. SLAM documentation master file, created by
   sphinx-quickstart on Thu Jun  8 15:40:30 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SLAM: Spectral Line Analysis/Modeling
================================

SLAM (Spectral Line Analysis/Modeling) is a python library to analyze/model spectral line data especially obtained with radio telescopes. The current main package, pvanalysis, is to derive the rotational velocity as a function of radius and fit the profile with a power-law function. The detail of the method is presented in Aso et al. (2015), Sai et al. (2020) and the reference therein. More analysis tools will be coming in future.


Installation
==================

In a local directory where you would like to install SLAM, type

::

   git clone https://github.com/jinshisai/SLAM

Also, setting the path in .bashrc (or .zshrc etc.) will be useful.

::

   export PYTHONPATH=${PYTHONPATH}:/YOUR_PATH_TO/SLAM


Versions
==================

The latest, stable version is v1.0.0.

To use the stable version, go to the SLAM directory and type


.. code-block:: bash

   git tag -l # list tags (versions)

   git checkout tags/v1.0.0 # choose v1.0.0


Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/example_pvanalysis