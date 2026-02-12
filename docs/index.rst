.. SLAM documentation master file, created by
   sphinx-quickstart on Thu Jun  8 15:40:30 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SLAM: Spectral Line Analysis/Modeling
================================

**SLAM (Spectral Line Analysis/Modeling)** is a Python library for analyzing and modeling spectral line data at (sub)millimeter wavelengths, with a particular focus on rotational motions around (proto)stellar objects. The current release (v2) includes several packages for deriving rotation curves and estimating the dynamical mass of the central object. Details of the methods are presented `Aso & Sai (2024) <https://ui.adsabs.harvard.edu/abs/2024PKAS...39...27A/abstract>`_ and Aso, Sai et al. (2026) in prep.


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

The latest, stable version is v1.2.0.

To use the stable version, go to the SLAM directory and type


.. code-block:: bash

   git tag -l # list tags (versions)

   git checkout tags/v1.2.0 # choose v1.2.0


Contents
=================

PV Analysis
------------

.. toctree::
   :maxdepth: 2

   tutorials/example_pvanalysis
   tutorials/notes_pvanalysis.md


More documents are coming soon
----------

.. toctree::
   :maxdepth: 2