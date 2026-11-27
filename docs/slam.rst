SLAM API
========

Main classes
------------

PV analysis
~~~~~~~~~~~

.. autoclass:: pvanalysis.PVAnalysis
   :members:
   :show-inheritance:

PV fitting
~~~~~~~~~~

.. autoclass:: pvfitting.PVFitting
   :members:
   :show-inheritance:

.. autoclass:: pvfitting.MockPVD
   :members:
   :show-inheritance:

Channel fitting
~~~~~~~~~~~~~~~

.. autoclass:: channelfit.ChannelFit
   :members:
   :show-inheritance:

Velocity-gradient analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: velgrad.VelGrad
   :members:
   :show-inheritance:

``filtering()`` and ``calc_mstar()`` accept ``kwargs_emcee_corner`` to configure
sampling and corner plots. For example, pass
``kwargs_emcee_corner={'nburnin': 500, 'nsteps': 1000}`` to either method.
Omitting it uses 16 walkers per dimension, 2000 burn-in steps, and 2000
production steps. ``figname`` and ``show_corner`` control saving and display.
Filtering applies these options to every preliminary and final fit, so a shared
filename is overwritten by subsequent passes.

Named ``range_corner`` dictionaries use ``xoff``, ``yoff``, ``pa`` for gradient
fits, or ``M_p``, ``v_break``, ``p_low``, ``voff`` for mass fits. Fixed parameters
are omitted and unspecified ranges default to 0.8. These names do not change
when custom plot labels are supplied. ``p_low`` refers to the sampled
radius-versus-speed exponent, before the reciprocal used in terminal reports.
Use the explicit ``return_chain`` and ``return_lnp`` arguments to retain sampler
outputs; the internal bounds, likelihood arguments, and output format are
protected. The caller's options dictionary is not modified.

PV-analysis utilities
---------------------

Analysis tools
~~~~~~~~~~~~~~

.. automodule:: pvanalysis.analysis_tools
   :members:
   :show-inheritance:

Fitting functions
~~~~~~~~~~~~~~~~~

.. automodule:: pvanalysis.fitfuncs
   :members:
   :show-inheritance:

Plotting tools
~~~~~~~~~~~~~~

.. automodule:: pvanalysis.pvplot
   :members:
   :show-inheritance:

PV-fitting utilities
--------------------

Nested grids
~~~~~~~~~~~~

.. automodule:: pvfitting.grid
   :members:
   :show-inheritance:

Model precalculation
~~~~~~~~~~~~~~~~~~~~

.. automodule:: pvfitting.precalculation
   :members:
   :show-inheritance:

Shared utilities
----------------

.. automodule:: utils
   :members:
   :show-inheritance:

Noise and intensity cutoffs
--------------------------

All four analysis modules use ``rms`` for the RMS noise in the same intensity
units as the input data. ``cutoff`` is a dimensionless noise multiplier: the
intensity threshold is ``cutoff * rms``. A noise input of ``None`` retains
automatic estimation where it was previously supported.

Existing ``sigma=`` noise inputs remain accepted as aliases for ``rms=``.
The edge/ridge extraction methods accept ``thr=`` as an alias for ``cutoff=``;
channel-map moment and CLEAN helpers also retain ``threshold=`` as an alias.
Positional arguments and defaults are unchanged. Supplying both names for one
parameter raises ``TypeError``, even if their values match. Legacy keyword
names emit ``FutureWarning`` recommending the preferred name; they remain
supported.

Noise is stored as ``self.rms``. In PVFitting, ChannelFit, and VelGrad, the
compatibility property ``self.sigma`` reads and writes the same value.
PVAnalysis has always used ``self.rms`` and has no ``self.sigma`` alias.
FITS-reader result dictionaries include both
``rms`` and the legacy ``sigma`` key. Existing moment-noise names such as
``sigma_mom0`` are unchanged. Gaussian widths, statistical uncertainties,
regularization thresholds, and arguments belonging to external functions keep
their existing names.
