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
