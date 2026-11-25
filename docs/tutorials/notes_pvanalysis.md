# Notes

* Edge/ridge x is derived by xcut in dv steps. Edge/ridge v is derived by vcut in bmaj / 2 steps.

* The derived points are removed (1) before the maximum value, (2) in the opposite quadrant, and (3) before the cross of xcut and vcut results. These removing operations can be switched on/off by nanbeforemax, nanopposite, and nanbeforecross, respectively, in get_edgeridge. Default is True for all the three.

* Each 1D profile of xcut and vcut are also provided as outname_pvfit_xcut.pdf and outname_pvfit_vcut.pdf.

* In outname.edge.dat and outname.ridge.dat files, xcut results have dv=0, and vcut results have dx=0.

* The double power-law fitting, fit_edgeridge, uses chi2 = ((x - Xmodel(v)) / dx)^2 + ((v - Vmodel(x)) / dv)^2. Vmodel = sgn(x) Vb (|x|/Rb)^-p, where p=pin for |x| < Rb and p=pin+dp for |x| > Rb. Xmodel(v) is the inverse function of Vmodel(x).

* dx and dv are the fitting undertainty for 'gaussian' ridge, the uncertainty propagated from the image noise for 'mean' ridge, and the image noise divided by the emission gradient for edge.

* Min, Mout, and Mb are stellar masses calculated from the best-fit model at the innermost, outermost, and Rb radii, respectively. When p is not 0.5, these masses are just for reference.

## Posterior summaries and display selection

`fit_edgeridge(outname='pvanalysis', best='pmid')` always writes
`pvanalysis.popt_e.txt` (edge) and `pvanalysis.popt_r.txt` (ridge) after a
successful fit. The prefix follows `outname`, just like the corner plots.
Skipped fits do not write files; existing files are not removed.

Each file has parameter rows `Rb, Vb, p_in, dp, dVsys` followed by a one-line
mass note and rows `M_b, M_in, M_out` (in solar masses), with columns
`popt, plow, pmid, phigh`: maximum posterior probability, 16th percentile,
median, and 84th percentile. Radius is in au and velocity is in km/s;
`dVsys` is an offset from the input systemic velocity. Fixed parameters have
identical entries in all four columns. Load the numeric table with `np.loadtxt`.

`best='pmid'` is the default and preserves the previous median parameter
selection. Use `best='popt'` to select the maximum-probability parameter vector.
`output_fitresult(best='popt')` and `plot_fitresult(best='popt')` can change
this selection after fitting. Subsequent reporting and model plots retain the
selection, with no new sampling or changes to saved files. Corner plots show
the posterior and the sampler's maximum-probability reference independently.
The legacy `self.popt` and returned `popt` still contain medians and symmetric
half-percentile-width errors; `self.fit_summaries[method]` holds all four summaries.

For `best='popt'`, terminal uncertainties use `value (-x, +y)`, where
`x = max(value - plow, 0)` and `y = max(phigh - value, 0)`. For systemic
velocity, the input systemic velocity is added to the displayed central value
and both bounds.

Joint posterior samples determine mass uncertainties, preserving parameter
correlations. For every sample, the existing fitted-range rules determine the
inner and outer boundaries; masses are evaluated there and at the break radius.
`mass_samples[method]` has rows `M_in, M_out, M_b` and one column per sample.
`mass_summaries[method]` holds `popt, plow, pmid, phigh` for those three masses.
For `best='popt'`, central masses come from the maximum-probability parameter
vector and its boundaries, with errors measured to the mass posterior's 16th
and 84th percentiles and clipped at zero. This central mass is not necessarily
the mode of the mass posterior. For `best='pmid'`, preserve the original
terminal output: fitted parameters are medians with symmetric
`(phigh - plow) / 2` errors, and masses are evaluated at those median parameters
and their fitted boundaries, with the original analytic error propagation.
All values use `value +/- error` formatting. Mass posterior summaries remain
available for inspection but are not used for the `pmid` terminal report. These are reference Keplerian masses where the index differs
from 0.5. Distance and inclination are held fixed in this posterior calculation.

The mass rows always contain the best-model mass in `popt` and the mass
posterior percentiles in `plow, pmid, phigh`, independently of `best`. Thus their
`pmid` column differs in definition from the legacy `best='pmid'` terminal
mass calculation. `np.loadtxt` returns an 8-by-4 table, ignoring the comment
note between the parameter and mass blocks. `fit_linear` retains its existing
workflow and does not produce these posterior files.

### Point and fit state

`get_edgeridge(..., methods=('edge', 'ridge'))` and
`fit_edgeridge(..., methods=('edge', 'ridge'))` select both methods by default.
Pass `methods=('ridge',)` or `methods=('edge',)` to select one method.
Extraction currently computes both estimators internally, retaining only the
requested methods. Extraction replaces the complete point dataset, including
previously enabled cuts; unselected methods contain empty arrays.

Point containers exist immediately after initialization. Empty point arrays have
shape `(4, 0)` in the existing `[x, v, dx, dv]` order. `point_status` distinguishes
`not_run`, `disabled`, `no_points`, and `success`. Individual invalid measurements
may still contain NaNs before filtering.

`popt` always has `edge` and `ridge` keys. An unavailable fit is `None`; successful
entries retain the existing `[parameters, uncertainties]` structure and parameter
order. The return value of `fit_edgeridge` also has both keys, with `None` for
unavailable fits. `fit_status` records `not_run`, `disabled`, `no_points`, `success`,
or `failed` (a sampling error still propagates to the caller).

Extracting or re-filtering points clears previous fit parameters, summaries,
chains, mass samples, ranges, and reduced chi-square values. Fitting only one
method replaces only its products, preserving any fit of the other method.
Automatic prior ranges use only the selected nonempty datasets. `chain`, `lnp`,
and `rvlim` are always dictionaries with both method keys; unavailable entries
are `None`. Existing code indexing an unavailable fit must first check for None.

### Independent fitting and reporting

`fit_edgeridge()` fits each requested nonempty method independently. Empty methods
have a `no_points` status and a `None` result, without preventing the other fit.
When both are fitted together, automatic priors remain shared across their
combined data range. If neither has points, the method returns the current
structured results without sampling or writing files. Unselected existing fits
remain available. Exceptions propagate, mark the affected fit as `failed`, and
clear its partial products.

`fit_linear()` fits ridge points even when no edge points exist. It retains the
existing preference for position cuts when available, but no longer deletes
velocity-cut points. It replaces only the ridge fit; edge parameters, posterior
products, and model remain available. `models` stores each method's model;
`model` retains the legacy most-recent-fit interface. Linear results contain
`[intercept, gradient]` and have no power-law mass posterior.

`get_range()` and `output_fitresult()` handle absent fits, power-law fits, linear
fits, and mixed edge/ridge fit types. Power-law range and mass calculations use
the cut types actually present for that method. With both cuts, the inner
boundary uses the maximum absolute position-cut velocity, and the outer boundary
uses the maximum absolute velocity-cut position. With only position cuts, both
boundaries use their velocity extrema; with only velocity cuts, both use their
position extrema. Missing ranges are `None`.
Both reporting estimators (`pmid` and `popt`) support independent methods; linear
fits report their analytic coefficients for either selection.

### Shared emcee and corner options

`fit_edgeridge()` accepts `kwargs_emcee_corner=None` and `save_corner=True`.
Both settings apply to all methods selected in that call. Existing calls retain
their sampling defaults and corner filenames (`.corner_e.png` and `.corner_r.png`).
For example:

```python
impv.fit_edgeridge(
    methods=('edge', 'ridge'),
    save_corner=False,
    kwargs_emcee_corner={
        'nwalkers_per_ndim': 16,
        'nburnin': 500,
        'nsteps': 1000,
        'rangelevel': None,
    },
)
```

Options are forwarded to `utils.emcee_corner`, including `range_corner`,
`labels`, `moves`, and chain-plot controls. Supplied options override ordinary
defaults, including the top-level `rangelevel` and `show_corner`. The dictionary
is copied rather than modified. A custom `figname` applies to every selected
method; use separate method calls to keep different custom filenames or sampling
settings, preserving the other method's existing fit.

`save_corner=False` overrides even a supplied `figname`, forcing it to None.
It disables corner-file saving and derived chain-file saving, while display
remains controlled by `show_corner` and `show_chain`. Setting `figname=None`
in the options also disables saving. Parameter-summary files remain enabled,
and terminal messages list only the corner filenames used for successful fits.
Skipped methods produce no corner-file announcement.

The internal arguments `bounds`, `log_prob_fn`, `args`, `ndata`, `simpleoutput`,
`return_chain`, and `return_lnp` are protected. Supplying any of them in the
options raises a clear ValueError before changing fit state. Use the top-level
`return_chain` and `return_lnp` arguments to control stored outputs. Sampling
always returns chains internally because mass posteriors require them.
These options configure emcee; the optional dynesty evidence calculation keeps
its existing configuration.

### Loading saved edge/ridge points

`read_edgeridge(outname, methods=('edge', 'ridge'), quadrant=None)` loads the
`.edge.dat` and `.ridge.dat` files written by `write_edgeridge()`. `outname` is
the filename prefix, rather than the complete filename. Both strings and Path
objects are accepted. To fit saved ridge points in a new session:

```python
impv = PVAnalysis(fitsfile, rms, vsys, dist, incl=incl)
impv.read_edgeridge('saved_points', methods=('ridge',))
impv.fit_edgeridge(
    methods=('ridge',),
    outname='new_fit',
    save_corner=False,
    kwargs_emcee_corner={'nburnin': 500, 'nsteps': 1000},
)
```

Loading restores the fitting dataset directly, without applying turnover,
quadrant, crossing, mass, or uncertainty-floor filters again. It populates the
raw, filtered, and sorted point containers, replaces previous points, and clears
all previous fits. Unrequested methods are cleared and marked `disabled`.
A header-only file means `no_points`. Missing requested files raise
FileNotFoundError; malformed rows or metadata raise ValueError. Every requested
file is validated before existing state is changed.

The numeric format remains four columns: `x, dx, v, dv` in au, au, km/s, km/s.
`dv == 0` identifies position-cut points; `dx == 0` identifies velocity-cut
points. Exactly one uncertainty must be zero and the other must be positive.
All entries must be finite. Row validation errors include the filename and line.
New files retain the original column header and add a versioned JSON comment
recording the method, units, position-sign convention, xsign, and input systemic
velocity. Existing `numpy.loadtxt` consumers can still read the numeric data.
`write_edgeridge(..., methods=('ridge',))` writes only the requested method;
requested empty methods produce header-only files.

File positions are normalized by xsign; velocities are relative to the input
systemic velocity. New metadata restores physical position signs. If the current
input vsys differs from the saved value, loading shifts relative velocities to
preserve absolute velocities. Saved red/blue grouping is retained. Positions
remain in au without distance rescaling; the current distance is used only to
reconstruct the raw arcsecond coordinates. The current inclination is retained.

Legacy files without metadata assume the current input vsys. Orientation comes
from an explicit `quadrant='13'` or `'24'`, the object's existing quadrant, or
inference from the PV image, in that order. Supply the original quadrant when it
was manually chosen during extraction. For new files, conflicting quadrant
arguments or inconsistent edge/ridge coordinate metadata raise ValueError.

The files contain already-filtered points, so loading cannot recover discarded
measurements. Calling `sort_fitresults()` explicitly can refilter the surviving
points; loading itself does not invoke it.

### Plotting at any workflow stage

`plot_fitresult()` works immediately after constructing `PVAnalysis`, after
extraction or loading, and after fitting. The PV intensity image and contours
are always drawn. Available points and fitted models are overlaid according to
`plotedgepoint`, `plotridgepoint`, `plotedgemodel`, and `plotridgemodel`.
Unavailable overlays are skipped automatically, including when extraction found
no points or a fit was skipped or failed. No dummy fit parameters are needed.
Both linear and log-log PNG files are produced as before.

```python
impv = PVAnalysis(fitsfile, rms, vsys, dist, incl=incl)
impv.plot_fitresult(outname='image_only', show=False)

impv.read_edgeridge('saved_points', methods=('ridge',))
impv.plot_fitresult(outname='ridge_points', show=False)
# After fitting, the same call also draws the available ridge model.
```

All high-level plotting methods accept `vsys_mode` and `best`:

| vsys_mode | Velocity reference |
| --- | --- |
| `'input'` (default) | Input `vsys`, regardless of fitted offsets |
| `'edge'` | Input `vsys` plus the selected edge power-law offset |
| `'ridge'` | Input `vsys` plus the selected ridge power-law offset |
| `'mean'` | Input `vsys` plus the mean of available successful power-law offsets; input `vsys` if none exist |

Explicit edge/ridge modes raise ValueError if that method has no successful
power-law fit. Linear-fit intercepts are not treated as systemic offsets.
`best='pmid'` or `'popt'` selects the fitted offset estimator as well as the model
parameters. The image, points, and models share the chosen velocity reference.
The default now uses input `vsys`, replacing the previous automatic averaging
of edge and ridge offsets; use `vsys_mode='mean'` to request averaging explicitly.

On folded log-log plots, both model branches are drawn when they differ after
velocity centering. This preserves fitted offsets and linear intercepts instead
of cancelling them by averaging the branches. Stored parameters, posterior
samples, and extracted points are never modified by plotting.

`plot_point()` and `plot_model()` also skip unavailable overlays. Their optional
`vsys_mode=None` retains the current plot reference (initially input `vsys`), so
both helpers can draw on the same axes. Custom model parameters supplied as a
list or NumPy array are copied; a custom model can use the image radius extent
when no fitted range exists.

The older `plotresults_pvdiagram()` and `plotresults_rvplane()` also support
empty and single-method datasets. They do not refilter points or invalidate fits.
Relative-velocity quick-look diagrams use the selected center; absolute-velocity
diagrams retain absolute velocities. Position uncertainties remain nonnegative
in both quadrants. The legacy image drawer supports 2D and singleton-Stokes PV
arrays. Blank maps without positive intensities can use logarithmic color
scaling; their color layer is masked with a finite default color range.
