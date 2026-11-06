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

Each file has rows `Rb, Vb, p_in, dp, dVsys` and columns
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

The parameter files contain fitted parameters only. `fit_linear` retains its
existing workflow and does not produce these posterior files.
