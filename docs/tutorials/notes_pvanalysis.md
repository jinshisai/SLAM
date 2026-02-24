# Notes

* Edge/ridge x is derived by xcut in dv steps. Edge/ridge v is derived by vcut in bmaj / 2 steps.

* The derived points are removed (1) before the maximum value, (2) in the opposite quadrant, and (3) before the cross of xcut and vcut results. These removing operations can be switched on/off by nanbeforemax, nanopposite, and nanbeforecross, respectively, in get_edgeridge. Default is True for all the three.

* Each 1D profile of xcut and vcut are also provided as outname_pvfit_xcut.pdf and outname_pvfit_vcut.pdf.

* In outname.edge.dat and outname.ridge.dat files, xcut results have dv=0, and vcut results have dx=0.

* The double power-law fitting, fit_edgeridge, uses chi2 = ((x - Xmodel(v)) / dx)^2 + ((v - Vmodel(x)) / dv)^2. Vmodel = sgn(x) Vb (|x|/Rb)^-p, where p=pin for |x| < Rb and p=pin+dp for |x| > Rb. Xmodel(v) is the inverse function of Vmodel(x).

* dx and dv are the fitting undertainty for 'gaussian' ridge, the uncertainty propagated from the image noise for 'mean' ridge, and the image noise divided by the emission gradient for edge.

* Min, Mout, and Mb are stellar masses calculated from the best-fit model at the innermost, outermost, and Rb radii, respectively. When p is not 0.5, these masses are just for reference.