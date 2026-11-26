import numpy as np
from pvanalysis import PVAnalysis

'-------- INPUTS --------'
fitsfile = './testfits/test.pvanalysis.fits'
outname = 'pvanalysis'  # file name header for outputs
methods = ('edge', 'ridge')  # Or ('edge',) or ('ridge',).
incl = 48.0  # deg
vsys = 6.4  # km/s
dist = 140.0  # pc
rms = 1.7e-3  # Jy/beam
thr = 5.0  # rms
ridgemode = 'mean'  # 'mean' or 'gauss'
xlim = [-200, 0, 0, 200]  # au; [-outlim, -inlim, inlim, outlim]
vlim = np.array([-5, 0, 0, 5]) + vsys  # km/s
Mlim = [0, 10]  # M_sun; to exclude unreasonable points
xlim_plot = [200.0 / 20.0, 200.0]  # au; [inlim, outlim]
vlim_plot = [6.0 / 20.0, 6.0]  # km/s
use_velocity = True  # cuts along the velocity direction
use_position = True  # cuts along the positional direction
include_vsys = False  # False fixes the offset from input vsys at zero.
include_dp = True  # False means a single power
include_pin = False  # False means pin=0.5 (Keplerian).
fixed_pin = 0.5  # Fixed pin when include_pin is False.
fixed_dp = 0.0  # Fixed dp when include_dp is False.
show_pv = True  # figures will be made regardless of this option.
vsys_mode = 'input'  # Plot center: 'input', 'edge', 'ridge', or 'mean'.
show_corner = True  # Display corner plots.
save_corner = True  # Save corner plots independently of display.
minabserr = 0  # minimum absolute errorbar in the unit of bmaj or dv.
minrelerr = 0  # minimum relative errorbar.
calc_evidence = False  # If calculate Bayesian evidence or not.
best = 'pmid'  # 'popt': maximum probability; 'pmid': posterior medians.
'------------------------'


'-------- HOW TO DO EACH STEP --------'
impv = PVAnalysis(fitsfile, rms, vsys, dist, incl=incl, pa=None)
impv.get_edgeridge(outname,
                   thr=thr,
                   ridgemode=ridgemode,
                   use_position=use_position,
                   use_velocity=use_velocity,
                   Mlim=Mlim,
                   xlim=np.array(xlim) / dist,
                   vlim=vlim,
                   minabserr=minabserr,
                   minrelerr=minrelerr,
                   nanbeforemax=True,
                   nanopposite=True,
                   nanbeforecross=True,
                   methods=methods,
                   )
impv.write_edgeridge(outname=outname, methods=methods)
fit_result = impv.fit_edgeridge(methods=methods,
                                include_vsys=include_vsys,
                                include_dp=include_dp,
                                include_pin=include_pin,
                                fixed_pin=fixed_pin,
                                fixed_dp=fixed_dp,
                                outname=outname,
                                rangelevel=0.8,
                                show_corner=show_corner,
                                save_corner=save_corner,
                                kwargs_emcee_corner={
                                    'nwalkers_per_ndim': 16,
                                    'nburnin': 2000,
                                    'nsteps': 2000,
                                    },
                                calc_evidence=calc_evidence,
                                best=best,
                                )
impv.output_fitresult()
impv.plot_fitresult(vlim=vlim_plot,
                    xlim=xlim_plot,
                    flipaxis=False,
                    clevels=[-9, -6, -3, 3, 6, 9],
                    outname=outname,
                    show=show_pv,
                    logcolor=True,
                    Tbcolor=False,
                    vsys_mode=vsys_mode,
                    kwargs_pcolormesh={'cmap': 'viridis', 'vmin': rms},
                    kwargs_contour={'colors': 'lime'},
                    fmt={'edge': 'v', 'ridge': 'o'},
                    linestyle={'edge': '--', 'ridge': '-'},
                    plotridgepoint=True,
                    plotedgepoint=True,
                    plotridgemodel=True,
                    plotedgemodel=True,
                    )
'-------------------------------------'
