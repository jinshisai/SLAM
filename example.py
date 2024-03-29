import numpy as np
from pvanalysis import PVAnalysis

'-------- INPUTS --------'
fitsfile = './testfits/test.fits'
outname = 'pvanalysis'  # file name header for outputs
incl = 48.  # deg
vsys = 6.4  # km/s
dist = 140.  # pc
rms = 1.7e-3  # Jy/beam
thr = 5.  # rms
ridgemode = 'mean'  # 'mean' or 'gauss'
xlim = [-200, 0, 0, 200]  # au; [-outlim, -inlim, inlim, outlim]
vlim = np.array([-5, 0, 0, 5]) + vsys  # km/s
Mlim = [0, 10]  # M_sun; to exclude unreasonable points
xlim_plot = [200. / 20., 200.]  # au; [inlim, outlim]
vlim_plot = [6. / 20., 6.]  # km/s
use_velocity = True  # cuts along the velocity direction
use_position = True  # cuts along the positional direction
include_vsys = False  # vsys offset. False means vsys=0.
include_dp = True  # False means a single power
include_pin = False  # False means pin=0.5 (Keplerian).
fixed_pin = 0.5  # Fixed pin when include_pin is False.
fixed_dp = 0.0  # Fixed dp when include_dp is False.
show_pv = True  # figures will be made regardless of this option.
show_corner = True  # figures will be made regardless of this option.
minabserr = 0.1  # minimum absolute errorbar in the unit of bmaj or dv.
minrelerr = 0.01  # minimum relative errorbar.
calc_evidence = False # If calculate Bayesian evidence or not.
'------------------------'



'-------- HOW TO DO EACH STEP --------'
impv = PVAnalysis(fitsfile, rms, vsys, dist, incl=incl, pa=None)
impv.get_edgeridge(outname, thr=thr, ridgemode=ridgemode,
                   use_position=use_position, use_velocity=use_velocity,
                   Mlim=Mlim, xlim=np.array(xlim) / dist, vlim=vlim,
                   minabserr=minabserr, minrelerr=minrelerr,
                   nanbeforemax=True, nanopposite=True, nanbeforecross=True)
impv.write_edgeridge(outname=outname)
impv.fit_edgeridge(include_vsys=include_vsys,
                   include_dp=include_dp,
                   include_pin=include_pin,
                   fixed_pin=fixed_pin, fixed_dp=fixed_dp,
                   outname=outname, rangelevel=0.8,
                   show_corner=show_corner,
                   calc_evidence=calc_evidence)
impv.output_fitresult()
impv.plot_fitresult(vlim=vlim_plot, xlim=xlim_plot, flipaxis=False,
                    clevels=[-9,-6,-3,3,6,9], outname=outname,
                    show=show_pv, logcolor=True, Tbcolor=False,
                    kwargs_pcolormesh={'cmap':'viridis'},
                    kwargs_contour={'colors':'lime'},
                    fmt={'edge':'v', 'ridge':'o'},
                    linestyle={'edge':'--', 'ridge':'-'},
                    plotridgepoint=True, plotedgepoint=True,
                    plotridgemodel=True, plotedgemodel=True)
'-------------------------------------'