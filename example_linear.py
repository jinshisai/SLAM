import numpy as np
from pvanalysis import PVAnalysis

'-------- INPUTS --------'
fitsfile = './testfits/testlinear.fits'
outname = 'linear'  # file name header for outputs
incl = 65.  # deg
vsys = 4.0  # km/s
dist = 139.  # pc
rms = 2.4e-3  # Jy/beam
thr = 5.  # rms
ridgemode = 'mean'  # 'mean' or 'gauss'
xlim = [-50, 0, 0, 50]  # au; [-outlim, -inlim, inlim, outlim]
vlim = np.array([-1, 0, 0, 1]) + vsys  # km/s
Mlim = [0, 10]  # M_sun; to exclude unreasonable points
xlim_plot = [200. / 20., 200.]  # au; [inlim, outlim]
vlim_plot = [6. / 20., 6.]  # km/s
use_velocity = True  # cuts along the velocity direction
use_position = False  # cuts along the positional direction
include_vsys = False  # vsys offset. False means vsys=0.
include_dp = True  # False means a single power
include_pin = False  # False means pin=0.5 (Keplerian).
show_pv = True  # figures will be made regardless of this option.
show_corner = False  # figures will be made regardless of this option.
minabserr = 0.1  # minimum absolute errorbar in the unit of bmaj or dv.
minrelerr = 0.01  # minimum relative errorbar.
'------------------------'



'-------- HOW TO DO EACH STEP --------'
impv = PVAnalysis(fitsfile, rms, vsys, dist, pa=None)
impv.get_edgeridge(outname, thr=thr, ridgemode=ridgemode, incl=incl,
                   use_position=use_position, use_velocity=use_velocity,
                   Mlim=Mlim, xlim=np.array(xlim) / dist, vlim=vlim,
                   minabserr=minabserr, minrelerr=minrelerr,
                   nanbeforemax=False, nanopposite=False, nanbeforecross=False)
impv.write_edgeridge(outname=outname)
impv.fit_linear(include_intercept=True)
#impv.output_fitresult()
#impv.plot_fitresult(vlim=vlim_plot, xlim=xlim_plot,
#                    clevels=[-9,-6,-3,3,6,9], outname=outname,
#                    show=show_pv, logcolor=True, Tbcolor=False,
#                    kwargs_pcolormesh={'cmap':'viridis'},
#                    kwargs_contour={'colors':'lime'},
#                    fmt={'edge':'v', 'ridge':'o'},
#                    linestyle={'edge':'--', 'ridge':'-'},
#                    plotridgepoint=True, plotedgepoint=False,
#                    plotridgemodel=True, plotedgemodel=False)
'-------------------------------------'