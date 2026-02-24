import numpy as np
from pvanalysis import PVAnalysis

'-------- INPUTS --------'
fitsfile = './testfits/test.pvanalysis.linear.fits'
outname = 'linear'  # file name header for outputs
incl = 65.  # deg
vsys = 4.0  # km/s
dist = 139.  # pc
rms = 2.4e-3  # Jy/beam
thr = 5.  # rms
ridgemode = 'mean'  # 'mean' or 'gauss'
xlim = [-70, 0, 0, 70]  # au; [-outlim, -inlim, inlim, outlim]
vlim = np.array([-1.4, 0, 0, 1.4]) + vsys  # km/s
Mlim = [0, 10]  # M_sun; to exclude unreasonable points
xlim_plot = [100. / 20., 100.]  # au; [inlim, outlim]
vlim_plot = [3. / 20., 3.]  # km/s
use_velocity = True  # cuts along the velocity direction
use_position = True  # cuts along the positional direction
include_intercept = True  # False means v(x=0) is fixed at 0.
show_pv = True  # figures will be made regardless of this option.
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
impv.fit_linear(include_intercept=include_intercept)
impv.plot_fitresult(vlim=vlim_plot, xlim=xlim_plot,
                    clevels=[-9,-6,-3,3,6,9], outname=outname,
                    show=show_pv, logcolor=True, Tbcolor=False,
                    kwargs_pcolormesh={'cmap':'viridis'},
                    kwargs_contour={'colors':'lime'},
                    fmt={'edge':'v', 'ridge':'o'},
                    linestyle={'edge':'--', 'ridge':'-'},
                    plotridgepoint=True, plotedgepoint=False,
                    plotridgemodel=True, plotedgemodel=True)
'-------------------------------------'