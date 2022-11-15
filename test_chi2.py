
import numpy as np
from pvanalysis import PVAnalysis
from pvanalysis.analysis_tools import (doublepower_r,
                                       doublepower_v,
                                       doublepower_v_error,
                                       doublepower_r_error)

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
xlim_plot = [400. / 20., 400.]  # au; [inlim, outlim]
vlim_plot = [6. / 20., 6.]  # km/s
use_velocity = True  # cuts along the velocity direction
use_position = True  # cuts along the positional direction
include_vsys = False  # vsys offset. False means vsys=0.
include_dp = True  # False means a single power
include_pin = False  # False means pin=0.5 (Keplerian).
show_pv = True  # figures will be made regardless of this option.
show_corner = True  # figures will be made regardless of this option.
minabserr = 0.1  # minimum absolute errorbar in the unit of bmaj or dv.
minrelerr = 0.01  # minimum relative errorbar.
'------------------------'



'-------- HOW TO DO EACH STEP --------'
impv = PVAnalysis(fitsfile, rms, vsys, dist, pa=None)
impv.get_edgeridge(outname, thr=thr, ridgemode=ridgemode, incl=incl,
                   use_position=use_position, use_velocity=use_velocity,
                   Mlim=Mlim, xlim=np.array(xlim) / dist, vlim=vlim,
                   minabserr=minabserr, minrelerr=minrelerr,
                   nanbeforemax=True, nanopposite=True, nanbeforecross=True)
impv.write_edgeridge(outname=outname)
impv.fit_edgeridge(include_vsys=include_vsys,
                   include_dp=include_dp,
                   include_pin=include_pin,
                   outname=outname, show_corner=show_corner)
'-------------------------------------'



include = [True, include_dp, include_pin, include_dp, include_vsys]
minabs = lambda a, i, j: np.min(np.abs(np.r_[a[i], a[j]]))
maxabs = lambda a, i, j: np.max(np.abs(np.r_[a[i], a[j]]))
for args, ext, in zip([impv._PVAnalysis__Es, impv._PVAnalysis__Rs], 
    ['edge', 'ridge'], ):
    popt, popt_err = impv.popt[ext]
    popt = popt[include]
    plim = np.array([
        [minabs(args, 1, 3), minabs(args, 0, 4), 0.01, 0, -1],
        [maxabs(args, 1, 3), maxabs(args, 0, 4), 10.0, 10, 1]])
    q0 = np.array([0, np.sqrt(plim[0][1] * plim[1][1]), 0.5, 0, 0])
    q0 = np.where(include, np.nan, q0)
    def wpow_r_custom(v, *p):
        (q := q0 * 1)[np.isnan(q0)] = p
        return doublepower_r(v, *q)
    def wpow_v_custom(r, *p):
        (q := q0 * 1)[np.isnan(q0)] = p
        return doublepower_v(r, *q)
    def lnprob(p, v0, x1, dx1, x0, v1, dv1):
        chi2 = np.sum(((x1 - wpow_r_custom(v0, *p)) / dx1)**2) \
               + np.sum(((v1 - wpow_v_custom(x0, *p)) / dv1)**2)
        return -0.5 * chi2
    def reduced_chi2(p, v0, x1, dx1, x0, v1, dv1):
        chi2 = np.sum(((x1 - wpow_r_custom(v0, *p)) / dx1)**2) \
        + np.sum(((v1 - wpow_v_custom(x0, *p)) / dv1)**2)
        print(len(p))
        return chi2/(len(v0) - len(p) -1 )
    logp = reduced_chi2(popt, *args)
    print(logp)