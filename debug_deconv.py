import numpy as np
import matplotlib.pyplot as plt
from pvanalysis import PVAnalysis
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
show_pv = True  # figures will be made regardless of this option.
show_corner = True  # figures will be made regardless of this option.
minabserr = 0.1  # minimum absolute errorbar in the unit of bmaj or dv.
minrelerr = 0.01  # minimum relative errorbar.
'------------------------'



'-------- HOW TO DO EACH STEP --------'
impv = PVAnalysis(fitsfile, rms, vsys, dist, pa=None)
#print(impv.fitsdata.naxis, impv.fitsdata.data.shape, impv.fitsdata.beam)
#impv.fitsdata.beam_deconvolution(sigmacut=rms*3., highfcut=2., solmode='nullcut') # highfcut=5.
impv.fitsdata.beam_deconvolution(sigmacut=rms*3., highfcut=2., solmode='gauss') # highfcut=5.
#impv.fitsdata.beam_deconvolution(highfcut=1.) # highfcut=5.

#plt.imshow(impv.fitsdata.data_deconv[0,:,:], origin='lower')
#plt.show()

xx, vv = np.meshgrid(impv.fitsdata.xaxis, impv.fitsdata.vaxis)
fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
for ax, data in zip(axes, [impv.fitsdata.data, impv.fitsdata.data_deconv]):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.05)

    pltcol  = ax.pcolor(xx, vv, data[0,:,:], shading='auto')
    pltcont = ax.contour(xx, vv, data[0,:,:], levels=np.array([-6,-3,3,6,9,12,15])*rms)
    fig.colorbar(pltcol, cax=cax)
    ax.set_aspect(1.)

    fig.subplots_adjust(wspace=0.4)

plt.show()
'''


impv.get_edgeridge(outname, thr=thr, ridgemode=ridgemode, incl=incl,
                   use_position=use_position, use_velocity=use_velocity,
                   Mlim=Mlim, xlim=np.array(xlim) / dist, vlim=vlim,
                   minabserr=minabserr, minrelerr=minrelerr,
                   nanbeforemax=True, nanopposite=True, nanbeforecross=True,
                   deconvolution=True)
impv.write_edgeridge(outname=outname)
impv.fit_edgeridge(include_vsys=include_vsys,
                   include_dp=include_dp,
                   include_pin=include_pin,
                   outname=outname, show_corner=show_corner)
impv.output_fitresult()
impv.fitsdata.data = impv.fitsdata.data_deconv
impv.plot_fitresult(vlim=vlim_plot, xlim=xlim_plot,
                    clevels=[-9,-6,-3,3,6,9], outname=outname,
                    show=show_pv, logcolor=True, Tbcolor=False,
                    kwargs_pcolormesh={'cmap':'viridis'},
                    kwargs_contour={'colors':'lime'},
                    fmt={'edge':'v', 'ridge':'o'},
                    linestyle={'edge':'--', 'ridge':'-'},
                    plotridgepoint=True, plotedgepoint=True,
                    plotridgemodel=True, plotedgemodel=True)
'''
'-------------------------------------'