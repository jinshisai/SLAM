from plotastrodata.plot_utils import PlotAstroData

h = 'IRAS16253_SBLB_C18O_robust_2.0.imsub'
p = PlotAstroData(fitsimage=h+'.fits', rmax=1, vmin=-2.5, vmax=2.5, vsys=4, vskip=1, ncols=7, nrows=5)
p.add_contour(fitsimage=h+'.fits', sigma=2e-3, levels=[-9,-6,-3,3,6,9,12,15], colors='k', vsys=4)
p.add_contour(fitsimage=h+'.model.fits', sigma=2e-3, levels=[-9,-6,-3,3,6,9,12,15], colors='r', vsys=4)
p.add_line(poslist=[[0.5, 0.5]] * 4, anglelist=[113, 113+90, 113+180, 113+270], rlist=[1.5] * 4)
p.set_axis()
p.savefig(h+'.model.png', show=True)

p = PlotAstroData(fitsimage=h+'.fits', rmax=1, vmin=-2.5, vmax=2.5, vsys=4, vskip=1, ncols=7, nrows=5)
p.add_contour(fitsimage=h+'.residual.fits', sigma=2e-3, levels=[-9,-6,-3,3,6,9,12,15], colors='k', vsys=4)
p.add_line(poslist=[[0.5, 0.5]] * 4, anglelist=[113, 113+90, 113+180, 113+270], rlist=[1.5] * 4)
p.set_axis()
p.savefig(h+'.residual.png', show=True)