import numpy as np
from pvanalysis.pvfits import Impvfits


infile = '../testfits/test.fits'
pa = 75
fitsdata = Impvfits(infile, pa=pa)
data = fitsdata.data
xaxis = fitsdata.xaxis
vaxis = fitsdata.vaxis
res_off = fitsdata.res_off
delv = fitsdata.delv
hob = int(np.round((res_off*0.5/fitsdata.delx)))
