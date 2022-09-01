import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution as difevo
from astropy.io import fits
from astropy import wcs

from pvanalysis.pvfits import Impvfits


rms = 1.4e-3
thre = 3
pa = 113
dist = 139
vsys = 4
xmax = 70
vmax = 6
fitsfile = '/Users/yusukeaso/Desktop/IRAS16253_SBLB_12COh_robust_0.5.pvmajor.fits'
fitsdata = Impvfits(fitsfile, pa=pa)
question = np.squeeze(fitsdata.data)
xaxis = fitsdata.xaxis * dist
vaxis = fitsdata.vaxis - vsys
res_off = fitsdata.res_off * dist
gbeam = np.exp2(-4 * (xaxis / res_off)**2)

i0 = np.argmin(np.abs(xaxis + xmax))
i1 = np.argmin(np.abs(xaxis - xmax)) + 1
j0 = np.argmin(np.abs(vaxis + vmax))
j1 = np.argmin(np.abs(vaxis - vmax)) + 1
xaxis, vaxis = xaxis[i0:i1], vaxis[j0:j1]
question = question[j0:j1, i0:i1]
gbeam = np.exp2(-4 * (xaxis / res_off)**2)
beamarea = np.sum(gbeam)
cmpthre = thre * rms / beamarea

def chisq(p, obs, l1, l2):
    obscut = np.where(obs > 0, obs, obs * 0)
    obsarea = np.sum(obscut)
    if obsarea == 0:
        return 1000
    l1_org = obsarea / beamarea
    l2_org = np.sum(np.abs(obs[:-1] - obs[1:])**2) / beamarea**2
    c0 = (np.convolve(p, gbeam, mode='same') - obscut) / rms
    c0 = np.mean(np.abs(c0)**2)
    c1 = l1 * np.sum(np.abs(p)) / l1_org
    c2 = l2 * np.sum(np.abs(p[:-1] - p[1:])**2) / l2_org
    c = (c0 + c1 + c2) / (1 + l1 + l2)
    return c

bounds = [[0, np.max(question) * 2]] * len(xaxis)

l1, l2 = 0.5, 0.01
popsize = 15

deconv = []
for v, y in zip(vaxis, question):
    if np.all(y < 0):
        deconv.append(y * 0)
        continue
    init = [y / beamarea] * popsize
    res = difevo(chisq, bounds=bounds, args=[y, l1, l2],
                 maxiter=int(1e6), tol=1e-6, popsize=popsize,
                 init=init, workers=1) #, updating='deferred')
    deconv.append(res.x)
deconv = np.array(deconv)
conv = np.array([np.convolve(d, gbeam, mode='same') for d in deconv])
deconv_pad = np.squeeze(fitsdata.data) * 0
deconv_pad[j0:j1, i0:i1] = deconv
###########################################################

w = wcs.WCS(naxis=2)
hdu = fits.PrimaryHDU(deconv_pad, header=w.to_header())
h = fits.open(fitsfile)[0].header
for k in h.keys():
    if not ('COMMENT' in k or 'HISTORY' in k):
        hdu.header[k] = h[k]
hdu.header['BUNIT'] = 'Jy/pixel'
hdu = fits.HDUList([hdu])
hdu.writeto('SpM.fits', overwrite=True)

###########################################################
levels = np.arange(1, 10) * 6 * rms
levels = np.sort(np.r_[-levels, levels])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolormesh(xaxis, vaxis, question, shading='nearest',
                  cmap='viridis', vmin=0)
fig.colorbar(c, ax=ax, label='Jy/beam')
ax.contour(xaxis, vaxis, question, colors='w',
           levels=levels, linewidths=1)
ax.set_xlabel('Position (au)')
ax.set_ylabel('Velocity (km/s)')
ax.set_xlim(-xmax, xmax)
ax.set_ylim(-vmax, vmax)
fig.savefig('q_SpM.png')
#plt.show()

levels = np.arange(1, 10) * 6 * rms
levels = np.sort(np.r_[-levels, levels])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolormesh(xaxis, vaxis, conv, shading='nearest',
                  cmap='viridis', vmin=0)
fig.colorbar(c, ax=ax, label='Jy/beam')
ax.contour(xaxis, vaxis, conv, colors='w',
           levels=levels, linewidths=1)
ax.set_xlabel('Position (au)')
ax.set_ylabel('Velocity (km/s)')
ax.set_xlim(-xmax, xmax)
ax.set_ylim(-vmax, vmax)
fig.savefig('c_SpM.png')
#plt.show()

levels = np.arange(1, 10) * 6 * rms
levels = np.sort(np.r_[-levels, levels])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolormesh(xaxis, vaxis, (question - conv) / rms, shading='nearest',
                  cmap='viridis', vmin=-3, vmax=3)
fig.colorbar(c, ax=ax, label='sigma')
ax.contour(xaxis, vaxis, question - conv, colors='w',
           levels=levels, linewidths=1)
ax.set_xlabel('Position (au)')
ax.set_ylabel('Velocity (km/s)')
ax.set_xlim(-xmax, xmax)
ax.set_ylim(-vmax, vmax)
fig.savefig('qvsc_SpM.png')
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolormesh(xaxis, vaxis, deconv, shading='nearest',
                  cmap='viridis', vmin=0)
fig.colorbar(c, ax=ax, label='Jy/pixel')
levels = np.arange(1, 10) * 3 * cmpthre
levels = np.sort(np.r_[-levels, levels])
ax.contour(xaxis, vaxis, deconv, colors='w',
           levels=levels, linewidths=1)
ax.set_xlabel('Position (au)')
ax.set_ylabel('Velocity (km/s)')
ax.set_xlim(-xmax, xmax)
ax.set_ylim(-vmax, vmax)
fig.savefig('d_SpM.png')
#plt.show()
