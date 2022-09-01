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

i0 = np.argmin(np.abs(xaxis + xmax))
i1 = np.argmin(np.abs(xaxis - xmax)) + 1
j0 = np.argmin(np.abs(vaxis + vmax))
j1 = np.argmin(np.abs(vaxis - vmax)) + 1
xaxis, vaxis = xaxis[i0:i1], vaxis[j0:j1]
question = question[j0:j1, i0:i1]
gbeam = np.exp2(-4 * (xaxis / res_off)**2)
beamarea = np.sum(gbeam)
cmpthre = thre * rms / beamarea

def clean(obs, threshold, gain=0.01):
    residual = obs.copy()
    clncmp = np.zeros_like(residual)
    prevpeak = 1000
    mask = (xaxis > 1.5 * res_off) * 0
    maskedres = residual - mask
    while (peak := np.max(maskedres)) > threshold \
          and peak < prevpeak:
        prevpeak = peak
        imax = np.argmax(maskedres)
        cctmp = np.zeros_like(residual)
        cctmp[imax] = peak * gain
        conv = np.convolve(cctmp, gbeam, mode='same')
        residual -= conv
        clncmp += cctmp
        maskedres = residual - mask
    #clncmp[(np.roll(clncmp, 1) < cmpthre) 
    #       * (np.roll(clncmp, -1) < cmpthre)] = 0
    return clncmp, residual

deconv = []
for v, y in zip(vaxis, question):
    clncmp, _ = clean(y, threshold=1 * rms)
    deconv.append(clncmp)
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
hdu.writeto('clean.fits', overwrite=True)

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
fig.savefig('q_clean.png')
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
fig.savefig('c_clean.png')
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
fig.savefig('qvsc_clean.png')
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
fig.savefig('d_clean.png')
#plt.show()
