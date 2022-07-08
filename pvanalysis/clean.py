import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution as difevo

from pvanalysis.pvfits import Impvfits


rms = 1 / 10.
thre = 2.0
pa = 0
fitsdata = Impvfits('./testfits/answer.fits', pa=pa)
answer = np.squeeze(fitsdata.data)
fitsdata = Impvfits('./testfits/question.fits', pa=pa)
question = np.squeeze(fitsdata.data)
bmaj, bmin, bpa = fitsdata.beam
phi = np.arctan(bmin / bmaj)
dpa = np.abs(np.radians(bpa - pa))
r = np.cos(2 * phi) * np.sin(2 * dpa) / 2.
print(f'cross term ratio: {r:.2f}')
xaxis = fitsdata.xaxis
vaxis = fitsdata.vaxis
res_off = fitsdata.res_off
print(f'res_off = {res_off / (xaxis[1] - xaxis[0]):.1f} pixels')

i0 = np.argmin(np.abs(xaxis + 0.6))
i1 = np.argmin(np.abs(xaxis - 0.6)) + 1
j0 = np.argmin(np.abs(vaxis - 6.4 + 3.5))
j1 = np.argmin(np.abs(vaxis - 6.4 - 3.5)) + 1
xaxis, vaxis = xaxis[i0:i1], vaxis[j0:j1]
question, answer = question[j0:j1, i0:i1], answer[j0:j1, i0:i1]
gbeam = np.exp2(-4 * (xaxis / res_off)**2)
beamarea = np.sum(gbeam)
beampix = res_off / (xaxis[1] - xaxis[0])
icent = np.argmax(gbeam)

def clean(obs, threshold, gain=0.1):
    residual = obs.copy()
    clncmp = np.zeros_like(residual)
    prevpeak = 1000
    while (peak := np.max(residual)) > threshold \
          and peak < prevpeak:
        prevpeak = peak
        imax = np.argmax(residual)
        cctmp = np.zeros_like(residual)
        cctmp[imax] = peak * gain
        #conv = np.convolve(cctmp, gbeam, mode='same')
        conv = np.roll(gbeam, imax - icent) * peak * gain
        residual -= conv
        clncmp += cctmp
    return clncmp, residual

print(len(xaxis), 'pixels')
deconv, clnres = [], []
for v, y, p in zip(vaxis, question, answer):
    print(f'{v:.3f} km/s')
    clncmp, residual = clean(y, threshold=thre * rms)
    deconv.append(clncmp)
    clnres.append(residual)
deconv = np.array(deconv)
conv = np.array([np.convolve(d, gbeam, mode='same') for d in deconv])
conv += np.array(clnres)
#deconv += np.array(clnres) / beamarea

xaxis = xaxis / res_off
levels = np.arange(3, 100, 1) / 6
levels = np.sort(np.r_[-levels, levels])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.contour(xaxis, vaxis, question, colors='k',
           levels=levels, linewidths=5)
ax.contour(xaxis, vaxis, conv, colors='r',
           levels=levels, linewidths=0.8)
ax.set_xlabel('Position (beam)')
ax.set_ylabel('Velocity (km/s)')
fig.savefig('qvsc.png')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#levels = np.array([1,2,3,4,5,6,7,8,9,10]) * 5e-3 / 5.
#ax.contour(xaxis, vaxis, question, colors='b',
#           levels=levels, linewidths=1.2)
levels = np.arange(1, 11) / 10. * 0.52
ax.contour(xaxis, vaxis, answer, colors='k',
           levels=levels, linewidths=1.2)
ax.contour(xaxis, vaxis, deconv, colors='r',
           levels=levels, linewidths=1.2)
ax.plot(vaxis * 0 + 0.5, vaxis, '--k')
ax.plot(vaxis * 0 - 0.5, vaxis, '--k')
ax.set_xlabel('Position (beam)')
ax.set_ylabel('Velocity (km/s)')
fig.savefig('avsd.png')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
m = ax.pcolormesh(xaxis, vaxis, (deconv - answer) / answer,
                  shading='nearest', cmap='jet',
                  vmin=-0.5, vmax=0.5,
                  )
fig.colorbar(m, ax=ax)
ax.plot(vaxis * 0 + 0.5, vaxis, '--k')
ax.plot(vaxis * 0 - 0.5, vaxis, '--k')
ax.set_xlabel('Position (beam)')
ax.set_ylabel('Velocity (km/s)')
plt.show()

for a, d in zip(answer, deconv):
    wa = np.sum(a) / np.max(a) / beampix
    #md = np.average(xaxis, weights=d) if np.any(d > 0) else 0
    #wd = (np.sqrt(np.average((xaxis - md)**2, weights=d)) / beampix if np.any(d > 0) else 0) * np.sqrt(6)
    wd = np.sum(d) / np.max(d) / beampix
    if not np.isnan(wa):
        print(f'{wa:.2f} {wd:.2f} {wd/wa:.2f}')