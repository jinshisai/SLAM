import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution as difevo
from scipy.interpolate import interp1d

from pvanalysis.pvfits import Impvfits


rms = 1 / 10
thre = 3
pa = 0
fitsdata = Impvfits('./testfits/triangle.fits', pa=pa)
answer = np.squeeze(fitsdata.data)
bmaj, bmin, bpa = fitsdata.beam
phi = np.arctan(bmin / bmaj)
dpa = np.abs(np.radians(bpa - pa))
r = np.cos(2*phi) * np.sin(2*dpa) / 2.
#print(f'cross term ratio: {r:.2f}')
xaxis = fitsdata.xaxis
vaxis = fitsdata.vaxis
dv = vaxis[1] - vaxis[0]
dx = xaxis[1] - xaxis[0]
res_off = fitsdata.res_off
gbeam = np.exp2(-4 * (xaxis / res_off)**2)
beampix = res_off / dx
#print(f'res_off = {beampix:.1f} pixels')

noise = np.random.randn(*np.shape(answer))
noise = np.array([np.convolve(n, gbeam, mode='same') for n in noise])
noise = noise / np.std(noise) * rms
question = np.array([np.convolve(a, gbeam, mode='same') for a in answer])
question = question + noise

vsys = 6.3 - 0.5 * dv
#i0 = np.argmin(np.abs(xaxis + 1.4))
#i1 = np.argmin(np.abs(xaxis - 1.4)) + 1
#j0 = np.argmin(np.abs(vaxis - vsys + 5))
#j1 = np.argmin(np.abs(vaxis - vsys - 5)) + 1
i0 = np.argmin(np.abs(xaxis + 0.6))
i1 = np.argmin(np.abs(xaxis - 0.6)) + 1
j0 = np.argmin(np.abs(vaxis - vsys + 4))
j1 = np.argmin(np.abs(vaxis - vsys - 4)) + 1
xaxis, vaxis = xaxis[i0:i1], vaxis[j0:j1] - vsys
question, answer = question[j0:j1, i0:i1], answer[j0:j1, i0:i1]
gbeam = np.exp2(-4 * (xaxis / res_off)**2)
beamarea = np.sum(gbeam)
cmpthre = thre * rms / beamarea
icent = np.argmax(gbeam)

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

#print(len(xaxis), 'pixels')
deconv, clnres = [], []
for v, y, p in zip(vaxis, question, answer):
    #print(f'{v:.3f} km/s')
    clncmp, residual = clean(y, threshold=1 * rms)
    deconv.append(clncmp)
    clnres.append(residual)
deconv = np.array(deconv)
conv = np.array([np.convolve(d, gbeam, mode='same') for d in deconv])

xaxis = xaxis / res_off
levels = np.arange(1, 10) * 3 * rms
levels = np.sort(np.r_[-levels, levels])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolormesh(xaxis, vaxis, question, shading='nearest',
                  cmap='jet', vmin=0 * rms, vmax=10 * rms)
fig.colorbar(c, ax=ax, label='Jy/beam')
ax.contour(xaxis, vaxis, question, colors='w',
           levels=levels, linewidths=1)
ax.set_xlabel('Position (beam)')
ax.set_ylabel('Velocity (km/s)')
fig.savefig('q_clean.png')
#plt.show()

levels = np.arange(1, 10) * 3 * rms
levels = np.sort(np.r_[-levels, levels])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolormesh(xaxis, vaxis, conv, shading='nearest',
                  cmap='jet', vmin=0 * rms, vmax=10 * rms)
fig.colorbar(c, ax=ax, label='Jy/beam')
ax.contour(xaxis, vaxis, conv, colors='w',
           levels=levels, linewidths=1)
ax.set_xlabel('Position (beam)')
ax.set_ylabel('Velocity (km/s)')
fig.savefig('c_clean.png')
#plt.show()

levels = np.arange(1, 10) * 3 * rms
levels = np.sort(np.r_[-levels, levels])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolormesh(xaxis, vaxis, question - conv, shading='nearest',
                  cmap='jet', vmin=-3 * rms, vmax=3 * rms)
fig.colorbar(c, ax=ax, label='Jy/beam')
ax.contour(xaxis, vaxis, question - conv, colors='w',
           levels=levels, linewidths=1)
ax.set_xlabel('Position (beam)')
ax.set_ylabel('Velocity (km/s)')
fig.savefig('qvsc_clean.png')
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolormesh(xaxis, vaxis, answer, shading='nearest',
                  cmap='jet', vmin=0, vmax=0.68)
fig.colorbar(c, ax=ax, label='Jy/pixel')
levels = 2**np.arange(10) * cmpthre
levels = np.sort(np.r_[-levels, levels])
ax.contour(xaxis, vaxis, answer, colors='w',
           levels=levels, linewidths=1)
ax.plot(vaxis * 0 - 1, vaxis, '--w')
ax.set_xlabel('Position (beam)')
ax.set_ylabel('Velocity (km/s)')
fig.savefig('a_clean.png')
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolormesh(xaxis, vaxis, deconv, shading='nearest',
                  cmap='jet', vmin=0, vmax=0.68)
fig.colorbar(c, ax=ax, label='Jy/pixel')
levels = 2**np.arange(10) * cmpthre
levels = np.sort(np.r_[-levels, levels])
ax.contour(xaxis, vaxis, deconv, colors='w',
           levels=levels, linewidths=1)
ax.plot(vaxis * 0 - 1, vaxis, '--w')
ax.set_xlabel('Position (beam)')
ax.set_ylabel('Velocity (km/s)')
fig.savefig('d_clean.png')
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolormesh(xaxis, vaxis, answer - deconv, shading='nearest',
                  cmap='jet', vmin=-0.68, vmax=0.68)
fig.colorbar(c, ax=ax, label='Jy/pixel')
levels = 2**np.arange(10) * cmpthre
levels = np.sort(np.r_[-levels, levels])
ax.contour(xaxis, vaxis, answer - deconv, colors='w',
           levels=levels, linewidths=1)
ax.plot(vaxis * 0 - 1, vaxis, '--w')
ax.set_xlabel('Position (beam)')
ax.set_ylabel('Velocity (km/s)')
fig.savefig('avsd_clean.png')
#plt.show()

xnew = np.linspace(xaxis[0], xaxis[-1], (len(xaxis) - 1) * 10 + 1)
result = []
for a, d, c, v in zip(answer, deconv, question, vaxis):
    wa = wd = wc = 0
    wa = xaxis[(a > 0) * (xaxis < 1.)]
    wa = np.nan if len(wa) == 0 else wa[-1]
    d[(np.roll(d, 1) == 0) * (np.roll(d, -1) == 0)] == 0
    ynew = interp1d(xaxis, d, kind='linear')(xnew)
    g = ynew[:-1] - ynew[1:]
    g[xnew[:-1] > 1.0] = 0
    #wd = xnew[np.argmax(g)]
    wd = xnew[(ynew > cmpthre) * (xnew < 1.)]
    wd = np.nan if len(wd) == 0 else wd[-1]
    ynew = interp1d(xaxis, c, kind='cubic')(xnew)
    wc = xnew[(ynew > rms * thre) * (xnew < 1.)]
    wc = np.nan if len(wc) == 0 else wc[-1]
    w = 3**(1 - (np.abs(v / dv) - 0.5) / 4.)
    w = 0 if w < 0.32 else w
    if w > 0:
        r = (wc - wd) / w
        result.append([int(round(1 / rms)), w, wc, wd])
result = np.array(result)
result = np.concatenate((result[:9], result[-1:-10:-1]), axis=0)
with open('/Users/yusukeaso/Desktop/rec_clean.dat', 'a') as f:
    np.savetxt(f, result)
#for r in result:
#    print(f'{r[0]:.2f} {r[1]:.2f} {r[2]:.2f}')