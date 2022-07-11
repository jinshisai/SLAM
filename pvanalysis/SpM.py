import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution as difevo
from scipy.interpolate import interp1d

from pvanalysis.pvfits import Impvfits


rms = 1 / 5
thre = 3
pa = 0
fitsdata = Impvfits('./testfits/answer.fits', pa=pa)
answer = np.squeeze(fitsdata.data)
fitsdata = Impvfits('./testfits/question.fits', pa=pa)
question = np.squeeze(fitsdata.data)
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
i0 = np.argmin(np.abs(xaxis + 0.6))
i1 = np.argmin(np.abs(xaxis - 0.6)) + 1
j0 = np.argmin(np.abs(vaxis - vsys + 4))
j1 = np.argmin(np.abs(vaxis - vsys - 4)) + 1
xaxis, vaxis = xaxis[i0:i1], vaxis[j0:j1] - vsys
question, answer = question[j0:j1, i0:i1], answer[j0:j1, i0:i1]
gbeam = np.exp2(-4 * (xaxis / res_off)**2)
beamarea = np.sum(gbeam)
cmpthre = 3 * rms / beamarea

ngroup = 10
def chisq(p, obs, l1, l2, l3, i_cve, mode=''):
    obscut = np.where(obs > 0, obs, obs * 0)
    obsarea = np.sum(obscut)
    if obsarea == 0:
        return 1000
    l1_org = obsarea / beamarea
    l2_org = np.sum(np.abs(obs[:-1] - obs[1:])**2) / beamarea**2
    m = xaxis - np.average(xaxis, weights=obscut)
    l3_org = np.average(m**2, weights=obscut) - beampix**2
    l3_org = l3_org.clip(1, None)
    c0 = (np.convolve(p, gbeam, mode='same') - obscut) / rms
    nmax = len(obs) // ngroup
    if mode == 'pre':
        r = list(range(ngroup))
        r.remove(i_cve)
        c0 = np.mean([np.abs(c0[j:nmax*ngroup:ngroup])**2 for j in r])             / (ngroup - 1) / nmax
    elif mode == 'cve':
        c0 = np.mean(np.abs(c0[i_cve:nmax*ngroup:ngroup])**2)
    else:
        c0 = np.mean(np.abs(c0)**2)
    c1 = l1 * np.sum(np.abs(p)) / l1_org
    c2 = l2 * np.sum(np.abs(p[:-1] - p[1:])**2) / l2_org
    m = xaxis - np.average(xaxis, weights=obscut)
    c3 = np.average(m**2, weights=obscut)
    c3 = l3 * c3.clip(1e-10, None) / l3_org
    c = (c0 + c1 + c2 + c3) / (1 + l1 + l2 + l3)
    return c

a = np.max(question) * 2
b = np.min(question) * 2 * 0
bounds = [[b, a]] * len(xaxis)
        
deconv = []
l1, l2, l3 = 0.5, 0.01, 0.6 * 0
#l1, l2, l3 = 0, 0.1, 0.6
popsize = 15
#print(len(xaxis), 'pixels')
for v, y, p in zip(vaxis, question, answer):
    #print(f'{v:.3f} km/s')
    if np.all(y < 0):
        deconv.append(y * 0)
        continue
    init = [y / beamarea] * popsize
    res = difevo(chisq, bounds=bounds, args=[y, l1, l2, l3, 0, ''],
                 maxiter=int(1e6), tol=1e-6, popsize=popsize,
                 init=init, workers=1) #, updating='deferred')
    c = chisq(p, y, l1, l2, l3, 0, '')
    #print(f'{res.fun:.3f} {res.fun / c:.3f}')
    deconv.append(res.x)
deconv = np.array(deconv)
conv = np.array([np.convolve(d, gbeam, mode='same') for d in deconv])

xaxis = xaxis / res_off
'''
levels = np.arange(1, 10) * 3 * rms
levels = np.sort(np.r_[-levels, levels])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.pcolormesh(xaxis, vaxis, question - conv, shading='nearest',
              cmap='jet', vmin=-3 * rms, vmax=3 * rms)
ax.contour(xaxis, vaxis, question, colors='k',
           levels=levels, linewidths=1)
#ax.contour(xaxis, vaxis, conv, colors='r',
#           levels=levels, linewidths=0.8)
ax.set_xlabel('Position (beam)')
ax.set_ylabel('Velocity (km/s)')
fig.savefig('qvsc_SpM.png')
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.pcolormesh(xaxis, vaxis, deconv, shading='nearest',
              cmap='jet', vmin=0, vmax=0.68)
#ax.contour(xaxis, vaxis, question, colors='b',
#           levels=levels, linewidths=1.2)
levels = 2**np.arange(10) * cmpthre
levels = np.sort(np.r_[-levels, levels])
ax.contour(xaxis, vaxis, answer, colors='k',
           levels=levels, linewidths=1.2)
ax.contour(xaxis, vaxis, deconv, colors='r',
           levels=levels, linewidths=1.2)
ax.plot(vaxis * 0 - 1, vaxis, '--k')
ax.set_xlabel('Position (beam)')
ax.set_ylabel('Velocity (km/s)')
fig.savefig('avsd_SpM.png')
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
m = ax.pcolormesh(xaxis, vaxis, (deconv - answer) / answer,
                  shading='nearest', cmap='jet',
                  vmin=-0.5, vmax=0.5,
                  )
fig.colorbar(m, ax=ax)
ax.plot(vaxis * 0 - 1, vaxis, '--k')
ax.set_xlabel('Position (beam)')
ax.set_ylabel('Velocity (km/s)')
#plt.show()
'''

xnew = np.linspace(xaxis[0], xaxis[-1], (len(xaxis) - 1) * 10 + 1)
result = []
for a, d, c, v in zip(answer, deconv, question, vaxis):
    wa = wd = wc = 0
    wa = xaxis[(a > 0) * (xaxis < 1.)]
    wa = np.nan if len(wa) == 0 else wa[-1]
    ynew = interp1d(xaxis, d, kind='cubic')(xnew)
    g = ynew[:-1] - ynew[1:]
    g[xnew[:-1] > 1.] = 0
    wd = xnew[np.argmax(g)]
    #wd = xnew[(ynew > cmpthre) * (xnew < 1.5)]
    #wd = np.nan if len(wd) == 0 else wd[-1]
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
with open('/Users/yusukeaso/Desktop/rec_SpM.dat', 'a') as f:
    np.savetxt(f, result)
#for r in result:
#    print(f'{r[0]:.2f} {r[1]:.2f} {r[2]:.2f}')