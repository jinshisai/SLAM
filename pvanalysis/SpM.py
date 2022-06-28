import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution as difevo

from pvanalysis.pvfits import Impvfits


infile = './testfits/test.fits'
pa = 75
rms = 1.7e-3
fitsdata = Impvfits(infile, pa=pa)
bmaj, bmin, bpa = fitsdata.beam
phi = np.arctan(bmin / bmaj)
dpa = np.abs(np.radians(bpa - pa))
r = np.cos(2*phi) * np.sin(2*dpa) / (1 - np.cos(2*phi) * np.cos(2*dpa))
print(f'cross term ratio: {r:.2f}')
dataorg = np.squeeze(fitsdata.data)
xaxis = fitsdata.xaxis
vaxis = fitsdata.vaxis
res_off = fitsdata.res_off * 3
print(f'res_off = {res_off / (xaxis[1] - xaxis[0]):.1f} pixels')

i0 = np.argmin(np.abs(xaxis + 210 / 140))
i1 = np.argmin(np.abs(xaxis - 210 / 140)) + 1
j0 = np.argmin(np.abs(vaxis - 6.4 + 5))
j1 = np.argmin(np.abs(vaxis - 6.4 - 5)) + 1
xaxis, vaxis, data = xaxis[i0:i1], vaxis[j0:j1], dataorg[j0:j1, i0:i1]
rmsvis = rms * np.sqrt(len(xaxis))
gbeam = np.exp2(-4 * (xaxis / res_off)**2)
gbeam = gbeam / np.sum(gbeam)
#w = np.abs(np.fft.fft(gbeam))
data = np.array([np.convolve(p, gbeam, mode='same') for p in dataorg])

ngroup = 10
def chisq(p, obs, l1, l2, i_cve, mode=''):
    #c0 = (np.fft.fft(p) * w - obs) / rmsvis
    c0 = (np.convolve(p, gbeam, mode='same') - obs) / rms
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
    c = (c0 + c1 + c2) / (1 + l1 + l2)
    return c


max = np.max(np.abs(data)) * 10
bounds = [[-max, max]] * len(xaxis)

deconv = []
l1, l2 = 0.1, 0.1
#ffty = np.fft.fft(y)
#ngrid = 1
#eval = np.empty((ngrid, ngrid))
#lamlist = np.geomspace(1, 10, ngrid)
#for il, jl in np.ndindex((ngrid, ngrid)):
#    print(il, jl)
#    l1, l2 = lamlist[il], lamlist[jl]
print(len(xaxis), 'pixels')
for v, y in zip(vaxis, data):
    print(f'{v:.3f} km/s')
    l1_org = np.sum(np.abs(y))
    l2_org = np.sum(np.abs(y[:-1] - y[1:])**2)
    cvelist = [None] * ngroup
    chilist = [None] * ngroup
    for i_cve in range(ngroup):
        #print('i_cve =', i_cve)
        init = [None] * 15
        for i in range(15):
            noise = np.convolve(np.random.randn(len(y)), gbeam, mode='same')
            noise = noise / np.std(noise) * rms
            init[i] = y + noise
        res = difevo(chisq, bounds=bounds, args=[y, l1, l2, i_cve, 'pre'],
                     init=init, workers=1) #, updating='deferred')
        chi2 = res.fun
        #print(f'root chi2 = {np.sqrt(chi2):.3f} sigma')
        chilist[i_cve] = chi2    
        p = res.x
        cve = chisq(p, y, l1, l2, i_cve, 'cve')
        #print(f'root cve  = {np.sqrt(cve):.3f} sigma')
        cvelist[i_cve] = cve
    meanchi2 = np.mean(chilist)
    meancve = np.mean(cvelist)
    print(f'root mean chi2 = {np.sqrt(meanchi2):.3f} sigma')
    print(f'root mean cve  = {np.sqrt(meancve):.3f} sigma')
    #eval[il, jl] = meancve < meanchi2
    #print(eval)
    for i in range(15):
        noise = np.convolve(np.random.randn(len(y)), gbeam, mode='same')
        noise = noise / np.std(noise) * rms
        init[i] = y + noise
    res = difevo(chisq, bounds=bounds, args=[y, l1, l2, i_cve, ''],
                 init=init, workers=1) #, updating='deferred')
    p = res.x
    deconv.append(p)

#pg = np.array([np.convolve(p, gbeam, mode='same') for p in deconv])
#levels = np.array([-12,-9,-6,-3,3,6,9,12,15,18,21,24,27,30]) * rms
levels = [-10,-8,-6,-4,-2,2,4,6,8,10]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#ax.contour(xaxis, vaxis, data, colors='k', levels=levels)
#ax.contour(xaxis, vaxis, data - pg, colors='r', levels=levels)
ax.pcolormesh(xaxis, vaxis, dataorg - p, vmin=-rms * 2, vmax=rms * 2, cmap='jet')
ax.set_xlabel('Position (arcsec)')
ax.set_ylabel('Velocity (km/s)')
plt.show()
'''        
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(xaxis, y, '-o', label='y')
ax.plot(xaxis, p, '-o', label='p')
ax.plot(xaxis, np.convolve(p, gbeam, mode='same'), '-o', label='p*b')
ax.plot(xaxis, gbeam / gbeam.max() * y.max(), '-o', label='b')
ax.plot(xaxis, xaxis * 0 + rms, '--k')
ax.plot(xaxis, xaxis * 0 - rms, '--k')
ax.plot(xaxis, xaxis * 0)
ax.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#ax.plot(xaxis, y, '-o', label='y')
#ax.plot(xaxis, p, '-o', label='p')
ax.plot(xaxis, np.convolve(p, gbeam, mode='same') - y, '-o', label='p*b')
#ax.plot(xaxis, gbeam / gbeam.max() * y.max(), '-o', label='b')
ax.plot(xaxis, xaxis * 0 + rms, '--k')
ax.plot(xaxis, xaxis * 0 - rms, '--k')
ax.plot(xaxis, xaxis * 0)
ax.legend()
plt.show()
'''