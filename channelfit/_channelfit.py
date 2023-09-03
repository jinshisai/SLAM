# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Yusuke Aso
# Created Date: 2022 Jan 27
# version = alpha
# ---------------------------------------------------------------------------
"""
This script makes model channel maps from the observed mom0 by assuming 2D velocity pattern.
The main class ChannelFit can be imported to do each steps separately.

Note. FITS files with multiple beams are not supported. The dynamic range for xlim_plot and vlim_plot should be >10 for nice tick labels.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import constants, units, wcs
from astropy.coordinates import SkyCoord
from scipy.signal import fftconvolve
import warnings
from tqdm import tqdm
from utils import emcee_corner

warnings.simplefilter('ignore', RuntimeWarning)

GG = constants.G.si.value
M_sun = constants.M_sun.si.value
au = units.au.to('m')
vunit = np.sqrt(GG * M_sun / au) * 1e-3

def rot(x, y, pa):
    s = x * np.cos(pa) - y * np.sin(pa)  # along minor axis
    t = x * np.sin(pa) + y * np.cos(pa)  # along major axis
    return np.array([s, t])

def irot(s, t, pa):
    x =  s * np.cos(pa) + t * np.sin(pa)  # along R.A. axis
    y = -s * np.sin(pa) + t * np.cos(pa)  # along Dec. axis
    return np.array([x, y])

def boxgauss(v_over_cs: np.ndarray, dv_over_cs: float) -> np.ndarray:
    clipsigma = 3. + dv_over_cs
    dv = min([2, dv_over_cs]) / 10.
    ndv = 2 * int(dv_over_cs / 2. / dv + 0.5) + 1
    n = 2 * int(clipsigma / dv + 0.5) + 1
    v = np.linspace(-clipsigma, clipsigma, n)
    g = np.exp(-0.5 * v**2)
    #g /= np.sum(g)
    p = np.sum([g[i:i + n - ndv + 1] for i in range(ndv)], axis=0)
    #p /= ndv
    n = n - ndv
    n0 = n // 2
    iv = (np.round(v_over_cs / dv) + n0).astype('int').clip(0, n)
    p = np.where((iv == 0) | (iv == n), 0, p[iv])
    return p
    
def makemom01(d: np.ndarray, v: np.ndarray, sigma: float) -> dict:
    dmasked = d * 1
    dmasked[np.isnan(dmasked)] = 0
    dv = np.min(v[1:] - v[:-1])
    mom0 = np.sum(d, axis=0) * dv
    sigma_mom0 = sigma * dv * np.sqrt(len(d))
    vv = np.moveaxis([[v]], 2, 0)
    dmasked[dmasked < 3 * sigma] = 0
    mom1 = np.sum(d * vv, axis=0) / np.sum(d, axis=0)
    mom2 = np.sqrt(np.sum(d * (vv - mom1)**2, axis=0), np.sum(d, axis=0))
    mom1[mom0 < 3 * sigma_mom0] = np.nan
    return {'mom0':mom0, 'mom1':mom1, 'mom2':mom2, 'sigma_mom0':sigma_mom0}
    

class ChannelFit():

#    def __init__(self):

    def read_cubefits(self, cubefits: str, center: str = None,
                      dist: float = 1, vsys: float = 0,
                      xmax: float = 1e4, ymax: float = 1e4,
                      vmin: float = -100, vmax: float = 100,
                      sigma: float = None) -> dict:
        """
        Read a position-velocity diagram in the FITS format.

        Parameters
        ----------
        cubefits : str
            Name of the input FITS file including the extension.
        center : str
            Coordinates of the target: e.g., "01h23m45.6s 01d23m45.6s".
        dist : float
            Distance of the target, used to convert arcsec to au.
        vsys : float
            Systemic velocity of the target.
        xmax : float
            The R.A. axis is limited to (-xmax, xmax) in the unit of au.
        ymax : float
            The Dec. axis is limited to (-xmax, xmax) in the unit of au.
        vmax : float
            The velocity axis of the PV diagram is limited to (vmin, vmax).
        vmin : float
            The velocity axis of the PV diagram is limited to (vmin, vmax).
        sigma : float
            Standard deviation of the FITS data. None means automatic.

        Returns
        ----------
        fitsdata : dict
            x (1D array), v (1D array), data (2D array), header, and sigma.
        """
        cc = constants.c.si.value
        f = fits.open(cubefits)[0]
        d, h = np.squeeze(f.data), f.header
        if center is None:
            cx, cy = 0, 0
        else:
            coord = SkyCoord(center, frame='icrs')
            cx = coord.ra.degree - h['CRVAL1']
            cy = coord.dec.degree - h['CRVAL2']
        if sigma is None:
            sigma = np.mean([np.nanstd(d[:2]), np.std(d[-2:])])
            print(f'sigma = {sigma:.3e}')
        x = (np.arange(h['NAXIS1']) - h['CRPIX1'] + 1) * h['CDELT1']
        y = (np.arange(h['NAXIS2']) - h['CRPIX2'] + 1) * h['CDELT2']
        v = (np.arange(h['NAXIS3']) - h['CRPIX3'] + 1) * h['CDELT3']
        v = v + h['CRVAL3']
        x = (x - cx) * 3600. * dist  # au
        y = (y - cy) * 3600. * dist  # au
        v = (1. - v / h['RESTFRQ']) * cc / 1.e3 - vsys  # km/s
        i0, i1 = np.argmin(np.abs(x - xmax)), np.argmin(np.abs(x + xmax))
        j0, j1 = np.argmin(np.abs(y + ymax)), np.argmin(np.abs(y - ymax))
        k0, k1 = np.argmin(np.abs(v - vmin)), np.argmin(np.abs(v - vmax))
        self.offpix = (i0, j0, k0)
        #x, y, v = x[i0:i1 + 1], y[j0:j1 + 1], v[k0:k1 + 1]
        x, y, v = x[i0:i1 + 1], y[j0:j1 + 1], v[:]
        #d =  d[k0:k1 + 1, j0:j1 + 1, i0:i1 + 1]
        d =  d[:, j0:j1 + 1, i0:i1 + 1]
        dx, dy, dv = x[1] - x[0], y[1] - y[0], v[1] - v[0]
        if 'BMAJ' in h.keys():
            bmaj = h['BMAJ'] * 3600. * dist  # au
            bmin = h['BMIN'] * 3600. * dist  # au
            bpa = h['BPA']  # deg
        else:
            bmaj, bmin, bpa = dy, -dx, 0
            print('No valid beam in the FITS file.')
        self.x, self.dx = x, dx
        self.y, self.dy = y, dy
        self.v, self.dv = v, dv
        self.data, self.header, self.sigma = d, h, sigma
        self.bmaj, self.bmin, self.bpa = bmaj, bmin, bpa
        self.cubefits, self.dist, self.vsys = cubefits, dist, vsys
        return {'x':x, 'y':y, 'v':v, 'data':d, 'header':h, 'sigma':sigma}

    def gridondisk(self, cubefits: str = None,
                   pa: float = 0, incl: float = 90, dist: float = 1,
                   center: str = None, vsys: float = 0,
                   rmax: float = 1e4, vlim: tuple = (-100, 0, 0, 100),
                   sigma: float = None):
        if not (cubefits is None):
            self.read_cubefits(cubefits, center, dist, vsys,
                               rmax, rmax, vlim[0], vlim[3], sigma)
            self.fitsname = cubefits
            v = self.v
        self.X, self.Y = np.meshgrid(self.x, self.y)
        xminor, xmajor = rot(self.X, self.Y, np.radians(pa))
        self.deproj = 1 / np.cos(np.radians(incl))
        xminor = xminor * self.deproj
        self.xminor, self.xmajor = xminor, xmajor
        
        self.v_nanblue = v[v < vlim[0]]
        self.v_blue = v[(vlim[0] <= v) * (v <= vlim[1])]
        self.v_nanmid = v[(vlim[1] < v) * (v < vlim[2])]
        self.v_red = v[(vlim[2] <= v) * (v <= vlim[3])]
        self.v_nanred = v[vlim[3] < v]
        self.v_valid = np.r_[self.v_blue, self.v_red]

        self.data_blue = self.data[(vlim[0] <= v) * (v <= vlim[1])]
        self.data_red = self.data[(vlim[2] <= v) * (v <= vlim[3])]
        self.data_valid = np.append(self.data_blue, self.data_red, axis=0) 
        self.maxsnr = np.max(self.data_valid / self.sigma)
        
        m = makemom01(self.data_valid, self.v_valid, sigma)
        self.mom0 = m['mom0']
        self.mom1 = m['mom1']
        self.mom2 = m['mom2']
        self.sigma_mom0 = m['sigma_mom0']
        self.peak = np.max(self.data_valid, axis=0)
        self.signmajor = np.sign(np.nansum(self.mom1 * xmajor))
        self.signminor = np.sign(np.nansum(self.mom1 * xminor)) * (-1)

        def modelvlos(xmajor: np.ndarray, xminor: np.ndarray,
                      Mstar: float, Rc: float):
            rdisk = np.hypot(xmajor, xminor)
            vkep = vunit * np.sqrt(Mstar / rdisk)
            vjrot = vunit * np.sqrt(Mstar * Rc) / rdisk
            vr = -vunit * np.sqrt(Mstar / rdisk) * np.sqrt(2 - Rc / rdisk)
            vr[rdisk < Rc] = 0
            vrot = np.where(rdisk < Rc, vkep, vjrot)
            vlos = (vrot * xmajor * self.signmajor 
                    + vr * xminor * self.signminor) / rdisk
            vlos = vlos * np.sin(np.radians(incl))
            return vlos
        self.modelvlos = modelvlos
                        
    def fitting(self, Mstar_range: list = [0.01, 10],
                Rc_range: list = [1, 1000],
                cs_range: list = [0.01, 1],
                offmajor_range: list = [-100, 100],
                offminor_range: list = [-100, 100],
                offvsys_range: list = [-0.2, 0.2],
                Mstar_fixed: float = None,
                Rc_fixed: float = None,
                cs_fixed: float = None,
                offmajor_fixed: float = None,
                offminor_fixed: float = None,
                offvsys_fixed: float = None,
                nwalkers_per_ndim: int = 16,
                nburnin: int = 100,
                nsteps: int = 300,
                filename: str = 'channelfit',
                show: bool = False,
                progressbar: bool = True,
                nsubpixel: int = 5):
        
        n = len(self.x) - 1 if len(self.x) // 2 == 0 else len(self.x)
        xb = (np.arange(n) - (n - 1) // 2) * self.dx
        yb = (np.arange(n) - (n - 1) // 2) * self.dy
        xb, yb = np.meshgrid(xb, yb)
        xb, yb = rot(xb, yb, np.radians(self.bpa))
        gaussbeam = np.exp(-((yb / self.bmaj)**2 + (xb / self.bmin)**2))
        pixperbeam = np.sum(gaussbeam)
        gaussbeam = gaussbeam / pixperbeam
        
        nsubx = nsuby = nsubpixel
        subx = ((np.arange(nsubx) + 0.5) / nsubx - 0.5) * self.dx * self.deproj
        suby = ((np.arange(nsuby) + 0.5) / nsuby - 0.5) * self.dy
        subx, suby = [np.ravel(a) for a in np.meshgrid(subx, suby)]
        if offmajor_fixed is not None:
            xmajor0 = np.add.outer(suby, self.xmajor - offmajor_fixed)  # subxy, y, x
        if offminor_fixed is not None:
            xminor0 = np.add.outer(subx, self.xminor - offminor_fixed * self.deproj)  # subxy, y, x

        def cubemodel(logMstar: float, logRc: float, logcs: float,
                      offmajor: float, offminor: float, offvsys: float):
            global xmajor, xminor
            Mstar, Rc, cs = 10**logMstar, 10**logRc, 10**logcs
            if offmajor_fixed is None:
                xmajor = np.add.outer(suby, self.xmajor - offmajor)
            else:
                xmajor = xmajor0
            if offminor_fixed is None:
                xminor = np.add.outer(subx, self.xminor - offminor * self.deproj)
            else:
                xminor= xminor0
            vmodel = self.modelvlos(xmajor, xminor, Mstar, Rc)  # subxy, y, x
            v = np.subtract.outer(self.v_valid, vmodel + offvsys)  # v, subxy, y, x
            m = boxgauss(v / cs, self.dv / cs)  # v, subxy, y, x
            m = np.mean(m, axis=1)  # v, y, x
            m = fftconvolve(m, [gaussbeam], mode='same', axes=(1, 2))
            mom0 = np.nansum(m, axis=0) * self.dv
            m = np.where((mom0 > 0) * (self.mom0 > 3 * self.sigma_mom0),
                         m * self.mom0 / mom0, 0)
            return m
        self.cubemodel = cubemodel

        p_fixed = np.array([Mstar_fixed, Rc_fixed, cs_fixed,
                            offmajor_fixed, offminor_fixed, offvsys_fixed])
        if None in p_fixed:
            c = (q := p_fixed[:3]) != None
            p_fixed[:3][c] = np.log10(q[c].astype('float'))
            if progressbar:
                bar = tqdm(total=(nwalkers_per_ndim 
                                  * len(p_fixed[p_fixed == None]) 
                                  * (nburnin + 1 + nsteps + 1)))
                bar.set_description('Within the ranges')
            def lnprob(p):
                if progressbar:
                    bar.update(1)
                q = p_fixed.copy()
                q[p_fixed == None] = p
                m = self.cubemodel(*q)
                chi2 = np.nansum((self.data_valid - m)**2) / self.sigma**2
                chi2 /= pixperbeam
                return -0.5 * chi2
            plim = np.array([np.log10(Mstar_range),
                             np.log10(Rc_range),
                             np.log10(cs_range),
                             offmajor_range, offminor_range, offvsys_range])
            plim = plim[p_fixed == None].T
            labels = np.array(['log Mstar', 'log Rc', 'log cs',
                               'offmajor', 'offminor', 'offvsys'])
            labels = labels[p_fixed == None]
            mcmc = emcee_corner(plim, lnprob,
                                nwalkers_per_ndim=nwalkers_per_ndim,
                                nburnin=nburnin, nsteps=nsteps,
                                labels=labels, rangelevel=0.90,
                                figname=filename+'.corner.png',
                                show_corner=show,
                                simpleoutput=False)
            popt = p_fixed.copy()
            popt[p_fixed == None] = mcmc[0]
            popt[:3] = 10**popt[:3]
            plow = p_fixed.copy()
            plow[p_fixed == None] = mcmc[1]
            plow[:3] = 10**plow[:3]
            pmid = p_fixed.copy()
            pmid[p_fixed == None] = mcmc[2]
            pmid[:3] = 10**pmid[:3]
            phigh = p_fixed.copy()
            phigh[p_fixed == None] = mcmc[3]
            phigh[:3] = 10**phigh[:3]
            self.popt = popt
            self.plow = plow
            self.pmid = pmid
            self.phigh = phigh
        else:
            self.popt = p_fixed
            self.plow = p_fixed
            self.pmid = p_fixed
            self.phigh = p_fixed
        print('plow :', ', '.join([f'{t:.2e}' for t in self.plow]))
        print('pmid :', ', '.join([f'{t:.2e}' for t in self.pmid]))
        print('phigh:', ', '.join([f'{t:.2e}' for t in self.phigh]))
        print('------------------------')
        print('popt :', ', '.join([f'{t:.2e}' for t in self.popt]))
        print('------------------------')
        np.savetxt(filename+'.popt.txt', [self.popt, self.plow, self.pmid, self.phigh])
   
    def modeltofits(self, Mstar: float = None, Rc: float = None,
                    offmajor: float = None, offminor: float = None,
                    offvsys: float = None,
                    cs: float = None, filehead: str = 'best'):
        w = wcs.WCS(naxis=3)
        h = fits.open(self.fitsname)[0].header
        h['NAXIS1'] = len(self.x)
        h['NAXIS2'] = len(self.y)
        #h['NAXIS3'] = len(self.v)
        h['CRPIX1'] = h['CRPIX1'] - self.offpix[0]
        h['CRPIX2'] = h['CRPIX2'] - self.offpix[1]
        #h['CRPIX3'] = h['CRPIX3'] - self.offpix[2]
        nx, ny, nv = h['NAXIS1'], h['NAXIS2'], h['NAXIS3']

        if None in [Mstar, Rc, cs, offmajor, offminor, offvsys]:
            Mstar, Rc, cs, offmajor, offminor, offvsys = self.popt
        logMstar, logRc, logcs = np.log10(Mstar), np.log10(Rc), np.log10(cs)
        m = self.cubemodel(logMstar, logRc, logcs, offmajor, offminor, offvsys)
        m_red = m[np.max(self.v_blue) < self.v_valid]
        m_blue = m[self.v_valid < np.min(self.v_red)]
        nanblue = np.full((len(self.v_nanblue), ny, nx), np.nan)
        nanmid = np.full((len(self.v_nanmid), ny, nx), np.nan)
        nanred = np.full((len(self.v_nanred), ny, nx), np.nan)
        model = m_blue
        if len(nanblue) > 0:
            model = np.append(nanblue, model, axis=0)
        if len(nanmid) > 0:
            model = np.append(model, nanmid, axis=0)
        model = np.append(model, m_red, axis=0)
        if len(nanred) > 0:
            model = np.append(model, nanred, axis=0)
                
        def tofits(d: np.ndarray, ext: str):
            header = w.to_header()
            hdu = fits.PrimaryHDU(d, header=header)
            for k in h.keys():
                if not ('COMMENT' in k or 'HISTORY' in k):
                    hdu.header[k]=h[k]
            hdu = fits.HDUList([hdu])
            hdu.writeto(f'{filehead}.{ext}.fits', overwrite=True)
        tofits(model, 'model')
        tofits(self.data - model, 'residual')
        
    def plotmodelmom(self, Mstar: float, Rc: float, cs: float,
                     offmajor: float, offminor: float, offvsys: float,
                     filename: str = 'modelmom01.png', pa: float = None):
        logMstar, logRc, logcs = np.log10(Mstar), np.log10(Rc), np.log10(cs)
        d = self.cubemodel(logMstar, logRc, logcs, offmajor, offminor, offvsys)
        m = makemom01(d, self.v_valid, self.sigma)
        mom0 = m['mom0']
        mom1 = m['mom1']
        sigma_mom0 = m['sigma_mom0']
        levels = np.arange(1, 20) * 6 * sigma_mom0
        levels = levels[::2]
        levels = np.sort(np.r_[-levels, levels])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        vplot = (np.nanpercentile(self.mom1, 95) 
                 - np.nanpercentile(self.mom1, 5)) / 2.
        m = ax.pcolormesh(self.x, self.y, mom1, cmap='jet',
                          vmin=-vplot, vmax=vplot)
        fig.colorbar(m, ax=ax, label=r'Model mom1 (km s$^{-1}$)')
        ax.contour(self.x, self.y, mom0, colors='gray', levels=levels)
        if pa is not None:
            r = np.linspace(-1, 1, 10) * self.x.max() * 1.5
            ax.plot(r * np.sin(np.radians(pa)), r * np.cos(np.radians(pa)),
                    'k:')
            ax.plot(r * np.cos(np.radians(pa)), -r * np.sin(np.radians(pa)),
                    'k:')
        ax.set_xlabel('R.A. offset (au)')
        ax.set_ylabel('Dec. offset (au)')
        ax.set_xlim(self.x.max() * 1.01, self.x.min() * 1.01)
        ax.set_ylim(self.y.min() * 1.01, self.y.max() * 1.01)
        ax.set_aspect(1)
        fig.savefig(filename)
        plt.close()

    def plotobsmom(self, filename: str = 'obsmom01.png', pa: float = None):
        levels = np.arange(1, 20) * 6 * self.sigma_mom0
        levels = np.sort(np.r_[-levels, levels])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        vplot = (np.nanpercentile(self.mom1, 95) 
                 - np.nanpercentile(self.mom1, 5)) / 2.
        m = ax.pcolormesh(self.x, self.y, self.mom1, cmap='jet',
                          vmin=-vplot, vmax=vplot)
        fig.colorbar(m, ax=ax, label=r'Obs. mom1 (km s$^{-1}$)')
        ax.contour(self.x, self.y, self.mom0, colors='gray', levels=levels)
        if pa is not None:
            r = np.linspace(-1, 1, 10) * self.x.max() * 1.5
            ax.plot(r * np.sin(np.radians(pa)), r * np.cos(np.radians(pa)),
                    'k:')
            ax.plot(r * np.cos(np.radians(pa)), -r * np.sin(np.radians(pa)),
                    'k:')
        ax.set_xlabel('R.A. offset (au)')
        ax.set_ylabel('Dec. offset (au)')
        ax.set_xlim(self.x.max() * 1.01, self.x.min() * 1.01)
        ax.set_ylim(self.y.min() * 1.01, self.y.max() * 1.01)
        ax.set_aspect(1)
        fig.savefig(filename)
        plt.close()

    def plotresidualmom(self, Mstar: float, Rc: float, cs: float,
                        offmajor: float, offminor: float, offvsys: float,
                        filename: str = 'residualmom01.png',
                        pa: float = None):
        logMstar, logRc, logcs = np.log10(Mstar), np.log10(Rc), np.log10(cs)
        d = self.cubemodel(logMstar, logRc, logcs, offmajor, offminor, offvsys)
        m = makemom01(d, self.v_valid, self.sigma)
        mom0 = m['mom0']
        mom1 = self.mom1 - m['mom1']
        sigma_mom0 = m['sigma_mom0']
        levels = np.arange(1, 20) * 6 * sigma_mom0
        levels = np.sort(np.r_[-levels, levels])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        m = ax.pcolormesh(self.x, self.y, mom1, cmap='jet',
                          vmin=-self.dv * 3, vmax=self.dv * 3)
        fig.colorbar(m, ax=ax, label=r'Obs. $-$ model mom1 (km s$^{-1}$)')
        ax.contour(self.x, self.y, mom0, colors='gray', levels=levels)
        if pa is not None:
            r = np.linspace(-1, 1, 10) * self.x.max() * 1.5
            ax.plot(r * np.sin(np.radians(pa)), r * np.cos(np.radians(pa)),
                    'k:')
            ax.plot(r * np.cos(np.radians(pa)), -r * np.sin(np.radians(pa)),
                    'k:')
        ax.set_xlabel('R.A. offset (au)')
        ax.set_ylabel('Dec. offset (au)')
        ax.set_xlim(self.x.max() * 1.01, self.x.min() * 1.01)
        ax.set_ylim(self.y.min() * 1.01, self.y.max() * 1.01)
        ax.set_aspect(1)
        fig.savefig(filename)
        plt.close()

