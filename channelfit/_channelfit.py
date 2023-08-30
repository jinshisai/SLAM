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
from scipy.signal import convolve
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

def makemom01(d: np.ndarray, v: np.ndarray, sigma: float) -> dict:
    dmasked = d * 1
    dmasked[np.isnan(dmasked)] = 0
    dv = np.min(v[1:] - v[:-1])
    mom0 = np.sum(d, axis=0) * dv
    sigma_mom0 = sigma * dv * np.sqrt(len(d))
    vv = np.broadcast_to(v, np.shape(d)[::-1])
    vv = np.moveaxis(vv, 2, 0)
    dmasked[dmasked < 3 * sigma] = 0
    mom1 = np.sum(d * vv, axis=0) / np.sum(d, axis=0)
    mom1[mom0 < 3 * sigma_mom0] = np.nan
    return {'mom0':mom0, 'mom1':mom1, 'sigma_mom0':sigma_mom0}
    

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
        x, y, v = x[i0:i1 + 1], y[j0:j1 + 1], v[k0:k1 + 1],
        d =  d[k0:k1 + 1, j0:j1 + 1, i0:i1 + 1]
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
        deproj = 1 / np.cos(np.radians(incl))
        xminor = xminor * deproj
        self.xminor, self.xmajor = xminor, xmajor
        
        self.v_blue = v[v <= vlim[1]]
        self.v_red = v[vlim[2] <= v]
        self.v_mid = v[(vlim[1] < v) * (v < vlim[2])]
        self.v_valid = np.r_[self.v_blue, self.v_red]
        self.data_blue = self.data[v <= vlim[1]]
        self.data_red = self.data[vlim[2] <= v]
        self.data_mid = self.data[(vlim[1] < v) * (v < vlim[2])]
        self.data_valid = np.append(self.data_blue, self.data_red, axis=0) 
        
        m = makemom01(self.data_valid, self.v_valid, sigma)
        self.mom0 = m['mom0']
        self.mom1 = m['mom1']
        self.sigma_mom0 = m['sigma_mom0']
        self.peak = np.max(self.data_valid, axis=0)
        self.signmajor = np.sign(np.nansum(self.mom1 * xmajor))
        self.signminor = np.sign(np.nansum(self.mom1 * xminor)) * (-1)

        def modelvlos(Mstar: float, Rc: float, offmajor: float, offminor:float):
            xmajor = self.xmajor - offmajor
            xminor = self.xminor - offminor * deproj
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

        n = len(self.x) - 1 if len(self.x) // 2 == 0 else len(self.x)
        xb = (np.arange(n) - (n - 1) // 2) * self.dx
        yb = (np.arange(n) - (n - 1) // 2) * self.dy
        xb, yb = np.meshgrid(xb, yb)
        xb, yb = rot(xb, yb, np.radians(self.bpa))
        gaussbeam = np.exp(-((yb / self.bmaj)**2 + (xb / self.bmin)**2))
        self.pixperbeam = np.sum(gaussbeam)
        gaussbeam = gaussbeam / self.pixperbeam
        self.pixel_valid = len(self.x) * len(self.y) * len(self.v_valid)
                
        def cubemodel(logMstar: float, logRc: float, logcs: float,
                      offmajor: float, offminor: float):
            #cs = self.mom2
            Mstar, Rc, cs = 10**logMstar, 10**logRc, 10**logcs
            v = self.v_valid
            m = [None] * len(v)
            for i in range(len(v)):
                m[i] = np.exp(-(v[i] - modelvlos(Mstar, Rc, offmajor, offminor))**2 / 2 / cs**2) \
                       / np.sqrt(2 * np.pi) / cs
                m[i] = convolve(m[i], gaussbeam, mode='same')
            m = np.array(m)
            mom0 = np.nansum(m, axis=0) * self.dv
            m = np.where((self.peak < 6 * self.sigma) * (mom0 < 3 * self.sigma_mom0), 0,
                         m * np.broadcast_to(self.mom0 / mom0, np.shape(m)))
            return m
        self.cubemodel = cubemodel
    
    def plotmodelmom(self, Mstar: float, Rc: float, cs: float,
                     offmajor: float, offminor: float,
                     filename: str = 'modelmom01.png', pa: float = None):
        logMstar, logRc, logcs = np.log10(Mstar), np.log10(Rc), np.log10(cs)
        d = self.cubemodel(logMstar, logRc, logcs, offmajor, offminor)
        m = makemom01(d, self.v_valid, self.sigma)
        mom0 = m['mom0']
        mom1 = m['mom1']
        sigma_mom0 = m['sigma_mom0']
        levels = np.arange(1, 20) * 3 * sigma_mom0
        levels = levels[::2]
        levels = np.sort(np.r_[-levels, levels])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        m = ax.pcolormesh(self.x, self.y, mom1, cmap='jet',
                          vmin=np.nanmin(self.mom1),
                          vmax=np.nanmax(self.mom1))
        fig.colorbar(m, ax=ax, label='km/s')
        ax.contour(self.x, self.y, mom0, colors='gray', levels=levels)
        if pa is not None:
            r = np.linspace(-1, 1, 10) * self.x.max() * 1.5
            ax.plot(r * np.sin(np.radians(pa)), r * np.cos(np.radians(pa)),
                    'k:')
            ax.plot(r * np.cos(np.radians(pa)), -r * np.sin(np.radians(pa)),
                    'k:')
        ax.set_xlabel('R.A. offset (au)')
        ax.set_ylabel('Dec. offset (au)')
        ax.set_xlim(self.x.max(), self.x.min())
        ax.set_ylim(self.y.min(), self.y.max())
        ax.set_aspect(1)
        fig.savefig(filename)
        plt.close()

    def plotobsmom(self, filename: str = 'obsmom01.png', pa: float = None):
        levels = np.arange(1, 20) * 3 * self.sigma_mom0
        levels = levels[::2]
        levels = np.sort(np.r_[-levels, levels])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        m = ax.pcolormesh(self.x, self.y, self.mom1, cmap='jet',
                          vmin=np.nanmin(self.mom1),
                          vmax=np.nanmax(self.mom1))
        fig.colorbar(m, ax=ax, label='km/s')
        ax.contour(self.x, self.y, self.mom0, colors='gray', levels=levels)
        if pa is not None:
            r = np.linspace(-1, 1, 10) * self.x.max() * 1.5
            ax.plot(r * np.sin(np.radians(pa)), r * np.cos(np.radians(pa)),
                    'k:')
            ax.plot(r * np.cos(np.radians(pa)), -r * np.sin(np.radians(pa)),
                    'k:')
        ax.set_xlabel('R.A. offset (au)')
        ax.set_ylabel('Dec. offset (au)')
        ax.set_xlim(self.x.max(), self.x.min())
        ax.set_ylim(self.y.min(), self.y.max())
        ax.set_aspect(1)
        fig.savefig(filename)
        plt.close()

    def plotresidualmom(self, Mstar: float, Rc: float, cs: float,
                        offmajor: float, offminor: float,
                        filename: str = 'residualmom01.png',
                        pa: float = None):
        logMstar, logRc, logcs = np.log10(Mstar), np.log10(Rc), np.log10(cs)
        d = self.cubemodel(logMstar, logRc, logcs, offmajor, offminor)
        m = makemom01(d, self.v_valid, self.sigma)
        mom0 = m['mom0']
        mom1 = self.mom1 - m['mom1']
        sigma_mom0 = m['sigma_mom0']
        levels = np.arange(1, 20) * 3 * sigma_mom0
        levels = levels[::2]
        levels = np.sort(np.r_[-levels, levels])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        m = ax.pcolormesh(self.x, self.y, mom1, cmap='jet',
                          vmin=np.nanmin(self.mom1),
                          vmax=np.nanmax(self.mom1))
        fig.colorbar(m, ax=ax, label='km/s')
        ax.contour(self.x, self.y, mom0, colors='gray', levels=levels)
        if pa is not None:
            r = np.linspace(-1, 1, 10) * self.x.max() * 1.5
            ax.plot(r * np.sin(np.radians(pa)), r * np.cos(np.radians(pa)),
                    'k:')
            ax.plot(r * np.cos(np.radians(pa)), -r * np.sin(np.radians(pa)),
                    'k:')
        ax.set_xlabel('R.A. offset (au)')
        ax.set_ylabel('Dec. offset (au)')
        ax.set_xlim(self.x.max(), self.x.min())
        ax.set_ylim(self.y.min(), self.y.max())
        ax.set_aspect(1)
        fig.savefig(filename)
        plt.close()
    
    def fitting(self, Mstar_range: list = [0.01, 10],
                Rc_range: list = [1, 1000],
                cs_range: list = [0.01, 1],
                offmajor_range: list = [-100, 100],
                offminor_range: list = [-100, 100],
                Mstar_fixed: float = None,
                Rc_fixed: float = None,
                cs_fixed: float = None,
                offmajor_fixed: float = None,
                offminor_fixed: float = None,
                filename: str = 'channelfit',
                show: bool = False,
                progressbar: bool = True):
        p_fixed = np.array([Mstar_fixed, Rc_fixed, cs_fixed,
                            offmajor_fixed, offminor_fixed])
        if None in p_fixed:
            c = (q := p_fixed[:3]) != None
            p_fixed[:3][c] = np.log10(q[c].astype('float'))
            if progressbar:
                bar = tqdm(total=(8 * len(p_fixed[p_fixed == None]) 
                                  * (100 + 1 + 100 + 1)))
                bar.set_description('Within the ranges')
            def lnprob(p):
                if progressbar:
                    bar.update(1)
                q = p_fixed.copy()
                q[p_fixed == None] = p
                m = self.cubemodel(*q)
                chi2 = np.nansum((self.data_valid - m)**2) / self.sigma**2
                chi2 /= self.pixperbeam
                return -0.5 * chi2
            plim = np.array([np.log10(Mstar_range),
                             np.log10(Rc_range),
                             np.log10(cs_range),
                             offmajor_range, offminor_range])
            plim = plim[p_fixed == None].T
            labels = np.array(['log Mstar', 'log Rc', 'log cs', 'offmajor', 'offminor'])
            labels = labels[p_fixed == None]
            mcmc = emcee_corner(plim, lnprob,
                                nwalkers_per_ndim=8, nburnin=100, nsteps=100,
                                labels=labels, rangelevel=0.95,
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
                    cs: float = None, filehead: str = 'best'):
        if None in [Mstar, Rc, cs, offmajor, offminor]:
            Mstar, Rc, cs, offmajor, offminor = self.popt
        logMstar, logRc, logcs = np.log10(Mstar), np.log10(Rc), np.log10(cs)
        m = self.cubemodel(logMstar, logRc, logcs, offmajor, offminor)
        m_red = m[np.max(self.v_blue) < self.v_valid]
        m_blue = m[self.v_valid < np.min(self.v_red)]
        model = np.full_like(self.data_mid, np.nan)
        model = np.append(m_blue, model, axis=0)
        model = np.append(model, m_red, axis=0)
                
        w = wcs.WCS(naxis=3)
        h = fits.open(self.fitsname)[0].header
        h['NAXIS1'] = len(self.x)
        h['NAXIS2'] = len(self.y)
        h['NAXIS3'] = len(self.v)
        h['CRPIX1'] = h['CRPIX1'] - self.offpix[0]
        h['CRPIX2'] = h['CRPIX2'] - self.offpix[1]
        h['CRPIX3'] = h['CRPIX3'] - self.offpix[2]
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
