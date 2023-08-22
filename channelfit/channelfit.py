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

############ INPUTS ############
cubefits = './channelfit/IRAS16253_SBLB_C18O_robust_2.0.imsub.fits'
center = '16h28m21.61526785s -24d36m24.32538414s'
pa = 113 - 180  # deg; pa is redshift rotation, pa + 90 is blueshifted infall
incl = 65  # deg
vsys = 4  # km/s
dist = 139  # pc
sigma = 2e-3  # Jy/beam; None means automatic calculation.
rmax = 1 * dist  # au
#vlim = (-2.52, -1.4, 1.4, 2.52)  # km/s
vlim = (-2.52, -0.9, 0.9, 2.52)  # km/s
show_figs = True
################################


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import constants, units, wcs
from astropy.coordinates import SkyCoord
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.signal import convolve
import warnings
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

def makemom01(d, v, sigma):
    dv = np.min(v[1:] - v[:-1])
    mom0 = np.nansum(d, axis=0) * dv
    sigma_mom0 = sigma * dv * np.sqrt(len(d))
    vv = np.broadcast_to(v, np.shape(d)[::-1])
    vv = np.moveaxis(vv, 2, 0)
    dmasked = d * 1
    dmasked[dmasked < 3 * sigma] = np.nan
    mom1 = np.nansum(d * vv, axis=0) / np.nansum(d, axis=0)
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
        dist : float
            Distance of the target, used to convert arcsec to au.
        center : str
            Coordinates of the target: e.g., "01h23m45.6s 01d23m45.6s".
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
                   pa: float = 0, incl: float = 0, dist: float = 1,
                   center: str = None, vsys: float = 0,
                   rmax: float = 1e4, vlim: tuple = (-100, 0, 0, 100),
                   sigma: float = None):
        if not (cubefits is None):
            self.read_cubefits(cubefits, center, dist, vsys,
                               rmax, rmax, vlim[0], vlim[3], sigma)
            self.fitsname = cubefits
        x, y, v = self.x, self.y, self.v
        s, t = np.meshgrid(x, y)
        s, t = rot(s, t, np.radians(pa))
        s = s / np.cos(np.radians(incl))
        self.dmaj = t
        self.dmin = s
        self.rdisk = np.hypot(s, t)
        
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

        def modelvlos(Mstar: float, Rc: float):
            vkep = vunit * np.sqrt(Mstar / self.rdisk)
            vjrot = vunit * np.sqrt(Mstar * Rc) / self.rdisk
            vr = -vunit * np.sqrt(Mstar / self.rdisk) * np.sqrt(2 - Rc / self.rdisk)
            vr[self.rdisk < Rc] = 0
            vrot = np.where(self.rdisk < Rc, vkep, vjrot)
            vlos = (vrot * self.dmaj + vr * self.dmin) / self.rdisk
            vlos = vlos * np.sin(np.radians(incl))
            return vlos
        self.modelvlos = modelvlos

        n = len(x) - 1 if len(x) // 2 == 0 else len(x)
        xb = (np.arange(n) - (n - 1) // 2) * self.dx
        yb = (np.arange(n) - (n - 1) // 2) * self.dy
        xb, yb = np.meshgrid(xb, yb)
        zb = (yb + 1j * xb) * np.exp(-1j * np.radians(self.bpa))
        yb = np.real(zb) / self.bmaj
        xb = np.imag(zb) / self.bmin
        gaussbeam = np.exp(-(yb**2 + xb**2))
        self.pixperbeam = np.sum(gaussbeam)
        gaussbeam = gaussbeam / self.pixperbeam
        self.pixel_valid = len(self.x) * len(self.y) * len(self.v_valid)
                
        def cubemodel(Mstar: float, Rc: float, cs: float):
            #cs = self.mom2
            v = self.v_valid
            m = [None] * len(v)
            for i in range(len(v)):
                m[i] = np.exp(-(v[i] - modelvlos(Mstar, Rc))**2 / 2 / cs**2) \
                       / np.sqrt(2 * np.pi) / cs
                m[i] = convolve(m[i], gaussbeam, mode='same')
            m = np.array(m)
            mom0 = np.nansum(m, axis=0) * self.dv
            m = m * np.broadcast_to(self.mom0 / mom0, np.shape(m))
            m[np.isnan(m)] = 0
            return m
        self.cubemodel = cubemodel
    
    def plotmodelmom(self, Mstar: float, Rc: float, cs: float,
                     filename: str = 'modelmom01.png', pa: float = None):
        d = self.cubemodel(Mstar, Rc, cs)
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
        plt.show()
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
        plt.show()
        plt.close()

    def plotresidualmom(self, Mstar: float, Rc: float, cs: float,
                        filename: str = 'residualmom01.png',
                        pa: float = None):
        d = self.cubemodel(Mstar, Rc, cs)
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
        plt.show()
        plt.close()
    
    def fitting(self, Mstar_range: list, Rc_range: list, cs_range: list,
                figname: str):

        def lnprob(p):
            q = 10**p
            m = self.cubemodel(*q)
            chi2 = np.nansum((self.data_valid - m)**2) / self.sigma**2
            chi2 = chi2 / self.pixperbeam
            return -0.5 * chi2
        plim = np.log10([Mstar_range, Rc_range, cs_range]).T
        mcmc = emcee_corner(plim, lnprob,
                            nwalkers_per_ndim=16, nburnin=200, nsteps=200,
                            labels=['log Mstar', 'log Rc', 'log cs'],
                            rangelevel=95,
                            figname=figname+'.corner.png',
                            show_corner=False)
        popt, perr = mcmc
        popt = 10**popt
        perr = popt * np.log(10) * perr
        self.popt = popt
        self.perr = perr
        print(popt)
   
    def modeltofits(self, Mstar: float = None, Rc: float = None,
                    cs: float = None, filehead: str = 'best'):
        if Mstar is None or Rc is None or cs is None:
            Mstar, Rc, cs = self.popt
        m = self.cubemodel(Mstar, Rc, cs)
        m_red = m[np.max(self.v_blue) < self.v]
        m_blue = m[self.v < np.min(self.v_red)]
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
        

#####################################################################
if __name__ == '__main__':
    filehead = cubefits.replace('.fits', '')
    chan = ChannelFit()
    chan.gridondisk(cubefits=cubefits, center=center, pa=pa, incl=incl,
                 vsys=vsys, dist=dist, sigma=sigma,
                 rmax=rmax, vlim=vlim)
    #chan.fitting(Mstar_range=[0.01, 10.0], Rc_range=[5, 500], cs_range=[0.2, 2],
    #             figname=filehead)
    #chan.modeltofits(filehead=filehead)
    #chan.popt
    #p = [0.08329, 283.34, 1.113]
    p = [0.04834, 345.02, 0.9338]
    chan.plotmodelmom(*p, pa=113, filename=filehead+'.modelmom01.png')
    chan.plotobsmom(pa=113, filename=filehead+'.obsmom01.png')
    chan.plotresidualmom(*p, pa=113, filename=filehead+'.residualmom01.png')
    #chan.modeltofits(0.0825, 573.635, 1.0429, filehead=filehead)