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
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import interp1d
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

def avefour(a: np.ndarray) -> np.ndarray:
    b = (a[:, 0::2, 0::2] + a[:, 0::2, 1::2] 
         + a[:, 1::2, 0::2] + a[:, 1::2, 1::2]) / 4.
    return b

def boxgauss(dv_over_cs: float) -> tuple:
    clipsigma = 3. + dv_over_cs  # in the unit of cs
    dv = min([2, dv_over_cs]) / 10.
    ndv = 2 * int(dv_over_cs / 2. / dv + 0.5) + 1  # 0.5 is for rounding
    n = 2 * int(clipsigma / dv + 0.5) + 1
    v = np.linspace(-clipsigma, clipsigma, n)
    g = np.exp(-0.5 * v**2)
    #g /= np.sum(g)
    n = n - ndv
    p = np.sum([g[i:i + n + 1] for i in range(ndv)], axis=0)  # n = len(p) - 1
    #p /= ndv
    p[0] = p[n] = 0
    return p, n, dv
    
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

    #def __init__(self):
        

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
        self.x, self.dx, self.nx = x, dx, len(x)
        self.y, self.dy, self.ny = y, dy, len(y)
        self.v, self.dv, self.nv = v, dv, len(v)
        self.data, self.header, self.sigma = d, h, sigma
        self.bmaj, self.bmin, self.bpa = bmaj, bmin, bpa
        self.cubefits, self.dist, self.vsys = cubefits, dist, vsys
        return {'x':x, 'y':y, 'v':v, 'data':d, 'header':h, 'sigma':sigma}


    def gridondisk(self, cubefits: str = None,
                   pa: float = 0, incl: float = 90, dist: float = 1,
                   center: str = None, vsys: float = 0,
                   rmax: float = 1e4, vlim: tuple = (-100, 0, 0, 100),
                   sigma: float = None, nlayer: int = 4):
        if not (cubefits is None):
            self.read_cubefits(cubefits, center, dist, vsys,
                               rmax, rmax, vlim[0], vlim[3], sigma)
            self.fitsname = cubefits
            v = self.v
        pa_rad = np.radians(pa)
        incl_rad = np.radians(incl)
        self.cospa = np.cos(pa_rad)
        self.sinpa = np.sin(pa_rad)
        self.sini = np.sin(incl_rad)
        self.deproj = 1 / np.cos(incl_rad)
        xminor, xmajor = rot(*np.meshgrid(self.x, self.y), pa_rad)
        self.xmajor = xmajor
        self.xminor = xminor * self.deproj
        
        self.v_nanblue = v[v < vlim[0]]
        self.v_blue = v[(vlim[0] <= v) * (v <= vlim[1])]
        self.v_nanmid = v[(vlim[1] < v) * (v < vlim[2])]
        self.v_red = v[(vlim[2] <= v) * (v <= vlim[3])]
        self.v_nanred = v[vlim[3] < v]
        self.v_valid = np.r_[self.v_blue, self.v_red]

        self.data_blue = self.data[(vlim[0] <= v) * (v <= vlim[1])]
        self.data_red = self.data[(vlim[2] <= v) * (v <= vlim[3])]
        self.data_valid = np.append(self.data_blue, self.data_red, axis=0) 
        
        m = makemom01(self.data_valid, self.v_valid, sigma)
        self.mom0 = m['mom0']
        self.mom1 = m['mom1']
        self.mom2 = m['mom2']
        self.sigma_mom0 = m['sigma_mom0']
        self.signmajor = np.sign(np.nansum(self.mom1 * xmajor))
        self.signminor = np.sign(np.nansum(self.mom1 * xminor)) * (-1)

        # 2d nested grid on the disk plane.
        # x and y are minor and major axis coordinates before projection.
        dpix = min([np.abs(self.dx), np.abs(self.dy)])
        npix = max([len(self.x), len(self.y)])
        npixnest = int(2**(np.ceil(np.log2(npix))))
        self.nq1 = npixnest // 4
        self.nq3 = self.nq1 + npixnest // 2
        self.nlayer = nlayer  # down to dpix / 2**(nlayer-1)
        xnest = [None] * nlayer
        ynest = [None] * nlayer
        erot = [None] * nlayer
        erad = [None] * nlayer
        Rnest = [None] * nlayer
        for l in range(nlayer):
            n = npixnest // 2 + 0.5
            s = np.linspace(-n, n, npixnest) * dpix / 2**l
            X, Y = np.meshgrid(s, s)
            R = np.hypot(X, Y)
            xnest[l] = s
            ynest[l] = s
            erot[l] = Y * self.signmajor / R
            erad[l] = X * self.signminor / R
            Rnest[l] = R
        self.xnest = np.array(xnest)
        self.ynest = np.array(ynest)
        self.erot = np.array(erot)
        self.erad = np.array(erad)
        self.Rnest = np.array(Rnest)
        z3d, y3d, x3d = np.meshgrid(self.xnest[0], self.ynest[0],
                                    self.xnest[0], indexing='ij')
        h_min = np.abs(z3d / np.hypot(x3d + z3d, y3d * np.cos(incl_rad)) / np.tan(incl_rad))
        self.h_min = np.moveaxis([h_min] * len(self.v_valid), 0, 1)
        print('-------- nested grid --------')
        for l in range(len(xnest)):
            print(f'x, dx, npix: +/-{xnest[l][-1]:.2f},'
                  + f' {xnest[l][1]-xnest[l][0]:.2f} au,'
                  + f' {npixnest:d}')
        print('-----------------------------')
        
        xb = (np.arange(npixnest + 1) - npixnest // 2) * self.dx
        yb = (np.arange(npixnest + 1) - npixnest // 2) * self.dy
        xb, yb = rot(*np.meshgrid(xb, yb), np.radians(self.bpa))
        gaussbeam = np.exp(-((yb / self.bmaj)**2 + (xb / self.bmin)**2))
        self.pixperbeam = np.sum(gaussbeam)
        self.gaussbeam = gaussbeam / self.pixperbeam

        
    def get_vlos(self, Rc: float) -> np.ndarray:
        vkep = self.Rnest**(-1/2)
        vjrot = np.sqrt(Rc) / self.Rnest
        vr = -self.Rnest**(-1/2) * np.sqrt(2 - Rc / self.Rnest)
        vr[self.Rnest < Rc] = 0
        vrot = np.where(self.Rnest < Rc, vkep, vjrot)
        vlos = vrot * self.erot + vr * self.erad
        vlos = vlos * self.sini * vunit
        return vlos

        
    def cubemodel(self, Mstar: float, Rc: float, cs: float,
                  hdisk: float,
                  offmajor: float, offminor: float, offvsys: float):
        if self.cs_fixed is None:
            prof, n_prof, dv_prof = boxgauss(self.dv / cs)
        else:
            prof, n_prof, dv_prof = self.prof, self.n_prof, self.dv_prof
        if self.Rc_fixed is None:
            vlos = self.get_vlos(Rc)
        else:
            vlos = self.vlos
        vlos = vlos * np.sqrt(Mstar)  # Don't use *=. It changes self.vlos.
        v = np.subtract.outer(self.v_valid, vlos) - offvsys  # v, layer, y, x
        v = np.moveaxis(v, 1, 0)  # layer, v, y, x
        iv = v / cs / dv_prof + n_prof // 2 + 0.5  # 0.5 is for rounding
        iv = iv.astype('int').clip(0, n_prof)
        Iout = prof[iv]
        for l in range(self.nlayer - 1, 0, -1):
            Iout[l - 1][:, self.nq1:self.nq3, self.nq1:self.nq3] \
                = avefour(Iout[l])
        Iout = Iout[0]  # v, y, x
        f = interp1d(self.xnest[0], Iout, kind='cubic', axis=2,
                     bounds_error=False, fill_value=0)
        Iout = f(self.xnest[0] * self.deproj)
        nx = len(self.xnest[0])
        j0 = np.arange(nx) - (nx - 1) / 2
        Iz = [None] * nx
        for i in range(nx):
            j = (j0 + i).astype('int').clip(0, nx - 1)
            Iz[i] = Iout[:, :, j]
        Iz = np.array(Iz)  # z, v, y, x
        Iout = np.sum(Iz * (self.h_min < hdisk), axis=0) / self.sini
        y = self.xmajor - offmajor
        x = self.xminor - offminor
        m = [None] * len(Iout)
        for i, c in enumerate(Iout):
            interp = RGI((self.ynest[0], self.xnew), c,
                         bounds_error=False, fill_value=0)
            m[i] = interp((y, x))
        #m = np.array(m) / np.max(m)
        Iout = convolve(m, [self.gaussbeam], mode='same')
        mom0 = np.nansum(Iout, axis=0) * self.dv
        Iout = np.where((self.mom0 > 3 * self.sigma_mom0),
                        Iout * self.mom0 / mom0, 0)
        return Iout

                        
    def fitting(self, Mstar_range: list = [0.01, 10],
                Rc_range: list = [1, 1000],
                cs_range: list = [0.01, 1],
                hdisk_range: list = [0.01, 1],
                offmajor_range: list = [-100, 100],
                offminor_range: list = [-100, 100],
                offvsys_range: list = [-0.2, 0.2],
                Mstar_fixed: float = None,
                Rc_fixed: float = None,
                cs_fixed: float = None,
                hdisk_fixed: float = None,
                offmajor_fixed: float = None,
                offminor_fixed: float = None,
                offvsys_fixed: float = None,
                filename: str = 'channelfit',
                show: bool = False,
                progressbar: bool = True,
                kwargs_emcee_corner: dict = {}):

        self.cs_fixed = cs_fixed        
        if cs_fixed is not None:            
            self.prof, self.n_prof, self.dv_prof = boxgauss(self.dv / cs_fixed)
        self.Rc_fixed = Rc_fixed
        if Rc_fixed is not None:
            self.vlos = self.get_vlos(Rc_fixed)
        
        p_fixed = np.array([Mstar_fixed, Rc_fixed, cs_fixed, hdisk_fixed,
                            offmajor_fixed, offminor_fixed, offvsys_fixed])
        if None in p_fixed:
            c = (q := p_fixed[:3]) != None
            p_fixed[:3][c] = np.log10(q[c].astype('float'))
            labels = np.array(['log Mstar', 'log Rc', 'log cs',
                               'hdisk',
                               'offmajor', 'offminor', 'offvsys'])
            labels = labels[p_fixed == None]
            kwargs0 = {'nwalkers_per_ndim':16, 'nburnin':1000, 'nsteps':1000,
                       'labels': labels, 'rangelevel':None,
                       'figname':filename+'.corner.png', 'show_corner':show}
            kwargs = dict(kwargs0, **kwargs_emcee_corner)
            if progressbar:
                total = kwargs['nwalkers_per_ndim'] * len(p_fixed[p_fixed == None])
                total *= kwargs['nburnin'] + kwargs['nsteps'] + 2
                bar = tqdm(total=total)
                bar.set_description('Within the ranges')
            def lnprob(p):
                if progressbar:
                    bar.update(1)
                q = p_fixed.copy()
                q[p_fixed == None] = p
                q[:3] = 10**q[:3]
                chi2 = np.nansum((self.data_valid - self.cubemodel(*q))**2) \
                       / self.sigma**2 / self.pixperbeam
                return -0.5 * chi2
            plim = np.array([np.log10(Mstar_range),
                             np.log10(Rc_range),
                             np.log10(cs_range),
                             hdisk_range,
                             offmajor_range, offminor_range, offvsys_range])
            plim = plim[p_fixed == None].T
            mcmc = emcee_corner(plim, lnprob, simpleoutput=False, **kwargs)
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
        k = ['Mstar', 'Rc', 'cs', 'hdisk', 'offmajor', 'offminor', 'offvsys']
        self.popt = dict(zip(k, self.popt))
        self.plow = dict(zip(k, self.plow))
        self.pmid = dict(zip(k, self.pmid))
        self.phigh = dict(zip(k, self.phigh))
 
   
    def modeltofits(self, Mstar: float = None, Rc: float = None,
                    cs: float = None, hdisk: float = None,
                    offmajor: float = None, offminor: float = None,
                    offvsys: float = None,
                    filehead: str = 'best'):
        w = wcs.WCS(naxis=3)
        h = fits.open(self.fitsname)[0].header
        h['NAXIS1'] = len(self.x)
        h['NAXIS2'] = len(self.y)
        #h['NAXIS3'] = len(self.v)
        h['CRPIX1'] = h['CRPIX1'] - self.offpix[0]
        h['CRPIX2'] = h['CRPIX2'] - self.offpix[1]
        #h['CRPIX3'] = h['CRPIX3'] - self.offpix[2]
        nx = h['NAXIS1']
        ny = h['NAXIS2']
        #nv = h['NAXIS3']
        self.cs_fixed = None
        self.Rc_fixed = None
        if None in (p := [Mstar, Rc, cs, hdisk, offmajor, offminor, offvsys]):
            m = self.cubemodel(**self.popt)
        else:
            m = self.cubemodel(*p)
        if len(self.v_red) > 0:
            m_blue = m[self.v_valid < np.min(self.v_red)]
        else:
            m_blue = m * 1
            m_red = np.full((0, ny, nx), np.nan)
        if len(self.v_blue) > 0:
            m_red = m[np.max(self.v_blue) < self.v_valid]
        else:
            m_red = m * 1
            m_blue = np.full((0, ny, nx), np.nan)
        nanblue = np.full((len(self.v_nanblue), ny, nx), np.nan)
        nanmid = np.full((len(self.v_nanmid), ny, nx), np.nan)
        nanred = np.full((len(self.v_nanred), ny, nx), np.nan)
        model = nanblue
        if len(m_blue) > 0:
            model = np.append(model, m_blue, axis=0)
        if len(nanmid) > 0:
            model = np.append(model, nanmid, axis=0)
        if len(m_red) > 0:
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
                     hdisk: float,
                     offmajor: float, offminor: float, offvsys: float,
                     filename: str = 'modelmom01.png', pa: float = None):
        d = self.cubemodel(Mstar, Rc, cs, hdisk, offmajor, offminor, offvsys)
        mom1 = makemom01(d, self.v_valid, self.sigma)['mom1']
        levels = np.arange(1, 20) * 6 * self.sigma_mom0
        levels = np.sort(np.r_[-levels, levels])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        vplot = (np.nanpercentile(self.mom1, 95) 
                 - np.nanpercentile(self.mom1, 5)) / 2.
        m = ax.pcolormesh(self.x, self.y, mom1, cmap='jet',
                          vmin=-vplot, vmax=vplot)
        fig.colorbar(m, ax=ax, label=r'Model mom1 (km s$^{-1}$)')
        ax.contour(self.x, self.y, self.mom0, colors='gray', levels=levels)
        if pa is not None:
            r = np.linspace(-1, 1, 10) * self.x.max() * 1.5
            ax.plot(r * self.sinpa, r * self.cospa, 'k:')
            ax.plot(r * self.cospa, -r * self.sinpa,'k:')
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
            ax.plot(r * self.sinpa, r * self.cospa, 'k:')
            ax.plot(r * self.cospa, -r * self.sinpa, 'k:')
        ax.set_xlabel('R.A. offset (au)')
        ax.set_ylabel('Dec. offset (au)')
        ax.set_xlim(self.x.max() * 1.01, self.x.min() * 1.01)
        ax.set_ylim(self.y.min() * 1.01, self.y.max() * 1.01)
        ax.set_aspect(1)
        fig.savefig(filename)
        plt.close()

    def plotresidualmom(self, Mstar: float, Rc: float, cs: float,
                        hdisk: float,
                        offmajor: float, offminor: float, offvsys: float,
                        filename: str = 'residualmom01.png',
                        pa: float = None):
        d = self.cubemodel(Mstar, Rc, cs, hdisk, offmajor, offminor, offvsys)
        mom1 = self.mom1 - makemom01(d, self.v_valid, self.sigma)['mom1']
        levels = np.arange(1, 20) * 6 * self.sigma_mom0
        levels = np.sort(np.r_[-levels, levels])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        vplot = (np.nanpercentile(self.mom1, 95) 
                 - np.nanpercentile(self.mom1, 5)) / 2.
        m = ax.pcolormesh(self.x, self.y, mom1, cmap='jet',
                          vmin=-vplot, vmax=vplot)
        fig.colorbar(m, ax=ax, label=r'Obs. $-$ model mom1 (km s$^{-1}$)')
        ax.contour(self.x, self.y, self.mom0, colors='gray', levels=levels)
        if pa is not None:
            r = np.linspace(-1, 1, 10) * self.x.max() * 1.5
            ax.plot(r * self.sinpa, r * self.cospa, 'k:')
            ax.plot(r * self.cospa, -r * self.sinpa, 'k:')
        ax.set_xlabel('R.A. offset (au)')
        ax.set_ylabel('Dec. offset (au)')
        ax.set_xlim(self.x.max() * 1.01, self.x.min() * 1.01)
        ax.set_ylim(self.y.min() * 1.01, self.y.max() * 1.01)
        ax.set_aspect(1)
        fig.savefig(filename)
        plt.close()

