# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Yusuke Aso
# Created Date: 2022 Jan 27
# Updated Date: 2023 Nov 21 by J.Sai
# version = alpha
# ---------------------------------------------------------------------------
"""
This script makes position-velocity diagrams along the major anc minor axes and reproduces their silhouette by the UCM envelope.
The main class PVSilhouette can be imported to do each steps separately.

Note. FITS files with multiple beams are not supported. The dynamic range for xlim_plot and vlim_plot should be >10 for nice tick labels.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
import warnings
from tqdm import tqdm
from utils import emcee_corner, ReadFits
from pvfitting.mockpvd import MockPVD

warnings.simplefilter('ignore', RuntimeWarning)


class PVFitting(ReadFits):

#    def __init__(self):
    
    def put_PV(self, pvmajorfits: str, pvminorfits: str,
               dist: float, vsys: float,
               rmax: float | None,
               vmin: float | None, vmax: float | None,
               sigma: float | None,
               xskip: int = 1, skipto: int | None = None,
               dNsampling: list | None = [5, 1]):
        d = []
        for pvfits in [pvmajorfits, pvminorfits]:
            self.read_pvfits(pvfits=pvfits, dist=dist, vsys=vsys,
                             xmin=-rmax, xmax=rmax, xskip=xskip, sigma=sigma)
            if type(skipto) is int:
                    iskip = int(self.bmin / (np.abs(self.dx) / xskip) / skipto)
                    if iskip == 0:
                        print('WARNING: \'skipto\' is ignored because the beam minor axis is smaller than \'skipto\' pixels.')
                        iskip = 1
                    else:
                        print(f'Adopt xskip={iskip:d}.')
                        ibmaj = self.bmaj / (np.abs(self.dx) * iskip)
                        ibmin = self.bmin / (np.abs(self.dx) * iskip)
                        print(f'Beam major/minor axis is {ibmaj:.1f}/{ibmin:.1f} pixels.')
            else:
                iskip = xskip
            self.vorg = self.v
            self.read_pvfits(pvfits=pvfits, dist=dist, vsys=vsys,
                             xmin=-rmax, xmax=rmax, xskip=iskip, sigma=sigma,
                             vmin=vmin, vmax=vmax)
            if dNsampling is not None:
                self.sampling(dNsampling)
            d.append(self.data)
        self.dpvmajor, self.dpvminor = d

    def sampling(self, steps):
        x_smpl, y_smpl = steps
        x_smpl = int(self.bmin / x_smpl / self.dx )
        y_smpl = int(1. / y_smpl)
        if x_smpl == 0: x_smpl = 1
        if y_smpl == 0: y_smpl = 1
        self.data = self.data[y_smpl//2::y_smpl, x_smpl//2::x_smpl]
        self.v = self.v[y_smpl//2::y_smpl]
        self.x = self.x[x_smpl//2::x_smpl]
        self.dx = self.x[1] - self.x[0]
        self.dv = self.v[1] - self.v[0]
        h = self.header
        h['CRPIX1'] = (h['CRPIX1'] - 1 - self.offpix[0] - x_smpl//2) // x_smpl + 1
        h['CDELT1'] = h['CDELT1'] * x_smpl
        self.header = h
        ibmaj = self.bmaj / self.dx
        ibmin = self.bmin / self.dx
        print(f'Adopt xskip={x_smpl:d} and vskip={y_smpl:d}.')
        print(f'Beam major/minor axis is {ibmaj:.1f}/{ibmin:.1f} pixels.')

    def check_modelgrid(self, nsubgrid: float = 1,
                        n_nest: list | None = None, reslim: float = 5):
        # model grid
        mpvd = MockPVD(self.x, self.x, self.v, 
                       nsubgrid=nsubgrid, nnest=n_nest, 
                       beam=self.beam, reslim=reslim)
        mpvd.grid.gridinfo()

    def fit_mockpvd(self, incl: float = 89.,
                    Mstar_range: list[float, float] = [0.01, 10],
                    Rc_range: list[float, float] = [1., 1000.],
                    alphainfall_range: list[float, float] = [0.0, 1],
                    taumax_range: list[float, float] = [0.1, 1e3],
                    frho_range: list[float, float] = [1., 1e4],
                    sig_mdl_range: list[float, float] = [0., 10.],
                    fixed_params: dict = {'Mstar':None, 'Rc':None,
                                          'alphainfall':None, 'taumax':None,
                                          'frho':None, 'sig_mdl':None},
                    vmask: list[float, float] = [0, 0],
                    zmax: float | None = None,
                    filename: str = 'PVsilhouette',
                    show: bool = False, progressbar: bool = True,
                    kwargs_emcee_corner: dict = {},
                    signmajor: int | None = None, signminor: int | None = None,
                    pa_major: float = 0., pa_minor: float = 90.,
                    linewidth: float | None = None,
                    nsubgrid: int = 1, n_nest: list[float] = [3, 3],
                    reslim: float = 5,
                    title: str | None = None,
                    log: bool = False):
        # Observed PV diagrams
        majobs = self.dpvmajor.copy()
        minobs = self.dpvminor.copy()
        # correction factor for over sampling
        beam_area = np.pi/(4.*np.log(2.)) * self.bmaj * self.bmin # beam area
        Rarea = beam_area / self.dx / self.dx # area ratio


        # grid & mask
        x, v = np.meshgrid(self.x, self.v)
        mask = (vmask[0] < v) * (v < vmask[1])
        majobs = np.where(mask, np.nan, majobs)
        minobs = np.where(mask, np.nan, minobs)
        majsig, minsig = self.sigma, self.sigma

        # get quadrant
        majquad = getquad(self.dpvmajor) if signmajor is None else signmajor
        minquad = getquad(self.dpvminor) * (-1) if signminor is None else signminor

        # model
        if zmax is None:
            z = self.x
        else:
            dx = self.x[1] - self.x[0]
            nz = int(zmax / dx + 0.5)
            z = (np.arange(2 * nz + 1) - nz) * dx
        mpvd = MockPVD(self.x, z, self.v,
                       nsubgrid=nsubgrid, nnest=n_nest,
                       beam=self.beam, reslim=reslim,
                       signmajor=majquad, signminor=minquad,
                       pa_major=pa_major, pa_minor=pa_minor)
        rout = np.max(z)
        def makemodel(Mstar, Rc, alphainfall, taumax, frho):
            major, minor = mpvd.generate_mockpvd(Mstar=Mstar, Rc=Rc,
                                                 alphainfall=alphainfall,
                                                 taumax=taumax, frho=frho,
                                                 incl=incl, linewidth=linewidth,
                                                 rout=rout, axis='both')
            # quadrant
            fflux = (np.nansum(majobs * major) + np.nansum(minobs * minor)) \
                    / (np.nansum(major * major) + np.nansum(minor * minor))
            return fflux * major, fflux * minor
        self.makemodel = makemodel
        # Fitting
        paramkeys = ['Mstar', 'Rc', 'alphainfall', 'taumax', 'frho', 'sig_mdl']
        p_fixed = {k:fixed_params[k] if k in fixed_params else None for k in paramkeys}
        free = {k:p_fixed[k] is None for k in paramkeys}
        p_fixed = np.array([p_fixed[k] for k in paramkeys])
        
        runfit = None in p_fixed
        if runfit:
            notfixed = p_fixed == None
            ilog = np.array([0, 1, 2, 3, 4], dtype=int)
            i = ilog[p_fixed[ilog] != None]
            p_fixed[i] = np.log10(p_fixed[i].astype('float'))
            labels = np.array(['Mstar', 'Rc', r'$\alpha$', r'$\tau_\mathrm{max}$',
                               r'$f_\rho$', r'$\sigma_\mathrm{model}$'])
            labels[ilog] = ['log'+labels[i] for i in ilog]
            labels = labels[notfixed]
            kwargs0 = {'nwalkers_per_ndim':4, 'nburnin':500, 'nsteps':500,
                       'rangelevel':None, 'range_corner':None, 'labels':labels,
                       'figname':filename+'.corner.png', 'show_corner':show,
                       'plot_chain':True, 'show_chain':show}
            kw = dict(kwargs0, **kwargs_emcee_corner)
            # progress bar
            if progressbar:
                total = kw['nwalkers_per_ndim'] * len(p_fixed[notfixed])
                total *= kw['nburnin'] + kw['nsteps']
                bar = tqdm(total=total)
                bar.set_description('Within the ranges')
            # Modified log likelihood
            def lnprob(p):
                if progressbar:
                    bar.update(1)
                # parameter
                q = p_fixed.copy()
                q[notfixed] = p # in linear scale
                q[ilog] = 10**q[ilog]
                # updated sigma
                majsig2 = (1. + q[-1]**2) * majsig**2
                minsig2 = (1. + q[-1]**2) * minsig**2
                # make model
                majmod, minmod = self.makemodel(*q[:-1])
                chi2maj = np.nansum((majobs - majmod)**2 / majsig2 + np.log(majsig2))
                chi2min = np.nansum((minobs - minmod)**2 / minsig2 + np.log(minsig2)) 
                return -0.5 * (chi2maj + chi2min) / np.sqrt(Rarea)
            # prior
            plim = np.array([Mstar_range, Rc_range, alphainfall_range,
                             taumax_range, frho_range, sig_mdl_range])
            plim[ilog] = np.log10(plim[ilog])
            plim = plim[notfixed].T
            if type(r_c := kw['range_corner']) is dict:
                r_c = [r_c[k] if k in r_c else 0.8 for k in paramkeys]
                for i in ilog:
                    r_c[i] = r_c[i] if type(r_c[i]) is float else np.log10(r_c[i])
                r_c = [a for a, k in zip(r_c, paramkeys) if free[k]]
                kw['range_corner'] = r_c

            # run mcmc fitting
            mcmc = emcee_corner(plim, lnprob, simpleoutput=False, **kw)
            # best parameters & errors
            def get_p(i: int):
                p = p_fixed.copy()
                p[notfixed] = mcmc[i]
                p[ilog] = 10**p[ilog]
                return p
            popt = get_p(0)
            plow = get_p(1)
            pmid = get_p(2)
            phigh = get_p(3)
        else:
            def chi2():
                # parameter
                q = p_fixed.copy()
                # updated sigma
                majsig2 = (1. + q[-1]**2) * majsig**2
                minsig2 = (1. + q[-1]**2) * minsig**2
                # make model
                majmod, minmod = self.makemodel(*q[:-1])
                # chi2 does not use log(majsig2) and log(minsig2).
                chi2maj = np.nansum((majobs - majmod)**2 / majsig2)
                chi2min = np.nansum((minobs - minmod)**2 / minsig2)
                return (chi2maj + chi2min) / np.sqrt(Rarea)
            dof = np.prod(np.shape(majobs)) + np.prod(np.shape(minobs))
            # The number of paramter is assumed to be 6 but won't change dof much.
            dof = dof / np.sqrt(Rarea) - 6 - 1
            self.chi2r = chi2() / dof
            
            popt = p_fixed
            plow = p_fixed
            pmid = p_fixed
            phigh = p_fixed
        self.popt = popt
        self.plow = plow
        self.pmid = pmid
        self.phigh = phigh
        ulist = ['Msun', 'au', '', '', '', 'sig_obs']
        digits = [2, 0, 2, 2, 2, 2]
        flist = ['f', 'f', 'f', 'e', 'e', 'f']
        for i, (k, d, u, f) in enumerate(zip(paramkeys, digits, ulist, flist)):
            p = [self.plow[i], self.popt[i], self.phigh[i]]
            print(f'{k} = {p[0]:.{d:d}{f}}, {p[1]:.{d:d}{f}}, {p[2]:.{d:d}{f}} {u}')
        if runfit:
            plist = [self.popt, self.plow, self.pmid, self.phigh]
            with open(filename+'.popt.txt', 'w') as f:
                f.write('#Rows:' + ','.join(paramkeys) + '\n')
                f.write('#Columns:' + ','.join(['popt', 'plow', 'pmid', 'phigh']) + '\n')
                np.savetxt(f, np.transpose(plist))
        self.popt = dict(zip(paramkeys, self.popt))
        self.plow = dict(zip(paramkeys, self.plow))
        self.pmid = dict(zip(paramkeys, self.pmid))
        self.phigh = dict(zip(paramkeys, self.phigh))

        # plot
        self.plot_pvds(filename=filename, color='model', contour='obs', 
                       vmask=vmask, title=title, show=show, log=log)

    def read_fitres(self, f: str):
        '''
        Read fitting result.

        Parameter
        ---------
        f (str): Path to a file containing the fitting result.
        '''
        self.popt, self.plow, self.pmid, self.phigh = np.loadtxt(f).T

    def plot_pvds(self, filename: str = 'PVsilhouette', 
                  color: str = 'model', contour: str = 'obs',
                  vmask: list[float, float] = [0., 0.],
                  cmap: str = 'viridis', cmap_residual: str = 'bwr', ext: str = '.png',
                  title: str | None = None, show: bool = False,
                  shadecolor: str = 'white', clevels: list[float] | None = None,
                  log: bool = False):
        '''
        Plot observed and model PV diagrams.
        '''

        # data
        majobs, minobs = self.dpvmajor.copy(), self.dpvminor.copy()
        # grid/data/mask/sigma
        x, v = np.meshgrid(self.x, self.v)
        mask = (vmask[0] < v) * (v < vmask[1]) # velocity mask
        majsig, minsig = self.sigma, self.sigma

        if 'model' in [color, contour]:
            # check if fitting result exists
            if hasattr(self, 'popt'):
                popt = np.array(list(self.popt.values()))
            else:
                print('ERROR\twriteout_fitres: No optimized parameters are found.')
                print('ERROR\twriteout_fitres: Run fitting or read fitting result first.')
                return 0
            # model pv diagrams
            majmod, minmod = self.makemodel(*popt[:-1])
            # residual
            majres = np.where(mask, -(mask.astype('int')), (majobs - majmod) / majsig)
            minres = np.where(mask, -(mask.astype('int')), (minobs - minmod) / minsig)
            plot_residual = True
            outlabel = '.model'
        else:
            plot_residual = False
            outlabel = '.obs'


        if clevels is None:
            clevels = (2**np.arange(0, 10) if log else np.arange(1, 11)) * 3 * self.sigma
        def makeplots(data_color, data_contour, cmap, vmin=None,
                      vmax=None, vmask=None, alpha=1.):
            # set figure
            fig, axes = plt.subplots(1, 2,)
            fig.set_figheight(3.2)
            fig.set_figwidth(4)
            ax1, ax2 = axes

            # major
            if log:
                vmin_plot = np.log10(2 * self.sigma) if vmin is None else np.log10(vmin)
                vmax_plot = None if vmax is None else np.log10(vmax)
            else:
                vmin_plot = vmin
                vmax_plot = vmax
            d_plot = data_color[0] * 1
            if log:
                d_plot = np.log10(d_plot.clip(vmin, None))
            im = ax1.pcolormesh(self.x, self.v, d_plot, 
                                cmap=cmap, vmin=vmin_plot, vmax=vmax_plot,
                                alpha=alpha, rasterized=True)
            ax1.contour(self.x, self.v, data_contour[0],
                       levels = clevels, colors='k')
            ax1.set_xlabel('Major offset (au)')
            ax1.set_ylabel(r'$V-V_{\rm sys}$ (km s$^{-1}$)')

            # minor
            d_plot = data_color[1] * 1
            if log:
                d_plot = np.log10(d_plot.clip(vmin, None))
            im = ax2.pcolormesh(self.x, self.v, d_plot,
                                cmap=cmap, vmin=vmin_plot, vmax=vmax_plot,
                                alpha=alpha, rasterized=True)
            ax2.contour(self.x, self.v, data_contour[1],
                       levels = clevels, colors='k')
            cax2 = ax2.inset_axes([1.02, 0., 0.05, 1.]) # x0, y0, dx, dy
            cb = plt.colorbar(im, cax=cax2, label=r'Jy beam$^{-1}$')
            if log:
                cbticks = np.outer([1, 2, 5], 10**np.arange(-6, 3, 1.0))
                cbticks = np.log10(np.sort(np.ravel(cbticks)))
                cbticks = cbticks[(vmin_plot < cbticks) * (cbticks < vmax_plot)]
                cb.set_ticks(cbticks)
                cb.set_ticklabels([f'{i:.1e}' for i in 10**cbticks])

            ax2.set_yticklabels('')
            ax2.set_xlabel('Minor offset (au)')

            for ax in axes:
                ax.set_xlim(np.min(self.x), np.max(self.x))
                ax.set_ylim(np.min(self.v), np.max(self.v))

            if vmask is not None:
                _v = self.v[ (self.v > vmask[0]) * (self.v < vmask[1])] # masked velocity ranges
                for ax in axes:
                    ax.fill_between(self.x,
                                    np.full(len(self.x), np.min(_v) - 0.5 * self.dv), 
                                    np.full(len(self.x), np.max(_v) + 0.5 * self.dv),
                                    color=shadecolor, alpha=0.6, edgecolor=None)
            if title is not None:
                fig.suptitle(title, y=0.92)

            return fig


        # color images and model images
        d_col = [majmod, minmod] if color == 'model' else [majobs, minobs]
        d_con = [majmod, minmod] if contour == 'model' else [majobs, minobs]


        figs = []
        # main plot
        vmin = 2 * self.sigma if log else np.nanmin(np.array(d_col))
        vmax = np.nanmax(np.array(d_col))
        vmask = vmask if vmask[1] > vmask[0] else None
        fig = makeplots(d_col, d_con, cmap, vmin=vmin, vmax=vmax,
                        vmask=vmask, alpha=0.8)
        fig.tight_layout()
        fig.savefig(filename + outlabel + ext, dpi=300)
        figs.append(fig)
        if show: plt.show()

        # residual plot
        if plot_residual:
            d_col = [majres, minres] if color == 'model' else [majobs, minobs]
            d_con = [majres, minres] if contour == 'model' else [majobs, minobs]
            vmin, vmax = np.nanmin(np.array(d_col)), np.nanmax(np.array(d_col))
            # plot
            fig2 = makeplots(d_col, d_con, cmap_residual, vmin=vmin, vmax=vmax,
                             vmask=None, alpha=0.5)
            fig2.tight_layout()
            fig2.savefig(filename + '.residual' + ext, dpi=300)
            figs.append(fig2)
            if show: plt.show()

        return figs

    def modeltofits(self, filehead: str = 'best', **kwargs):
        w = wcs.WCS(naxis=2)
        h = self.header
        h['NAXIS1'] = len(self.x)
        nx = h['NAXIS1']
        p = self.popt if kwargs == {} else kwargs 
        if 'sig_mdl' in p.keys():
            del p['sig_mdl']
        majmod, minmod = self.makemodel(**p)
        majres = self.dpvmajor - majmod
        minres = self.dpvminor - minmod
        v_nanblue = self.vorg[self.vorg < np.min(self.v)]
        v_nanred = self.vorg[np.max(self.v) < self.vorg]
        nanblue = np.full((len(v_nanblue), nx), np.nan)
        nanred = np.full((len(v_nanred), nx), np.nan)
        majmod = np.concatenate((nanblue, majmod, nanred), axis=0)
        minmod = np.concatenate((nanblue, minmod, nanred), axis=0)
        majres = np.concatenate((nanblue, majres, nanred), axis=0)
        minres = np.concatenate((nanblue, minres, nanred), axis=0)
        
        def tofits(d: np.ndarray, ext: str):
            header = w.to_header()
            hdu = fits.PrimaryHDU(d, header=header)
            for k in h.keys():
                if not ('COMMENT' in k or 'HISTORY' in k):
                    hdu.header[k]=h[k]
            hdu = fits.HDUList([hdu])
            hdu.writeto(f'{filehead}.{ext}.fits', overwrite=True)
            
        tofits(majmod, 'model.major')
        tofits(minmod, 'model.minor')
        tofits(majres, 'residual.major')
        tofits(minres, 'residual.minor')


def getquad(m):
    '''
    Get quadrant
    '''
    nv, nx = np.shape(m)
    q = np.sum(m[:nv//2, :nx//2]) + np.sum(m[nv//2:, nx//2:]) \
        - np.sum(m[nv//2:, :nx//2]) - np.sum(m[:nv//2, nx//2:])
    return int(np.sign(q))
