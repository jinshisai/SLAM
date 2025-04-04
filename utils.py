import numpy as np
import matplotlib.pyplot as plt
import emcee, corner
from multiprocessing import Pool
from dynesty import DynamicNestedSampler as DNS
from dynesty import NestedSampler as NS
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
from astropy.io import fits
from astropy import constants, units
from astropy.coordinates import SkyCoord


def gauss1d(x, amp, mean, fwhm):
    return amp * np.exp2(-4. * ((x - mean) / fwhm)**2)


def emcee_corner(bounds, log_prob_fn, args: list = [],
                 nwalkers_per_ndim: int = 16,
                 nburnin: int = 2000, nsteps: int = 2000,
                 gr_check: bool = False, ndata: int = 1000,
                 labels: list = None, rangelevel: float = 0.8,
                 range_corner: list | None = None,
                 figname: str = None, show_corner: bool = False,
                 plot_chain: bool = False, show_chain: bool = False,
                 ncore: int = 1, simpleoutput: bool = True, 
                 moves = emcee.moves.StretchMove()):
    ndim = len(bounds[0])
    nwalkers = ndim * nwalkers_per_ndim
    plim = np.array(bounds)

    def lnL(p, *args):
        if np.all((plim[0] < p) * (p < plim[1])):
            return log_prob_fn(p, *args)
        else:
            return -np.inf

    def gelman_rubin(samples):
        nsteps = len(samples[0])
        B = np.std(np.mean(samples, axis=1), axis=0)
        W = np.mean(np.std(samples, axis=1), axis=0)
        V = (nsteps - 1) / nsteps * W + (nwalkers + 1) / (nwalkers - 1) * B
        d = ndata - ndim - 1
        R = np.sqrt((d + 3) / (d + 1) * V / W)
        return R

    p0 = plim[0] + (plim[1] - plim[0]) * np.random.rand(nwalkers, ndim)
    for n in [nburnin, nsteps]:
        converge = True
        if ncore > 1:
            with Pool(ncore) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn,
                                                args=args, pool=pool, moves=moves)
                # The inner function lnL can't be pickled for multiprocessing.
                sampler.run_mcmc(p0, n)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnL,
                                            args=args, moves=moves)
            sampler.run_mcmc(p0, n)
        #samples = sampler.get_chain()  # emcee 3.1.1
        samples = sampler.chain  # emcee 2.2.1
        if gr_check:
            GR = gelman_rubin(samples)
            if np.max(GR) > 1.25:
                converge = False
        p0 = samples[:, -1, :]
    if not converge:
        print('\nWARNING: emcee did not converge (Gelman-Rubin =',
              np.round(GR, 2), '> 1.25).\n')
    #lnp = sampler.get_log_prob()  # emcee 3.1.1
    lnp = sampler.lnprobability  # emcee 2.2.1
    popt = samples[np.unravel_index(np.argmax(lnp), lnp.shape)]
    _samples = samples.copy()
    samples = samples.reshape((-1, ndim))
    plow = np.percentile(samples, 16, axis=0)
    pmid = np.percentile(samples, 50, axis=0)
    phigh = np.percentile(samples, 84, axis=0)
    perr = (phigh - plow) / 2.
    if range_corner is None:
        r_c = [rangelevel] * ndim if rangelevel is not None else np.transpose(bounds)
    else:
        r_c = range_corner
    if show_corner or figname is not None:
        corner.corner(samples, truths=popt,
                      quantiles=[0.16, 0.5, 0.84], show_titles=True,
                      range=r_c, labels=labels),
        if figname is not None:
            plt.savefig(figname)
        if show_corner: plt.show()
        plt.close()

    if plot_chain:
        fig, axes = plt.subplots(ndim, 1, sharex=True)
        xplot = np.arange(0, nsteps, 1)
        for i, ax in enumerate(axes):
            for iwalk in range(nwalkers):
                ax.plot(xplot, _samples[iwalk,:,i].T, 'k')
            ax.set_ylabel(labels[i])
            ax.tick_params(which='both', direction='in',
                           bottom=True, top=True,
                           left=True, right=True,
                           labelbottom=False)
        axes[0].set_xlim(0, nsteps)
        axes[-1].tick_params(labelbottom=True)
        axes[-1].set_xlabel('Step number')
        if figname is not None:
            if '.corner' in figname:
                fname = figname.replace('.corner', '.chains')
            else:
                fname = figname.replace('.png', '.chains.png')
            fig.savefig(fname)
        if show_chain:
            plt.show()
        plt.close()

    if simpleoutput:
        return [pmid, perr]
    else:
        return [popt, plow, pmid, phigh]


def dynesty_corner(bounds, 
    log_prob_fn, args: list = [],
    labels: list = None, 
    figname: str = None, 
    show_corner: bool = False,
    return_evidence: bool = False,
    simpleoutput: bool = True, 
    wt_kwargs = None):
    # dimensions
    ndim = len(bounds[0])
    plim = np.array(bounds)

    # likelihood/prior
    lnlike = lambda p: log_prob_fn(p, *args)
    ptform = lambda u: plim[0] + (plim[1] - plim[0]) * u

    # Static nested sampling
    #sampler = NS(lnlike, ptform, ndim)
    #sampler.run_nested(print_progress=False)
    #sresults = sampler.results
    # Dynamic nested sampling.
    dsampler = DNS(lnlike, ptform, ndim)
    dsampler.run_nested(print_progress=False, wt_kwargs=wt_kwargs)
    #dresults = dsampler.results
    results = dsampler.results
    #results = dyfunc.merge_runs([sresults, dresults])
    if (figname is not None) & (show_corner == True):
        cfig, caxes = dyplot.cornerplot(results, labels=labels, quantiles=[0.16, 0.5, 0.84])
        if figname is not None: cfig.savefig(figname)
        if show_corner:
            plt.show()
        else:
            plt.close()
    # Compute 16%--84% quantiles
    weights = results.importance_weights()
    quantiles = [dyfunc.quantile(samps, [0.16, 0.5, 0.84], weights=weights)
    for samps in results.samples.T]

    # Evidence?
    if return_evidence:
        evidence = np.exp(results.logz[-1])
        evid_err = evidence * results.logzerr[-1]
        print(f'Evidence: {evidence:.2e} +/- {evid_err:.2e}')

    # Output
    if simpleoutput:
        pmid = np.array(quantiles).T[1]
        perr = np.array([(q[2] - q[0]) * 0.5 for q in quantiles])
        return [pmid, perr]
    else:
        popt = results.samples[np.argmax(results.logl), :] # highest probability
        plow, pmid, phigh = np.array(quantiles).T
        return [popt, plow, pmid, phigh]


class ReadFits():
    def read_cubefits(self, cubefits: str, center: str | None = None,
                      dist: float = 1, vsys: float = 0,
                      xmin: float | None = None, xmax: float | None = None,
                      ymin: float | None = None, ymax: float | None = None,
                      vmin: float | None = None, vmax: float | None = None,
                      xskip: int = 1, yskip: int = 1,
                      sigma: float | None = None) -> dict:
        """Read channel maps in the FITS format.

        Args:
            cubefits (str): Name of the input FITS file including the extension.
            center (str | None, optional): Coordinates of the target: e.g., "01h23m45.6s 01d23m45.6s". Defaults to None.
            dist (float, optional): Distance of the target in the unit of pc, used to convert arcsec to au. Defaults to 1.
            vsys (float, optional): Systemic velocity of the target in the unit of km/s. Defaults to 0.
            xmin (float | None, optional): The x-axis is limited to (xmin, xmax) in the unit of au. Defaults to None.
            xmax (float | None, optional): The x-axis is limited to (xmin, xmax) in the unit of au. Defaults to None.
            ymin (float | None, optional): The y-axis is limited to (ymin, ymax) in the unit of au. Defaults to None.
            ymax (float | None, optional): The y-axis is limited to (ymin, ymax) in the unit of au. Defaults to None.
            vmin (float | None, optional): The velocity axis is limited to (vmin, vmax) in the unit of km/s. Defaults to None.
            vmax (float | None, optional): The velocity axis is limited to (vmin, vmax) in the unit of km/s. Defaults to None.
            xskip (int, optional): Skip xskip pixels in the x axis. Defaults to 1.
            yskip (int, optional): Skip yskip pixels in the y axis. Defaults to 1.
            sigma (float | None, optional): Standard deviation of the FITS data. None means automatic. Defaults to None.

        Returns:
            dict: x (1D array), y (1D array), v (1D array), data (2D array), header, and sigma.
        """
        cc = constants.c.si.value
        f = fits.open(cubefits)[0]
        d, h = np.squeeze(f.data), f.header
        if center is None:
            cx, cy = 0, 0
        else:
            c0 = SkyCoord('00h00m00s 00d00m00s', frame='icrs')
            c1 = [h['CRVAL1'] * units.degree, h['CRVAL2'] * units.degree]
            c1 = c0.spherical_offsets_by(*c1)
            c3 = c1.spherical_offsets_to(SkyCoord(center, frame='icrs'))
            cx = c3[0].degree
            cy = c3[1].degree
        if sigma is None:
            sigma = np.mean([np.nanstd(d[:2]), np.std(d[-2:])])
            print(f'sigma = {sigma:.3e}')
        x = (np.arange(h['NAXIS1']) - h['CRPIX1'] + 1) * h['CDELT1']
        y = (np.arange(h['NAXIS2']) - h['CRPIX2'] + 1) * h['CDELT2']
        v = (np.arange(h['NAXIS3']) - h['CRPIX3'] + 1) * h['CDELT3']
        crpix = int(h['CRPIX1']) - 1
        startpix = crpix % xskip
        x = x[startpix::xskip]
        h['CRPIX1'] = (crpix - startpix) // xskip + 1
        h['CDELT1'] = h['CDELT1'] * xskip
        d = d[:, :, startpix::xskip]
        crpix = int(h['CRPIX2']) - 1
        startpix = crpix % yskip
        y = y[startpix::yskip]
        h['CRPIX2'] = (crpix - startpix) // yskip + 1
        h['CDELT2'] = h['CDELT2'] * yskip
        d = d[:, startpix::yskip, :]
        v = v + h['CRVAL3']
        x = (x - cx) * 3600. * dist  # au
        y = (y - cy) * 3600. * dist  # au
        if h['CUNIT3'] == 'Hz':
            if 'RESTFRQ' in h:
                restfreq = h['RESTFRQ']
            elif 'RESTFREQ' in h:
                restfreq = h['RESTFREQ']
            else:
                restfreq = np.mean(v)
                print('No rest frequency found. The middle frequency adopted.')
            v = (1. - v / restfreq) * cc / 1.e3 - vsys  # km/s
        elif h['CUNIT3'] == 'm/s':
            v = v * 1e-3 - vsys
        i0 = 0 if xmax is None else np.argmin(np.abs(x - xmax))
        i1 = len(x) - 1 if xmin is None else np.argmin(np.abs(x - xmin))
        x = x[i0:i1 + 1]
        j0 = 0 if ymin is None else np.argmin(np.abs(y - ymin))
        j1 = len(y) - 1 if ymax is None else np.argmin(np.abs(y - ymax))
        y = y[j0:j1 + 1]
        k0 = 0 if vmin is None else np.argmin(np.abs(v - vmin))
        k1 = len(v) - 1 if vmax is None else np.argmin(np.abs(v - vmax))
        v = v[k0:k1 + 1]
        d =  d[k0:k1 + 1, j0:j1 + 1, i0:i1 + 1]
        self.offpix = (i0, j0, k0)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dv = v[1] - v[0]
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
        self.beam = np.array([bmaj, bmin, bpa])
        self.cubefits, self.dist, self.vsys = cubefits, dist, vsys
        return {'x':x, 'y':y, 'v':v, 'data':d, 'header':h, 'sigma':sigma}

    def read_pvfits(self, pvfits: str,
                    dist: float = 1, vsys: float = 0,
                    xmin: float | None = None, xmax: float | None = None,
                    vmin: float | None = None, vmax: float | None = None,
                    xskip: int = 1,
                    sigma: float | None = None) -> dict:
        """Read a position-velocity diagram in the FITS format.

        Args:
            pvfits (str): Name of the input FITS file including the extension.
            dist (float, optional): Distance of the target in the unit of pc, used to convert arcsec to au. Defaults to 1.
            vsys (float, optional): Systemic velocity of the target in the unit of km/s. Defaults to 0.
            xmin (float | None, optional): The positional axis is limited to (xmin, xmax) in the unit of au. Defaults to None.
            xmax (float | None, optional): The positional axis is limited to (xmin, xmax) in the unit of au. Defaults to None.
            vmin (float | None, optional): The velocity axis is limited to (vmin, vmax) in the unit of km/s. Defaults to None.
            vmax (float | None, optional): The velocity axis is limited to (vmin, vmax) in the unit of km/s. Defaults to None.
            xskip (int, optional): Skip xskip pixels in the x axis. Defaults to 1.
            sigma (float | None, optional): Standard deviation of the FITS data. None means automatic. Defaults to None.

        Returns:
            dict: The keys are x (1D array), v (1D array), data (2D array), header, and sigma.
        """
        cc = constants.c.si.value
        f = fits.open(pvfits)[0]
        d, h = np.squeeze(f.data), f.header
        if sigma is None:
            sigma = np.mean([np.std(d[:2, 10:-10]), np.std(d[-2:, 10:-10]),
                             np.std(d[2:-2, :10]), np.std(d[2:-2, -10:])])
            print(f'sigma = {sigma:.3e}')
        x = (np.arange(h['NAXIS1']) - h['CRPIX1'] + 1) * h['CDELT1']
        v = (np.arange(h['NAXIS2']) - h['CRPIX2'] + 1) * h['CDELT2']
        crpix = int(h['CRPIX1']) - 1
        startpix = crpix % xskip
        x = x[startpix::xskip]
        h['CRPIX1'] = (crpix - startpix) // xskip + 1
        h['CDELT1'] = h['CDELT1'] * xskip
        d = d[:, startpix::xskip]
        v = v + h['CRVAL2']
        x = (x + h['CRVAL1']) * dist  # au
        if h['CUNIT2'] == 'Hz':
            if 'RESTFRQ' in h:
                restfreq = h['RESTFRQ']
            elif 'RESTFREQ' in h:
                restfreq = h['RESTFREQ']
            else:
                restfreq = np.mean(v)
                print('No rest frequency found. The middle frequency adopted.')
            v = (1. - v / restfreq) * cc / 1.e3 - vsys  # km/s
        elif h['CUNIT2'] == 'm/s':
            v = v * 1e-3 - vsys
        i0 = 0 if xmin is None else np.argmin(np.abs(x - xmin))
        i1 = len(x) - 1 if xmax is None else np.argmin(np.abs(x - xmax))
        x = x[i0:i1 + 1]
        k0 = 0 if vmin is None else np.argmin(np.abs(v - vmin))
        k1 = len(v) - 1 if vmax is None else np.argmin(np.abs(v - vmax))
        v = v[k0:k1 + 1]
        d = d[k0:k1 + 1, i0:i1 + 1]
        self.offpix = (i0, k0)
        dx = x[1] - x[0]
        dv = v[1] - v[0]
        if 'BMAJ' in h.keys():
            bmaj = h['BMAJ'] * 3600. * dist  # au
            bmin = h['BMIN'] * 3600. * dist  # au
            bpa = h['BPA']  # deg
        else:
            bmaj, bmin, bpa = dx, dx, 0
            print('No valid beam in the FITS file.')
        self.x, self.dx, self.nx = x, dx, len(x)
        self.v, self.dv, self.nv = v, dv, len(v)
        self.data, self.header, self.sigma = d, h, sigma
        self.bmaj, self.bmin, self.bpa = bmaj, bmin, bpa
        self.beam = np.array([bmaj, bmin, bpa])
        self.pvfits, self.dist, self.vsys = pvfits, dist, vsys
        return {'x':x, 'v':v, 'data':d, 'header':h, 'sigma':sigma}

def rot(x, y, pa):
    s = x * np.cos(pa) - y * np.sin(pa)  # along minor axis
    t = x * np.sin(pa) + y * np.cos(pa)  # along major axis
    return np.array([s, t])
