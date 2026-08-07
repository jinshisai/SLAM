import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from collections.abc import Callable
from multiprocessing import Pool
from dynesty import DynamicNestedSampler as DNS
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
from astropy.io import fits
from astropy import constants, units
from astropy.coordinates import SkyCoord


def gauss1d(x: float | np.ndarray, amp: float, mean: float,
            fwhm: float) -> float | np.ndarray:
    """Evaluate a one-dimensional Gaussian parameterized by its FWHM.

    Args:
        x (float or np.ndarray): Coordinates at which to evaluate the model.
        amp (float): Gaussian peak amplitude.
        mean (float): Gaussian center in the same unit as ``x``.
        fwhm (float): Full width at half maximum in the same unit as ``x``.

    Returns:
        float or np.ndarray: Gaussian evaluated at ``x``.
    """
    return amp * np.exp2(-4. * ((x - mean) / fwhm)**2)


def emcee_corner(bounds: list[list[float]] | np.ndarray,
                 log_prob_fn: Callable[..., float],
                 args: list[object] = [],
                 nwalkers_per_ndim: int = 16,
                 nburnin: int = 2000, nsteps: int = 2000,
                 gr_check: bool = False, ndata: int = 1000,
                 labels: list[str] | None = None,
                 rangelevel: float | None = 0.8,
                 range_corner: list[float | tuple[float, float]] | None = None,
                 figname: str | None = None, show_corner: bool = False,
                 plot_chain: bool = False, show_chain: bool = False,
                 ncore: int = 1, simpleoutput: bool = True,
                 return_chain: bool = False,
                 return_lnp: bool = False,
                 moves: emcee.moves.Move = emcee.moves.StretchMove()
                 ) -> list[np.ndarray]:
    """Sample a bounded posterior with emcee and optionally plot the result.

    Args:
        bounds (list or np.ndarray): Lower and upper parameter bounds, with
            shape ``(2, ndim)``.
        log_prob_fn (callable): Function returning the log probability for a
            parameter vector followed by the values in ``args``.
        args (list, optional): Additional positional arguments passed to
            ``log_prob_fn``. Defaults to an empty list.
        nwalkers_per_ndim (int, optional): Number of walkers per fitted
            dimension. Defaults to 16.
        nburnin (int, optional): Number of burn-in steps. Defaults to 2000.
        nsteps (int, optional): Number of production steps. Defaults to 2000.
        gr_check (bool, optional): Whether to evaluate the Gelman--Rubin
            convergence statistic. Defaults to False.
        ndata (int, optional): Number of data points used in the convergence
            correction. Defaults to 1000.
        labels (list or None, optional): Parameter labels for the corner and
            chain plots. Defaults to None.
        rangelevel (float or None, optional): Fraction of samples shown for
            every parameter in the corner plot. None uses ``bounds``.
            Defaults to 0.8.
        range_corner (list or None, optional): Per-parameter corner-plot
            ranges. Each entry is either a sample fraction or an explicit
            ``(minimum, maximum)`` pair. Defaults to None.
        figname (str or None, optional): Corner-plot output filename. When
            chain plotting is enabled, it is also used to derive the chain
            filename. Defaults to None.
        show_corner (bool, optional): Whether to display the corner plot.
            Defaults to False.
        plot_chain (bool, optional): Whether to create a walker-chain plot.
            Defaults to False.
        show_chain (bool, optional): Whether to display the chain plot.
            Defaults to False.
        ncore (int, optional): Number of worker processes. Values greater than
            one enable multiprocessing. Defaults to 1.
        simpleoutput (bool, optional): Return median values and symmetric
            uncertainties instead of maximum-probability values and three
            percentiles. Defaults to True.
        return_chain (bool, optional): Append the flattened parameter chains
            to the output. Defaults to False.
        return_lnp (bool, optional): Append flattened log probabilities to
            the output. Defaults to False.
        moves (emcee.moves.Move, optional): emcee proposal move. Defaults to
            :class:`emcee.moves.StretchMove`.

    Returns:
        list: Fit summaries as NumPy arrays. The first two entries are
            ``[median, uncertainty]`` when ``simpleoutput`` is True; otherwise
            they begin ``[maximum_probability, percentile_16, median,
            percentile_84]``. Requested chains and log probabilities follow.

    Notes:
        Parameter bounds are enforced by the local log-probability wrapper in
        serial execution. The underlying emcee sampler determines the precise
        chain layout.
    """
    ndim = len(bounds[0])
    nwalkers = ndim * nwalkers_per_ndim
    plim = np.array(bounds)

    def lnL(p: np.ndarray, *args: object) -> float:
        """Evaluate the log probability inside the parameter bounds."""
        if np.all((plim[0] < p) * (p < plim[1])):
            return log_prob_fn(p, *args)
        else:
            return -np.inf

    def gelman_rubin(samples: np.ndarray) -> np.ndarray:
        """Estimate the corrected Gelman--Rubin statistic by parameter."""
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
        # samples = sampler.get_chain()  # emcee 3.1.1
        samples = sampler.chain  # emcee 2.2.1
        if gr_check:
            GR = gelman_rubin(samples)
            if np.max(GR) > 1.25:
                converge = False
        p0 = samples[:, -1, :]
    if not converge:
        print('\nWARNING: emcee did not converge (Gelman-Rubin =',
              np.round(GR, 2), '> 1.25).\n')
    # lnp = sampler.get_log_prob()  # emcee 3.1.1
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
        if show_corner:
            plt.show()
        plt.close()

    if plot_chain:
        fig, axes = plt.subplots(ndim, 1, sharex=True)
        xplot = np.arange(0, nsteps, 1)
        for i, ax in enumerate(axes):
            for iwalk in range(nwalkers):
                ax.plot(xplot, _samples[iwalk, :, i].T, 'k')
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
        output = [pmid, perr]
    else:
        output = [popt, plow, pmid, phigh]
    if return_chain:
        output.append(samples.T)
    if return_lnp:
        output.append(lnp.reshape(-1))
    return output


def dynesty_corner(bounds: list[list[float]] | np.ndarray,
                   log_prob_fn: Callable[..., float],
                   args: list[object] = [],
                   labels: list[str] | None = None,
                   figname: str | None = None,
                   show_corner: bool = False,
                   return_evidence: bool = False,
                   simpleoutput: bool = True,
                   wt_kwargs: dict[str, object] | None = None
                   ) -> list[np.ndarray]:
    """Sample a bounded likelihood with dynamic nested sampling.

    Args:
        bounds (list or np.ndarray): Lower and upper parameter bounds, with
            shape ``(2, ndim)``.
        log_prob_fn (callable): Log-likelihood function accepting a parameter
            vector followed by the values in ``args``.
        args (list, optional): Additional positional arguments passed to
            ``log_prob_fn``. Defaults to an empty list.
        labels (list or None, optional): Parameter labels for the corner plot.
            Defaults to None.
        figname (str or None, optional): Corner-plot output filename. Defaults
            to None.
        show_corner (bool, optional): Whether to display the corner plot. A
            corner plot is created only when ``figname`` is also supplied.
            Defaults to False.
        return_evidence (bool, optional): Whether to calculate and print the
            Bayesian evidence and its uncertainty. Defaults to False.
        simpleoutput (bool, optional): Return median values and symmetric
            uncertainties instead of the maximum-likelihood values and three
            percentiles. Defaults to True.
        wt_kwargs (dict or None, optional): Weight-function options forwarded
            to :meth:`dynesty.DynamicNestedSampler.run_nested`. Defaults to
            None.

    Returns:
        list: ``[median, uncertainty]`` when ``simpleoutput`` is True;
            otherwise ``[maximum_likelihood, percentile_16, median,
            percentile_84]``. Every entry is a NumPy array with one value per
            fitted parameter.
    """
    # dimensions
    ndim = len(bounds[0])
    plim = np.array(bounds)

    # likelihood/prior
    def lnlike(p: np.ndarray) -> float:
        """Evaluate the supplied log likelihood."""
        return log_prob_fn(p, *args)

    def ptform(u: np.ndarray) -> np.ndarray:
        """Transform a unit-cube sample to the bounded parameter space."""
        return plim[0] + (plim[1] - plim[0]) * u

    # Static nested sampling
    # sampler = NS(lnlike, ptform, ndim)
    # sampler.run_nested(print_progress=False)
    # sresults = sampler.results
    # Dynamic nested sampling.
    dsampler = DNS(lnlike, ptform, ndim)
    dsampler.run_nested(print_progress=False, wt_kwargs=wt_kwargs)
    # dresults = dsampler.results
    results = dsampler.results
    # results = dyfunc.merge_runs([sresults, dresults])
    if (figname is not None) & show_corner:
        cfig, caxes = dyplot.cornerplot(results, labels=labels, quantiles=[0.16, 0.5, 0.84])
        if figname is not None:
            cfig.savefig(figname)
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
        popt = results.samples[np.argmax(results.logl), :]  # highest probability
        plow, pmid, phigh = np.array(quantiles).T
        return [popt, plow, pmid, phigh]


class ReadFits:
    """Provide FITS-reading methods and store the resulting data as attributes."""

    def read_cubefits(self, cubefits: str, center: str | None = None,
                      dist: float = 1, vsys: float = 0,
                      xmin: float | None = None, xmax: float | None = None,
                      ymin: float | None = None, ymax: float | None = None,
                      vmin: float | None = None, vmax: float | None = None,
                      xskip: int = 1, yskip: int = 1,
                      sigma: float | None = None
                      ) -> dict[str, np.ndarray | fits.Header | float]:
        """Read channel maps in the FITS format.

        Args:
            cubefits (str): Input channel-map FITS file.
            center (str or None, optional): Sky coordinates of the target, for
                example, ``"01h23m45.6s 01d23m45.6s"``. Defaults to None.
            dist (float, optional): Source distance in pc, used to convert
                arcseconds to au. Defaults to 1.
            vsys (float, optional): Systemic velocity in km/s. Defaults to 0.
            xmin (float or None, optional): Minimum x coordinate in au.
                Defaults to None.
            xmax (float or None, optional): Maximum x coordinate in au.
                Defaults to None.
            ymin (float or None, optional): Minimum y coordinate in au.
                Defaults to None.
            ymax (float or None, optional): Maximum y coordinate in au.
                Defaults to None.
            vmin (float or None, optional): Minimum velocity relative to
                ``vsys`` in km/s. Defaults to None.
            vmax (float or None, optional): Maximum velocity relative to
                ``vsys`` in km/s. Defaults to None.
            xskip (int, optional): Positive pixel stride along the x axis.
                The returned header is updated to describe the subsampled
                axis. Defaults to 1.
            yskip (int, optional): Positive pixel stride along the y axis.
                The returned header is updated to describe the subsampled
                axis. Defaults to 1.
            sigma (float or None, optional): RMS noise of the FITS data. None
                means automatic estimation. Defaults to None.

        Returns:
            dict: Coordinates ``x`` and ``y`` in au, velocity ``v`` relative
                to ``vsys`` in km/s, the ``(v, y, x)`` data cube, the updated
                FITS ``header``, and RMS noise ``sigma``.

        Notes:
            The selected coordinates, sampling intervals, data, beam, header,
            noise, source distance, systemic velocity, and crop offsets are
            also stored as attributes of this instance. Right ascension axes
            commonly have a negative FITS pixel increment; the x limits are
            applied in the corresponding array order.
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
        d = d[k0:k1 + 1, j0:j1 + 1, i0:i1 + 1]
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
        return {'x': x, 'y': y, 'v': v, 'data': d, 'header': h, 'sigma': sigma}

    def read_pvfits(self, pvfits: str,
                    dist: float = 1, vsys: float = 0,
                    xmin: float | None = None, xmax: float | None = None,
                    vmin: float | None = None, vmax: float | None = None,
                    xskip: int = 1,
                    sigma: float | None = None
                    ) -> dict[str, np.ndarray | fits.Header | float]:
        """Read a position-velocity diagram in the FITS format.

        Args:
            pvfits (str): Input position-velocity FITS file.
            dist (float, optional): Source distance in pc, used to convert the
                position axis from arcseconds to au. Defaults to 1.
            vsys (float, optional): Systemic velocity in km/s. Defaults to 0.
            xmin (float or None, optional): Minimum position in au. Defaults
                to None.
            xmax (float or None, optional): Maximum position in au. Defaults
                to None.
            vmin (float or None, optional): Minimum velocity relative to
                ``vsys`` in km/s. Defaults to None.
            vmax (float or None, optional): Maximum velocity relative to
                ``vsys`` in km/s. Defaults to None.
            xskip (int, optional): Positive pixel stride along the position
                axis. The returned header is updated to describe the
                subsampled axis. Defaults to 1.
            sigma (float or None, optional): RMS noise of the FITS data. None
                means automatic estimation from its edges. Defaults to None.

        Returns:
            dict: Position ``x`` in au, velocity ``v`` relative to ``vsys`` in
                km/s, the ``(v, x)`` data array, the updated FITS ``header``,
                and RMS noise ``sigma``.

        Notes:
            The selected coordinates, sampling intervals, data, beam, header,
            noise, source distance, systemic velocity, and crop offsets are
            also stored as attributes of this instance.
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
        return {'x': x, 'v': v, 'data': d, 'header': h, 'sigma': sigma}


def rot(x: float | np.ndarray, y: float | np.ndarray,
        pa: float) -> np.ndarray:
    """Rotate Cartesian coordinates onto minor and major axes.

    Args:
        x (float or np.ndarray): First Cartesian coordinate.
        y (float or np.ndarray): Second Cartesian coordinate, broadcastable
            with ``x``.
        pa (float): Counterclockwise rotation angle in radians.

    Returns:
        np.ndarray: Rotated minor- and major-axis coordinates ``[s, t]``. Its
            leading dimension has length two.
    """
    s = x * np.cos(pa) - y * np.sin(pa)  # along minor axis
    t = x * np.sin(pa) + y * np.cos(pa)  # along major axis
    return np.array([s, t])
