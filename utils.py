import numpy as np
import matplotlib.pyplot as plt
import emcee, corner
from multiprocessing import Pool
from dynesty import DynamicNestedSampler as DNS
from dynesty import NestedSampler as NS
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot


def gauss1d(x, amp, mean, fwhm):
    return amp * np.exp2(-4. * ((x - mean) / fwhm)**2)


def emcee_corner(bounds, log_prob_fn, args: list = [],
                 nwalkers_per_ndim: int = 16,
                 nburnin: int = 100, nsteps: int = 10000,
                 gr_check: bool = False, ndata: int = 1000,
                 labels: list = None, rangelevel: float = 0.8,
                 figname: str = None, show_corner: bool = False,
                 ncore: int = 1,
                 simpleoutput: bool = True):
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
        return np.min(R)

    p0 = plim[0] + (plim[1] - plim[0]) * np.random.rand(nwalkers, ndim)
    for n in [nburnin, nsteps]:
        converge = True
        if ncore > 1:
            with Pool(ncore) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn,
                                                args=args, pool=pool)
                # The inner function lnL can't be pickled for multiprocessing.
                sampler.run_mcmc(p0, n)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnL, args=args)
            sampler.run_mcmc(p0, n)
        #samples = sampler.get_chain()  # emcee 3.1.1
        samples = sampler.chain  # emcee 2.2.1
        if gr_check:
            GR = gelman_rubin(samples)
            if GR > 1.25:
                converge = False
        p0 = samples[:, -1, :]
    if not converge:
        print(f'\nWARNING: emcee did not converge (Gelman-Rubin={GR:.2f} > 1.25).\n')
    #lnp = sampler.get_log_prob()  # emcee 3.1.1
    lnp = sampler.lnprobability  # emcee 2.2.1
    popt = samples[np.unravel_index(np.argmax(lnp), lnp.shape)]
    samples = samples.reshape((-1, ndim))
    plow = np.percentile(samples, 16, axis=0)
    pmid = np.percentile(samples, 50, axis=0)
    phigh = np.percentile(samples, 84, axis=0)
    perr = (phigh - plow) / 2.
    cornerrange = [rangelevel] * ndim if rangelevel is not None else np.transpose(bounds)
    if show_corner or (not figname is None):
        corner.corner(samples, truths=popt,
                      quantiles=[0.16, 0.5, 0.84], show_titles=True,
                      range=cornerrange, labels=labels),
        if not figname is None: plt.savefig(figname)
        if show_corner: plt.show()
        plt.close()
    if simpleoutput:
        return [pmid, perr]
    else:
        return [popt, plow, pmid, phigh]



def dynesty_corner(bounds, 
    log_prob_fn, args: list = [],
    labels: list = None, 
    rangelevel: float = 0.8,
    figname: str = None, 
    show_corner: bool = False,
    ncore: int = 1,
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