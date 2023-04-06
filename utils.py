import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import emcee, corner
from multiprocessing import Pool


def gauss1d(x, amp, mean, fwhm):
    return amp * np.exp2(-4. * ((x - mean) / fwhm)**2)


def emcee_corner(bounds, log_prob_fn, args=None, nwalkers_per_ndim=16,
                 nburnin=2000, nsteps=1000, gr_check=False, ndata=1000,
                 labels=None, rangelevel=0.8, figname=None,
                 show_corner=False, ncore=1):
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
    converge = True
    for n in [nburnin, nsteps]:
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
        nsteps *= 2  # (nsteps - nsteps_min) steps are burned in.
    if not converge:
        print('\nWARNING: emcee did not converge (Gelman-Rubin > 1.25).\n')
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
    return [pmid, perr]
