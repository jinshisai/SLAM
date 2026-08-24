'''
Fitting functions
'''

# modules
import numpy as np
# import os
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
from scipy import optimize
# from scipy.stats import norm
# from typing import Callable


# functions
# gaussian
def gauss1d(x: float | np.ndarray, amp: float, mean: float,
            sig: float) -> float | np.ndarray:
    """Evaluate a one-dimensional Gaussian.

    Args:
        x (float or np.ndarray): Coordinates at which to evaluate the model.
        amp (float): Gaussian peak amplitude.
        mean (float): Gaussian center.
        sig (float): Gaussian standard deviation in the same unit as ``x``.

    Returns:
        float or np.ndarray: Gaussian evaluated at ``x``.
    """
    return amp * np.exp(-(x-mean)*(x-mean)/(2.0*sig*sig))


# for chi-square
def chi_gauss1d(param: np.ndarray | list[float], xdata: np.ndarray,
                 ydata: np.ndarray,
                 ysig: float | np.ndarray) -> np.ndarray:
    """Calculate normalized residuals for a one-dimensional Gaussian.

    Args:
        param (np.ndarray or list): Gaussian ``[amplitude, mean, sigma]``.
        xdata (np.ndarray): Coordinates of the observed profile.
        ydata (np.ndarray): Observed intensity at each coordinate.
        ysig (float or np.ndarray): Intensity uncertainty.

    Returns:
        np.ndarray: Residuals divided by ``ysig``.
    """
    return (ydata - (gauss1d(xdata, *param))) / ysig


def edge(xdata: np.ndarray, ydata: np.ndarray, yerr: float,
         threshold: float, goodflag: np.ndarray | None = None,
         edgesign: int = 1) -> list[float]:
    """Locate a threshold-defined edge in a one-dimensional profile.

    Args:
        xdata (np.ndarray): Coordinates of the profile samples.
        ydata (np.ndarray): Intensity at each coordinate.
        yerr (float): RMS intensity uncertainty.
        threshold (float): Minimum accepted intensity.
        goodflag (np.ndarray or None, optional): Boolean mask selecting usable
            samples. None accepts every sample. Defaults to None.
        edgesign (int, optional): Edge direction. A negative value selects the
            first accepted coordinate; zero or a positive value selects the
            last. Defaults to 1.

    Returns:
        list: Edge coordinate and its uncertainty. Both values are NaN when
            no sample passes the threshold and mask.
    """
    grad = (np.roll(ydata, -1) - np.roll(ydata, 1)) / (xdata[2] - xdata[0])
    cond = (ydata > threshold) if goodflag is None else (ydata > threshold) * goodflag
    grad, x = grad[cond], xdata[cond]
    if len(x) == 0:
        return [np.nan, np.nan]
    val = x[0] if edgesign < 0 else x[-1]
    grad = grad[0] if edgesign < 0 else grad[-1]
    err = yerr / np.abs(grad)
    return [val, err]


'''
def ridge_gauss(xdata, ydata, yerr):
    if len(xdata) < 4:
        return [np.nan, np.nan]
    bounds = [[0, np.min(xdata), np.abs(xdata[1] - xdata[0])],
              [np.max(ydata) * 2., np.max(xdata), np.max(xdata)]]
    try:
        popt, pcov = curve_fit(gauss1d, xdata, ydata,
                               sigma = np.full_like(xdata, yerr),
                               absolute_sigma=True, bounds=bounds)
        val, err = popt[1], np.sqrt(pcov[1][1])
    except RuntimeError:
        return [np.nan, np.nan]
    return [val, err]
'''


def gaussfit(xdata: np.ndarray, ydata: np.ndarray,
             yerr: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit a one-dimensional Gaussian by minimizing chi-square residuals.

    Args:
        xdata (np.ndarray): Coordinates of the observed profile.
        ydata (np.ndarray): Observed intensity at each coordinate.
        yerr (float): RMS intensity uncertainty used for every sample.

    Returns:
        tuple: Best-fit ``[amplitude, mean, sigma]`` and their one-sigma
            uncertainties. Arrays contain NaNs when a fit cannot be obtained.
    """

    # Get estimate of the initial parameters
    indx_pini = ydata >= 3.*yerr
    mx = np.nansum(ydata[indx_pini]*xdata[indx_pini])/np.nansum(ydata[indx_pini])  # weighted mean
    sigx = np.sqrt(np.nansum(ydata[indx_pini]*(xdata[indx_pini] - mx)**2.)/np.nansum(ydata[indx_pini]))  # standerd deviation

    # if sigx is too small
    if sigx <= 1e-6:
        sigx = np.abs(xdata[1] - xdata[0])/len(xdata)

    amp = np.nanmax(ydata)
    pinp = [amp, mx, sigx]

    if len(xdata) < 3:
        param_out = np.full(3, np.nan)
        param_err = np.full(3, np.nan)
        return param_out, param_err

    # fitting
    results = optimize.leastsq(chi_gauss1d, pinp, args=(xdata, ydata, yerr), full_output=True)
    param_out = results[0]
    param_cov = results[1]
    # print(results)
    # print(param_out, param_cov)
    # Do not multiply covariance by reduced chi^2 to obtain absolute errors

    # parameter error estimates
    if param_cov is not None:
        param_err = np.array([
            np.abs(param_cov[j][j])**0.5 for j in range(len(pinp))
        ])
    else:
        param_err = np.full(3, np.nan)
        param_out = np.full(3, np.nan) if (param_out == pinp).all else param_out

    # print results
    # print ('Chi2: ', reduced_chi2)
    # print ('Fitting results: amp   mean   sigma')
    # print (amp_fit, mx_fit, sigx_fit)
    # print ('Errors')
    # print (amp_fit_err, mx_fit_err, sigx_fit_err)

    return param_out, param_err


def ridge_mean(xdata: np.ndarray, ydata: np.ndarray,
               yerr: float) -> tuple[float, float]:
    """Locate a profile ridge from its intensity-weighted mean coordinate.

    Args:
        xdata (np.ndarray): Coordinates of the profile samples.
        ydata (np.ndarray): Intensity used as the coordinate weights.
        yerr (float): RMS intensity uncertainty.

    Returns:
        tuple: Intensity-weighted coordinate and its estimated uncertainty.
            Both values are NaN when fewer than two samples are supplied.
    """
    if len(xdata) < 2:
        return np.nan, np.nan
    val = np.average(xdata, weights=ydata)
    err = yerr * np.sqrt(np.sum((xdata - val)**2)) / np.sum(ydata)
    return val, err

'''
# functions
def splaw(r: float | np.ndarray, params: np.ndarray | list[float],
          r0: float = 100.) -> tuple[float | np.ndarray,
                                     float | np.ndarray]:
    """Evaluate a single power law and its radial derivative.

    Args:
        r (float or np.ndarray): Radius at which to evaluate the model.
        params (np.ndarray or list): ``[vsys, v0, p]``, where ``v0`` is the
            velocity at ``r0`` and ``p`` is the power-law index. ``vsys`` is
            retained for use by :func:`chi_splaw` but is not added here.
        r0 (float, optional): Reference radius in the same unit as ``r``.
            Defaults to 100.

    Returns:
        tuple: Power-law velocity and its derivative with respect to radius.
    """
    vsys, v0, p = params
    vout = v0*(r/r0)**(-p)
    dydx = (v0/r0)*(-p)*(r/r0)**(-p-1)
    return vout, dydx
'''
'''
def chi_splaw(params: np.ndarray | list[float], xdata: np.ndarray,
              ydata: np.ndarray, xsig: float | np.ndarray,
              ysig: float | np.ndarray) -> np.ndarray:
    """Calculate normalized residuals for the single power-law model.

    Args:
        params (np.ndarray or list): ``[vsys, v0, p]`` model parameters.
        xdata (np.ndarray): Observed radii.
        ydata (np.ndarray): Observed velocities.
        xsig (float or np.ndarray): Radius uncertainty. Currently unused.
        ysig (float or np.ndarray): Velocity uncertainty.

    Returns:
        np.ndarray: Absolute-velocity residuals divided by ``ysig``.
    """
    vsys = params[0]
    vout, dydx = splaw(xdata, params)
    # sig       = np.sqrt((xsig*dydx)*(xsig*dydx) + ysig*ysig)
    sig = ysig
    chi_out = (np.abs(ydata - vsys) - vout)/sig
    return chi_out
'''
'''
def dplaw(radii: np.ndarray, params: np.ndarray | list[float]
          ) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a double power law and its radial derivative.

    Args:
        radii (np.ndarray): Radii at which to evaluate the model.
        params (np.ndarray or list): ``[v_break, r_break, p_in, p_out]``.
            ``v_break`` is the velocity at ``r_break``; ``p_in`` applies below
            the break and ``p_out`` applies at and above it.

    Returns:
        tuple: Model velocities and their derivatives with respect to radius.
    """
    vb, rb, pin, pout = params

    vout = np.array([vb * (r / rb)**(-pin)
                     if r < rb else
                     vb * (r / rb)**(-pout)
                     for r in radii])
    dydx = np.array([(vb / rb) * (-pin) * (r/rb)**(-pin - 1)
                     if r < rb else
                     (vb / rb) * (-pout) * (r / rb)**(-pout - 1)
                     for r in radii])

    return vout, dydx
'''
'''
def chi_dplaw(params: np.ndarray | list[float], xdata: np.ndarray,
              ydata: np.ndarray, xsig: float | np.ndarray,
              ysig: float | np.ndarray) -> np.ndarray:
    """Calculate normalized residuals for the double power-law model.

    Args:
        params (np.ndarray or list): ``[v_break, r_break, p_in, p_out]``.
        xdata (np.ndarray): Observed radii.
        ydata (np.ndarray): Observed velocities.
        xsig (float or np.ndarray): Radius uncertainty. Currently unused.
        ysig (float or np.ndarray): Velocity uncertainty.

    Returns:
        np.ndarray: Absolute residuals divided by ``ysig``.
    """
    vout, dydx = dplaw(xdata, params)
    # sig       = np.sqrt((xsig*dydx)*(xsig*dydx) + ysig*ysig)
    sig = ysig
    chi_out = (np.abs(ydata - vout))/sig
    return chi_out
'''
r'''
def estimate_perror(params: np.ndarray | list[float],
                    func: Callable[..., np.ndarray],
                    x: np.ndarray, y: np.ndarray,
                    xerr: float | np.ndarray, yerr: float | np.ndarray,
                    niter: int = 3000) -> np.ndarray:
    """Estimate fitting-parameter uncertainties with Monte Carlo trials.

    Args:
        params (np.ndarray or list): Initial fitting parameters.
        func (Callable): Residual function accepted by
            ``scipy.optimize.leastsq``.
        x (np.ndarray): Observed coordinates.
        y (np.ndarray): Observed values.
        xerr (float or np.ndarray): One-sigma coordinate uncertainties.
        yerr (float or np.ndarray): One-sigma value uncertainties.
        niter (int, optional): Number of Monte Carlo trials. Defaults to 3000.

    Returns:
        np.ndarray: Standard deviation of the fitted value of each parameter.

    Notes:
        The function prints the estimated standard deviations and medians and
        writes parameter histograms to ``errest.pdf``.
    """
    nparams = len(params)
    perrors = np.zeros((0, nparams), float)

    for i in range(niter):
        offest = norm.rvs(size=len(x), loc=x, scale=xerr)
        velest = norm.rvs(size=len(y), loc=y, scale=yerr)
        result = optimize.leastsq(func, params, args=(offest, velest, xerr, yerr), full_output=True)
        perrors = np.vstack((perrors, result[0]))
        # print param_esterr[:,0]

    sigmas = np.array([np.std(perrors[:, i]) for i in range(nparams)])
    medians = np.array([np.median(perrors[:, i]) for i in range(nparams)])

    with np.printoptions(precision=4, suppress=True):
        print('Estimated errors (standard deviation):')
        print(sigmas)
        print('Medians:')
        print(medians)

    # plot the Monte-Carlo results
    fig_errest = plt.figure(figsize=(11.69, 8.27), frameon=False)
    gs = GridSpec(nparams, 2)
    if nparams == 3:
        xlabels = [r'$V_\mathrm{sys}$', r'$V_\mathrm{100}$', r'$p$']
    elif nparams == 4:
        xlabels = [r'$V_\mathrm{break}$', r'$R_\mathrm{break}$', r'$p_\mathrm{in}$', r'$p_\mathrm{out}$']
    else:
        xlabels = [r'$p%i$' % (i+1) for i in range(nparams)]

    for i in range(nparams):
        # histogram
        ax1 = fig_errest.add_subplot(gs[i, 0])
        ax1.set_xlabel(xlabels[i])
        if i == (nparams - 1):
            ax1.set_ylabel('Frequency')
        ax1.hist(perrors[:, i], bins=50, cumulative=False)  # density = True

        # cumulative histogram
        ax2 = fig_errest.add_subplot(gs[i, 1])
        ax2.set_xlabel(xlabels[i])
        if i == (nparams - 1):
            ax2.set_ylabel('Cumulative\n frequency')
        ax2.hist(perrors[:, i], bins=50, density=True, cumulative=True)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # plt.show()
    fig_errest.savefig('errest.pdf', transparent=True)
    fig_errest.clf()

    return sigmas
'''