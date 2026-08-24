'''
Analysis tools for PVfit

'''

#import sys
import numpy as np
# import scipy.optimize
# from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt

#sys.path.append('../')
#from utils import gauss1d

'''
def edge(xdata: np.ndarray, ydata: np.ndarray, yerr: float,
         threshold: float, goodflag: np.ndarray,
         edgesign: int) -> list[float]:
    """Locate a threshold-defined edge in a one-dimensional profile.

    Args:
        xdata (np.ndarray): Coordinates of the profile samples.
        ydata (np.ndarray): Intensity at each coordinate.
        yerr (float): RMS intensity uncertainty.
        threshold (float): Minimum accepted intensity.
        goodflag (np.ndarray): Boolean mask selecting usable samples.
        edgesign (int): Edge direction. A negative value selects the first
            accepted coordinate; zero or a positive value selects the last.

    Returns:
        list: Edge coordinate and its uncertainty. Both values are NaN when
            no sample passes the threshold and mask.
    """
    grad = (np.roll(ydata, -1) - np.roll(ydata, 1)) / (xdata[2] - xdata[0])
    cond = (ydata > threshold) * goodflag
    grad, x = grad[cond], xdata[cond]
    if len(x) == 0:
        return [np.nan, np.nan]
    val = x[0] if edgesign < 0 else x[-1]
    grad = grad[0] if edgesign < 0 else grad[-1]
    err = yerr / np.abs(grad)
    return [val, err]
'''
'''
def ridge_gauss(xdata: np.ndarray, ydata: np.ndarray,
                yerr: float) -> list[float]:
    """Locate a profile ridge by fitting a one-dimensional Gaussian.

    Args:
        xdata (np.ndarray): Coordinates of the profile samples.
        ydata (np.ndarray): Intensity at each coordinate.
        yerr (float): RMS intensity uncertainty used for every sample.

    Returns:
        list: Fitted Gaussian center and its one-sigma uncertainty. Both
            values are NaN when fewer than four samples are supplied or the
            fit does not converge.
    """
    if len(xdata) < 4:
        return [np.nan, np.nan]
    bounds = [[0, np.min(xdata), np.abs(xdata[1] - xdata[0])],
              [np.max(ydata) * 2., np.max(xdata), np.max(xdata)]]
    try:
        popt, pcov = curve_fit(gauss1d, xdata, ydata,
                               sigma=np.full_like(xdata, yerr),
                               absolute_sigma=True, bounds=bounds)
        val, err = popt[1], np.sqrt(pcov[1][1])
    except RuntimeError:
        return [np.nan, np.nan]
    return [val, err]
'''
'''
def ridge_mean(xdata: np.ndarray, ydata: np.ndarray,
               yerr: float) -> list[float]:
    """Locate a profile ridge from its intensity-weighted mean coordinate.

    Args:
        xdata (np.ndarray): Coordinates of the profile samples.
        ydata (np.ndarray): Intensity used as the coordinate weights.
        yerr (float): RMS intensity uncertainty.

    Returns:
        list: Intensity-weighted coordinate and its estimated uncertainty.
            Both values are NaN when fewer than two samples are supplied.
    """
    if len(xdata) < 2:
        return [np.nan, np.nan]
    val = np.average(xdata, weights=ydata)
    err = yerr * np.abs(np.sum(xdata - val)) / np.sum(ydata)
    return [val, err]
'''

def p_inout(p_in: float, dp: float, t0: float | np.ndarray,
            t1: float | np.ndarray) -> float | np.ndarray:
    """Select the inner or outer index of a broken power law.

    Args:
        p_in (float): Inner power-law index.
        dp (float): Outer index minus the inner index.
        t0 (float or np.ndarray): First values used to select the index.
        t1 (float or np.ndarray): Second values used to select the index.

    Returns:
        float or np.ndarray: ``p_in`` where ``t0 < t1`` and ``p_in + dp``
            where ``t0 > t1``. Equality gives ``p_in + dp / 2``.
    """
    return p_in + dp * (1 + np.sign(t0 - t1)) / 2.


def doublepower_v(r: float | np.ndarray, r_break: float, v_break: float,
                  p_in: float, dp: float,
                  vsys: float) -> float | np.ndarray:
    """Evaluate velocity as a double power law of signed position.

    Args:
        r (float or np.ndarray): Signed position or radius.
        r_break (float): Break radius in the same unit as ``r``.
        v_break (float): Absolute rotation velocity at ``r_break``.
        p_in (float): Power-law index inside ``r_break``.
        dp (float): Outer index minus ``p_in``.
        vsys (float): Systemic-velocity offset.

    Returns:
        float or np.ndarray: Signed model velocity including ``vsys``.
    """
    r_s, r_a = np.sign(r), np.abs(r)
    p = p_inout(p_in, dp, r_a, r_break)
    return v_break * r_s / (r_a / r_break)**p + vsys


def doublepower_v_error(r: float | np.ndarray, r_break: float,
                        v_break: float, p_in: float, dp: float, vsys: float,
                        dr_break: float, dv_break: float, dp_in: float,
                        ddp: float, dvsys: float) -> float | np.ndarray:
    """Propagate parameter uncertainties to double-power-law velocity.

    Args:
        r (float or np.ndarray): Signed position or radius.
        r_break (float): Break radius.
        v_break (float): Absolute rotation velocity at ``r_break``.
        p_in (float): Power-law index inside ``r_break``.
        dp (float): Outer index minus ``p_in``.
        vsys (float): Systemic-velocity offset. This value does not affect the
            propagated uncertainty.
        dr_break (float): Uncertainty of ``r_break``.
        dv_break (float): Uncertainty of ``v_break``.
        dp_in (float): Uncertainty of ``p_in``.
        ddp (float): Uncertainty of ``dp``.
        dvsys (float): Uncertainty of ``vsys``.

    Returns:
        float or np.ndarray: Propagated one-sigma velocity uncertainty,
            neglecting parameter covariances.
    """
    # p_out = p_in + dp
    dp_out = np.sqrt(dp_in**2 + ddp**2)
    r_a = np.abs(r)
    v0 = doublepower_v(r, r_break, v_break, p_in, dp, vsys=0)
    p = p_inout(p_in, dp, r_a, r_break)
    perr = p_inout(dp_in, dp_out - dp_in, r_a, r_break)
    err2 = ((dv_break / v_break)**2 + (dr_break / r_break * p)**2
            + (np.log(r_break / r_a) * perr)**2) * v0**2 + dvsys**2
    return np.sqrt(err2)


def doublepower_r(v: float | np.ndarray, r_break: float, v_break: float,
                  p_in: float, dp: float,
                  vsys: float) -> float | np.ndarray:
    """Evaluate signed position by inverting the double power law.

    Args:
        v (float or np.ndarray): Velocity including the systemic offset.
        r_break (float): Break radius.
        v_break (float): Absolute rotation velocity at ``r_break``.
        p_in (float): Power-law index inside ``r_break``.
        dp (float): Outer index minus ``p_in``.
        vsys (float): Systemic-velocity offset.

    Returns:
        float or np.ndarray: Signed model position or radius.
    """
    v_s, v_a = np.sign(v - vsys), np.abs(v - vsys)
    p = p_inout(p_in, dp, v_break, v_a)
    return r_break * v_s / (v_a / v_break)**(1 / p)


def doublepower_r_error(v: float | np.ndarray, r_break: float,
                        v_break: float, p_in: float, dp: float, vsys: float,
                        dr_break: float, dv_break: float, dp_in: float,
                        ddp: float, dvsys: float) -> float | np.ndarray:
    """Propagate parameter uncertainties to inverted power-law position.

    Args:
        v (float or np.ndarray): Velocity including the systemic offset.
        r_break (float): Break radius.
        v_break (float): Absolute rotation velocity at ``r_break``.
        p_in (float): Power-law index inside ``r_break``.
        dp (float): Outer index minus ``p_in``.
        vsys (float): Systemic-velocity offset.
        dr_break (float): Uncertainty of ``r_break``.
        dv_break (float): Uncertainty of ``v_break``.
        dp_in (float): Uncertainty of ``p_in``.
        ddp (float): Uncertainty of ``dp``.
        dvsys (float): Uncertainty of ``vsys``.

    Returns:
        float or np.ndarray: Propagated one-sigma position uncertainty,
            neglecting parameter covariances.
    """
    # p_out = p_in + dp
    dp_out = np.sqrt(dp_in**2 + ddp**2)
    v_a = np.abs(v - vsys)
    r0 = doublepower_r(v, r_break, v_break, p_in, dp, vsys)
    p = p_inout(p_in, dp, v_break, v_a)
    perr = p_inout(dp_in, dp_out - dp_in, v_break, v_a)
    err2 = (dr_break / r_break)**2 + (dv_break / v_break / p)**2 \
        + (np.log(v_break / v_a) * perr / p**2)**2 + (dvsys / v_a / p)**2
    return np.sqrt(err2) * r0

'''
def read_pvfitres(fname: str, inner_threshold: float | None = None,
                  outer_threshold: float | None = None,
                  toau: bool = False, dist: float = 140.
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read edge or ridge points from a PV-analysis result file.

    Args:
        fname (str): Input text file containing offset, offset uncertainty,
            velocity, and velocity uncertainty columns.
        inner_threshold (float or None, optional): Minimum absolute offset to
            retain. Defaults to None.
        outer_threshold (float or None, optional): Maximum absolute offset to
            retain. Defaults to None.
        toau (bool, optional): Whether to convert offsets from arcseconds to au
            by multiplying by ``dist``. Defaults to False.
        dist (float, optional): Distance in pc used for the arcsecond-to-au
            conversion. Defaults to 140.

    Returns:
        tuple: Arrays of offset, velocity, offset uncertainty, and velocity
            uncertainty, in that order.
    """
    # read files
    offset, offerr, velocity, velerr = np.genfromtxt(fname, comments='#', unpack=True)

    # offset threshold of used data point
    if inner_threshold:
        thrindx = np.where(np.abs(offset) >= inner_threshold)
        offset = offset[thrindx]
        velocity = velocity[thrindx]
        offerr = offerr[thrindx]
        velerr = velerr[thrindx]

    # offset threshold of used data point
    if outer_threshold:
        thrindx = np.where(np.abs(offset) <= outer_threshold)
        offset = offset[thrindx]
        velocity = velocity[thrindx]
        offerr = offerr[thrindx]
        velerr = velerr[thrindx]

    if toau:
        offset = offset*dist
        offerr = offerr*dist

    return offset, velocity, offerr, velerr
'''