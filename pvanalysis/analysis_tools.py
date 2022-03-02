'''
Analysis tools for PVfit

'''

import sys
import numpy as np
#import scipy.optimize
from scipy.optimize import curve_fit
#import matplotlib.pyplot as plt

sys.path.append('../')
from utils import gauss1d

def edge(xdata, ydata, yerr, threshold, goodflag, edgesign):
    grad = (np.roll(ydata, -1) - np.roll(ydata, 1)) / (xdata[2] - xdata[0])
    cond = (ydata > threshold) * goodflag
    grad, x = grad[cond], xdata[cond]
    if len(x) == 0:
        return [np.nan, np.nan]
    val = x[0] if edgesign < 0 else x[-1]
    grad = grad[0] if edgesign < 0 else grad[-1]
    err = yerr / np.abs(grad)
    return [val, err]


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

def ridge_mean(xdata, ydata, yerr):
    if len(xdata) < 2:
        return [np.nan, np.nan]
    val = np.average(xdata, weights=ydata)
    err = yerr * np.abs(np.sum(xdata - val)) / np.sum(ydata)
    return [val, err]
    
def p_inout(p_in, dp, t0, t1):
    return p_in + dp * (1 + np.sign(t0 - t1)) / 2.

def doublepower_v(r, r_break, v_break, p_in, dp, vsys):
    r_s, r_a = np.sign(r), np.abs(r)
    p = p_inout(p_in, dp, r_a, r_break)
    return v_break * r_s / (r_a / r_break)**p + vsys

def doublepower_v_error(r, r_break, v_break, p_in, dp, vsys,
                        dr_break, dv_break, dp_in, ddp, dvsys):
    p_out = p_in + dp
    dp_out = np.sqrt(dp_in**2 + ddp**2)
    r_a = np.abs(r)
    v0 = doublepower_v(r, r_break, v_break, p_in, dp, vsys=0)
    p = p_inout(p_in, dp, r_a, r_break)
    perr = p_inout(dp_in, dp_out - dp_in, r_a, r_break)
    err2 = ((dv_break / v_break)**2 + (dr_break / r_break * p)**2
            + (np.log(r_break / r_a) * perr)**2) * v0**2 + dvsys**2
    return np.sqrt(err2)

def doublepower_r(v, r_break, v_break, p_in, dp, vsys):
    v_s, v_a = np.sign(v - vsys), np.abs(v - vsys)
    p = p_inout(p_in, dp, v_break, v_a)
    return r_break * v_s / (v_a / v_break)**(1 / p)

def doublepower_r_error(v, r_break, v_break, p_in, dp, vsys,
                        dr_break, dv_break, dp_in, ddp, dvsys):
    p_out = p_in + dp
    dp_out = np.sqrt(dp_in**2 + ddp**2)
    v_a = np.abs(v - vsys)
    r0 = doublepower_r(v, r_break, v_break, p_in, dp, vsys)
    p = p_inout(p_in, dp, v_break, v_a)
    perr = p_inout(dp_in, dp_out - dp_in, v_break, v_a)
    err2 = (dr_break / r_break)**2 + (dv_break / v_break / p)**2 \
           + (np.log(v_break / v_a) * perr / p**2)**2 + (dvsys / v_a / p)**2
    return np.sqrt(err2) * r0



# read file
def read_pvfitres(fname, inner_threshold=None, outer_threshold=None, toau=True, dist=140.):
	# read files
	offset, velocity, offerr, velerr = np.genfromtxt(fname, comments='#', unpack = True)

	# offset threshold of used data point
	if inner_threshold:
		thrindx  = np.where(np.abs(offset) >= inner_threshold)
		offset   = offset[thrindx]
		velocity = velocity[thrindx]
		offerr   = offerr[thrindx]
		velerr   = velerr[thrindx]

	# offset threshold of used data point
	if outer_threshold:
		thrindx  = np.where(np.abs(offset) <= outer_threshold)
		offset   = offset[thrindx]
		velocity = velocity[thrindx]
		offerr   = offerr[thrindx]
		velerr   = velerr[thrindx]

	if toau:
		offset = offset*dist
		offerr = offerr*dist

	return offset, velocity, offerr, velerr
