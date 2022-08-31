'''
Fitting functions
'''

# modules
import numpy as np
#import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import optimize
from scipy.stats import norm, uniform




# functions
# gaussian
def gauss1d(x,amp,mean,sig):
	return amp * np.exp(-(x-mean)*(x-mean)/(2.0*sig*sig))

# for chi-square
def chi_gauss1d(param, xdata, ydata, ysig):
	return (ydata - (gauss1d(xdata, *param))) / ysig

def edge(xdata, ydata, yerr, threshold, goodflag=None, edgesign=1):
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

def gaussfit(xdata, ydata, yerr, pini=[]):
	'''
	Gaussian fit through chi-square fit.
	'''

	# Get estimate of the initial parameters
	if len(pini) == 3:
		pinp = pini
	else:
		indx_pini = ydata >= 3.*yerr
		mx    = np.nansum(ydata[indx_pini]*xdata[indx_pini])/np.nansum(ydata[indx_pini]) # weighted mean
		sigx  = np.sqrt(np.nansum(ydata[indx_pini]*(xdata[indx_pini] - mx)**2.)/np.nansum(ydata[indx_pini])) # standerd deviation

		# if sigx is too small
		if sigx <= 1e-6:
			sigx = np.abs(xdata[1] - xdata[0])/len(xdata)

		amp  = np.nanmax(ydata)
		pinp = [amp, mx, sigx]

	if len(xdata) < 3:
		param_out = np.full(3, np.nan)
		param_err = np.full(3, np.nan)
		return param_out, param_err

	# fitting
	results   = optimize.leastsq(chi_gauss1d, pinp, args=(xdata, ydata, yerr), full_output=True)
	param_out = results[0]
	param_cov = results[1]
	#print(results)
	#print(param_out, param_cov)
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
	#print ('Chi2: ', reduced_chi2)
	#print ('Fitting results: amp   mean   sigma')
	#print (amp_fit, mx_fit, sigx_fit)
	#print ('Errors')
	#print (amp_fit_err, mx_fit_err, sigx_fit_err)

	return param_out, param_err

def complexgauss1d(k, peak, k0, sig, phase):
	return peak * np.exp(-(k-k0)*(k-k0)/(2.0*sig*sig)) * np.exp(-1.j*phase*k*2.*np.pi)

def chi_complexgauss1d(param, xdata, ydata, ysig):
	ymodel = complexgauss1d(xdata, *param)
	complex_chi2 = ((ydata.real - ymodel.real)**2.\
	 + (ydata.imag - ymodel.imag)**2.)/np.abs(ysig)**2.
	return np.sqrt(complex_chi2)

def complexgaussfit(xdata, ydata, yerr, pini=[]):
	'''
	Gaussian fit through chi-square fit.
	'''

	# Get estimate of the initial parameters
	nparams = 4
	if len(pini) == nparams:
		pinp = pini
	else:
		indx_pini = ydata >= 3.*yerr
		mx    = np.nansum(ydata[indx_pini]*xdata[indx_pini])/np.nansum(ydata[indx_pini]) # weighted mean
		sigx  = np.sqrt(np.nansum(ydata[indx_pini]*(xdata[indx_pini] - mx)**2.)/np.nansum(ydata[indx_pini])) # standerd deviation

		# if sigx is too small
		if sigx <= 1e-6:
			sigx = np.abs(xdata[1] - xdata[0])/len(xdata)

		amp   = np.nanmax(ydata)
		phase = np.arctan2(ydata.real[np.nanargmin(np.abs(xdata))], ydata.imag[np.nanargmin(np.abs(xdata))])
		pinp  = [amp, mx, sigx, phase]

	if len(xdata) < nparams:
		param_out = np.full(nparams, np.nan)
		param_err = np.full(nparams, np.nan)
		return param_out, param_err

	# fitting
	results   = optimize.leastsq(chi_complexgauss1d, pinp, args=(xdata, ydata, yerr), full_output=True)
	param_out = results[0]
	param_cov = results[1]
	#print(results)
	#print(param_out, param_cov)
	# Do not multiply covariance by reduced chi^2 to obtain absolute errors

	# parameter error estimates
	if param_cov is not None:
		param_err = np.array([
			np.abs(param_cov[j][j])**0.5 for j in range(len(pinp))
			])
	else:
		param_err = np.full(nparams, np.nan)
		param_out = np.full(nparams, np.nan) if (param_out == pinp).all else param_out

	# print results
	#print ('Chi2: ', reduced_chi2)
	#print ('Fitting results: amp   mean   sigma')
	#print (amp_fit, mx_fit, sigx_fit)
	#print ('Errors')
	#print (amp_fit_err, mx_fit_err, sigx_fit_err)

	return param_out, param_err


def ridge_mean(xdata, ydata, yerr):
    if len(xdata) < 2:
        return np.nan, np.nan
    val = np.average(xdata, weights=ydata)
    err = yerr * np.sqrt(np.sum((xdata - val)**2)) / np.sum(ydata)
    return val, err


# functions
def splaw(r, params, r0=100.):
	'''
	Single power-law function.

	Args:
	 - r: radius (au or any)
	 - params: [vsys, v0, p]
	 - r0: 100 (au)
	'''
	vsys, v0, p = params
	vout        = v0*(r/r0)**(-p)
	dydx        = (v0/r0)*(-p)*(r/r0)**(-p-1)
	return vout, dydx

def chi_splaw(params, xdata, ydata, xsig, ysig):
	'''
	Calculate chi for chi-square for the single power-law function.
	'''
	vsys       = params[0]
	vout, dydx = splaw(xdata, params)
	#sig       = np.sqrt((xsig*dydx)*(xsig*dydx) + ysig*ysig)
	sig        = ysig
	chi_out    = (np.abs(ydata - vsys) - vout)/sig
	return chi_out


def dplaw(radii, params):
	'''
	Double power-law function.

	Args:
	 - r: radius (au or any)
	 - params: [vb, rb, pin, pout]
	  - vb: rotational velocity at rb [km/s]
	  - rb: break point raidus at which powers change [au]
	  - pin: power at r < rb
	  - pout: power at r >= rb
	'''
	vb, rb, pin, pout = params

	vout = np.array([
		vb*(r/rb)**(-pin) if r < rb else vb*(r/rb)**(-pout)
		for r in radii
		])
	dydx = np.array([
		(vb/rb)*(-pin)*(r/rb)**(-pin-1) if r < rb else (vb/rb)*(-pout)*(r/rb)**(-pout-1)
		for r in radii
		])

	return vout, dydx


def chi_dplaw(params, xdata, ydata, xsig, ysig):
	'''
	Chi for chi square for the double power-law function.

	Args:
	 - func: a function.
	 - params: input parameters for the function.
	 - xdata: x of data
	 - ydata: y of data
	 - xsig: sigma for x
	 - ysig: sigma for y
	'''
	vout, dydx = dplaw(xdata, params)
	#sig       = np.sqrt((xsig*dydx)*(xsig*dydx) + ysig*ysig)
	sig        = ysig
	chi_out    = (np.abs(ydata - vout))/sig
	return chi_out


def estimate_perror(params, func, x, y, xerr, yerr, niter=3000):
	'''
	Estimate fitting-parameter errors by Monte-Carlo method.

	'''
	nparams = len(params)
	perrors = np.zeros((0,nparams), float)

	for i in range(niter):
		offest = norm.rvs(size = len(x), loc = x, scale = xerr)
		velest = norm.rvs(size = len(y), loc = y, scale = yerr)
		result = optimize.leastsq(func, params, args=(offest, velest, xerr, yerr), full_output = True)
		perrors = np.vstack((perrors, result[0]))
		#print param_esterr[:,0]

	sigmas = np.array([
		np.std(perrors[:,i]) for i in range(nparams)
		])
	medians = np.array([
		np.median(perrors[:,i]) for i in range(nparams)
		])


	with np.printoptions(precision=4, suppress=True):
		print ('Estimated errors (standard deviation):')
		print (sigmas)
		print ('Medians:')
		print (medians)


	# plot the Monte-Carlo results
	fig_errest = plt.figure(figsize=(11.69,8.27), frameon = False)
	gs         = GridSpec(nparams, 2)
	if nparams == 3:
		xlabels    = [r'$V_\mathrm{sys}$', r'$V_\mathrm{100}$', r'$p$' ]
	elif nparams == 4:
		xlabels    = [r'$V_\mathrm{break}$', r'$R_\mathrm{break}$', r'$p_\mathrm{in}$', r'$p_\mathrm{out}$']
	else:
		xlabels = [r'$p%i$'%(i+1) for i in range(nparams)]

	for i in range(nparams):
		# histogram
		ax1 = fig_errest.add_subplot(gs[i,0])
		ax1.set_xlabel(xlabels[i])
		if i == (nparams - 1):
			ax1.set_ylabel('Frequency')
		ax1.hist(perrors[:,i], bins = 50, cumulative = False) # density = True

		# cumulative histogram
		ax2 = fig_errest.add_subplot(gs[i,1])
		ax2.set_xlabel(xlabels[i])
		if i == (nparams - 1):
			ax2.set_ylabel('Cumulative\n frequency')
		ax2.hist(perrors[:,i], bins = 50, density=True, cumulative = True)

	plt.subplots_adjust(wspace=0.4, hspace=0.4)
	#plt.show()
	fig_errest.savefig(outname + '_errest.pdf', transparent=True)
	fig_errest.clf()

	return sigmas