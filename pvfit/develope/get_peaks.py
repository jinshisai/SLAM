from scipy.signal import convolve
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


def gaussian1D(x, A, mx, sigx, peak=True):
	'''
	A one-dimensional Gaussian function.

	Parameters
	----------
	x: x value (coordinate)
	A: Amplitude. Not a peak value, but the integrated value.
	mx: mean value
	sigx: standard deviation
	'''
	coeff = A if peak else A/np.sqrt(2.0*np.pi*sigx*sigx)
	expx  = np.exp(-(x-mx)*(x-mx)/(2.0*sigx*sigx))
	gauss = coeff*expx
	return gauss


def get_peaks(signal, noise):
	# Get derivative
	kernel = [1, 0 ,-1]
	derivative = convolve(signal, kernel, 'valid')

	# Check inversion of sign of derivative
	sign  = np.sign(derivative)
	dsign = convolve(sign, kernel, 'valid')

	# Peak candidates
	#  These candidates are basically all negative slope positions
	#  Add one since using 'valid' shrinks the arrays
	candidates = np.where(derivative < 0.)[0] + (len(kernel) - 1)

	# Peaks
	# Filtering candidates
	peaks = sorted(set(candidates).intersection(np.where(dsign == 2)[0] + 1))

	return derivative, dsign



def main():
	# dummy signal
	x = np.linspace(0.,40.,128)
	x0s = [17., 23.]
	amps  = [1., 1.]
	sigxs  = [0.1, 0.1]
	sig_raw = gaussian1D(x, amps[0], x0s[0], sigxs[0]) + gaussian1D(x, amps[1], x0s[1], sigxs[1])
	#sig_raw = np.zeros(len(x))
	noise   = np.random.normal(loc=0.,scale=0.1*amps[0], size=len(x))
	signal  = sig_raw + noise
	rms     = np.std(noise)
	#print (rms)

	peaks, _ = find_peaks(signal, height=3.*rms)
	print (peaks)
	#deriv, dsign = get_peaks(signal, rms)

	plt.plot(x, signal)
	plt.scatter(x[peaks], signal[peaks], marker='x')
	#plt.plot(x[2:-2], dsign)
	plt.show()





if __name__ == '__main__':
	main()