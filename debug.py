import numpy as np
from pvfit import PVFit



def main():
	# ------- input -------
	fitsfile= 'testfits/l1489.c18o.contsub.gain01.rbp05.mlt100.cf15.pbcor.pv69d.fits'

	outname = 'test'
	rms     = 5.1e-3 # Jy/beam
	thr     = 6.     # 6 sigma
	clevels = np.array([-3,3,6,9,12,15,20,25,30])*rms
	# ---------------------


	# -------- main --------
	impv = PVFit(fitsfile)
	impv.pvfit_xcut(outname, rms, thr)
	impv.plotresults_onpvdiagram(outname='test_xcut', clevels=clevels, colors=['r'])

	impv = PVFit(fitsfile) # just to reflesh results recorded above
	impv.pvfit_vcut(outname, rms, thr)
	impv.plotresults_onpvdiagram(outname='test_vcut', clevels=clevels, colors=['r'])
	# ----------------------



if __name__ == '__main__':
	main()