import numpy as np
import matplotlib.pyplot as plt
from pvanalysis import PVAnalysis
from pvanalysis.pvplot import PVPlot


def main():
    # ------- input -------
    '''
    fitsfile = 'testfits/l1489.c18o.contsub.gain01.rbp05.mlt100.cf15.pbcor.pv69d.fits'
    vsys    = 7.3
    incl    = 73.
    pa      = 69.
    fitsfile = 'testfits/tmc1a.c18o.contsub.rbp05.mlt100.clean.majpv.fits'
    vsys    = 6.4
    incl    = 65
    pa      = 72
    fitsfile = 'testfits/l1527.c18o.contsub.rbp05.mlt100.clean.majpv.fits'
    vsys    = 5.8
    incl    = 85
    pa      = 0.
    '''
    fitsfile = 'testfits/TMC1A_C18O_t2000klam.image.pv_invert.fits'
    vsys    = 6.4
    incl    = 48.
    pa      = None


    outname = 'test'
    rms     = 1.7e-3 # Jy/beam
    thr     = 5.     # 6 sigma
    dist    = 140.
    clevels = np.array([-3,3,6,9,12,15,20,25,30])
    Mlim = [0, 10],
    xlim = [-200 / 140, 0, 0, 200 / 140]
    vlim = np.array([-5, 0, 0, 5]) + vsys
    # ---------------------


    # -------- main --------
    impv = PVAnalysis(fitsfile, rms, vsys, dist, pa=pa)
    impv.get_edgeridge(outname, thr=thr, mode='mean', incl=incl,
                       use_position=True, use_velocity=True,
                       Mlim=Mlim, xlim=xlim, vlim=vlim,
                       interp_ridge=False)
    impv.plotresults_pvdiagram(clevels=clevels*rms)
    impv.plotresults_rvplane()

    print ('Fitting edge/ridge.')
    impv.fit_edgeridge(include_vsys=False, include_dp=True,
                       include_pin=False,
                       filehead='testfit', show_corner=False)
    impv.output_fitresult()
    impv.write_edgeridge()

    impv.plot_fitresult(vlim=[6/20, 6], xlim=[200/20, 200],
                        clevels=clevels, outname=outname, show=True)
    # ----------------------



if __name__ == '__main__':
    main()
