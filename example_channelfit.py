from channelfit import ChannelFit

'-------- INPUTS --------'
cubefits = './channelfit/IRAS16253_SBLB_C18O_robust_2.0.imsub.fits'
center = '16h28m21.61526785s -24d36m24.32538414s'
pa = 113 - 180  # deg; pa is redshift rotation, pa + 90 is blueshifted infall
incl = 65  # deg
vsys = 4  # km/s
dist = 139  # pc
sigma = 2e-3  # Jy/beam; None means automatic calculation.
rmax = 1 * dist  # au
vlim = (-2.52, -0.9, 0.9, 2.52)  # km/s
show_figs = True
'------------------------'


'-------- HOW TO DO EACH STEP --------'
if __name__ == '__main__':
    filehead = cubefits.replace('.fits', '')
    chan = ChannelFit()
    chan.gridondisk(cubefits=cubefits, center=center, pa=pa, incl=incl,
                    vsys=vsys, dist=dist, sigma=sigma,
                    rmax=rmax, vlim=vlim)
    chan.fitting(Mstar_range=[0.01, 10.0],
                 Rc_range=[5, 500],
                 cs_range=[0.2, 2],
                 offmajor_fixed=0, offminor_fixed=0, offvsys_fixed=0,
                 filename=filehead)
    chan.modeltofits(filehead=filehead)
    #p = chan.popt
    #chan.plotmodelmom(*p, pa=pa, filename=filehead+'.modelmom01.png')
    #chan.plotobsmom(pa=pa, filename=filehead+'.obsmom01.png')
    #chan.plotresidualmom(*p, pa=pa, filename=filehead+'.residualmom01.png')
'-------------------------------------'