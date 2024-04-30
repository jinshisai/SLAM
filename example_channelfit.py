from channelfit import ChannelFit

'-------- INPUTS --------'
cubefits = './channelfit/IRAS16253_SBLB_C18O_robust_2.0.imsub.fits'
center = '16h28m21.61526785s -24d36m24.32538414s'
pa = 113 - 180  # deg
incl = 65  # deg
vsys = 4  # km/s
dist = 139  # pc
sigma = 2e-3  # Jy/beam; None means automatic calculation.
rmax = 1 * dist  # au
vlim = (-2.52, -0.9, 0.9, 2.52)  # km/s; from vsys
'------------------------'


'-------- HOW TO DO EACH STEP --------'
if __name__ == '__main__':
    filehead = cubefits.replace('.fits', '')
    chan = ChannelFit(scaling='mom0ft', progressbar=True)
    chan.makegrid(cubefits=cubefits, center=center, pa=pa, incl=incl,
                  vsys=vsys, dist=dist, sigma=sigma, rmax=rmax, vlim=vlim,
                  autoskip=True,  # autoskip=True will resample the pixels if the beam minor axis > 10 pixels.
                  )
    chan.fitting(Mstar_range=[0.01, 1.0],
                 #Mstar_fixed=0.5,
                 Rc_range=[3, 300],
                 #Rc_fixed=100,
                 pI_range=[0.0, 3.0],
                 #pI_fixed=0,
                 cs_fixed=0.5,
                 h1_fixed=0,
                 h2_fixed=-1,  # h2_fixed<0 means to use h1 only.
                 Rin_fixed=0,
                 Ienv_fixed=0,
                 xoff_fixed=0,
                 yoff_fixed=0,
                 voff_fixed=0,
                 incl_fixed=0,
                 kwargs_emcee_corner={'nwalkers_per_ndim':2,
                                      'nburnin':100,
                                      'nsteps':100,
                                      'rangelevel':None}, 
                 filename=filehead)
    p = chan.popt
    chan.modeltofits(**p, filehead=filehead)
    for s in ['obs', 'model', 'residual']:
        chan.plotmom(mode=s, **p, filename=f'{filehead}.{s}mom01.png')
'-------------------------------------'