from channelfit import ChannelFit

'-------- INPUTS --------'
cubefits = './channelfit/test3D.fits'
center = '04h04m43.07s 26d18m56.20s'
pa = 45  # deg
incl = 45  # deg
vsys = 0  # km/s
dist = 140  # pc
sigma = 1.5e-3  # Jy/beam; None means automatic calculation.
rmax = 0.5 * dist  # au
vlim = (-5.0, -3.0, 3.0, 5.0)  # km/s; from vsys
'------------------------'


'-------- HOW TO DO EACH STEP --------'
if __name__ == '__main__':
    filehead = cubefits.replace('.fits', '')
    chan = ChannelFit(scaling='uniform', progressbar=True)
    chan.makegrid(cubefits=cubefits, center=center, pa=pa, incl=incl,
                  vsys=vsys, dist=dist, sigma=sigma, rmax=rmax, vlim=vlim)
    chan.fitting(Mstar_range=[0.03, 3.0],
                 cs_range=[0.0, 3.0],
                 fixed_params={'Rc': 100, 'h1': 0, 'h2': -1, 'Rin': 0,
                               'pI': 0, 'Ienv': 0,
                               'xoff': 0, 'yoff': 0, 'voff': 0, 'incl': 0},
                 kwargs_emcee_corner={'nwalkers_per_ndim': 4,
                                      'nburnin': 100,
                                      'nsteps': 200,
                                      'rangelevel': 0.99},
                 filename=filehead)
    p = chan.popt
    chan.modeltofits(**p, filehead=filehead)
    for s in ['obs', 'model', 'residual']:
        chan.plotmom(mode=s, **p, filename=f'{filehead}.{s}mom01.png')
'-------------------------------------'