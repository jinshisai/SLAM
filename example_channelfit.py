from channelfit import ChannelFit

'-------- INPUTS --------'
cubefits = './testfits/test.cube.fits'
center = '04h39m53.878s +26d03m09.43s'
pa = 2.0  # deg
incl = 85  # deg
vsys = 5.9  # km/s
dist = 140  # pc
sigma = 1.7e-3  # Jy/beam; None means automatic calculation.
rmax = 200  # au
vlim = [-3.6, -2.0, 2.0, 3.6]  # km/s; from vsys
'------------------------'


'-------- HOW TO DO EACH STEP --------'
filehead = 'test.channelfit'
chan = ChannelFit(scaling='uniform', progressbar=True)
chan.makegrid(cubefits=cubefits, center=center, pa=pa, incl=incl,
              vsys=vsys, dist=dist, sigma=sigma, rmax=rmax, vlim=vlim)
chan.fitting(Mstar_range=[0.1, 1.0],
             Rc_range=[30, 300],
             pI_range=[0, 3],
             h1_range=[0, 0.5],
             h2_range=[0, 0.5],
             fixed_params={'cs': 0.4, 'Rin': 0, 'Ienv': 0,
                           'xoff': 0, 'yoff': 0, 'voff': 0,
                           'incloff': 0, 'paoff':0},
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