from channelfit import ChannelFit

'-------- INPUTS --------'
cubefits = './channelfit/testcube.fits'
center = '16h28m21.615s -24d36m23.33s'
pa = 0  # deg
incl = 45  # deg
vsys = 0  # km/s
dist = 140  # pc
sigma = 2e-3  # Jy/beam; None means automatic calculation.
rmax = 1.5 * dist  # au
vlim = (-4.0, -0.6, 0.6, 4.0)  # km/s; from vsys
'------------------------'


'-------- HOW TO DO EACH STEP --------'
filehead = cubefits.replace('.fits', '')
chan = ChannelFit(scaling='uniform', progressbar=True)
chan.makegrid(cubefits=cubefits, center=center, pa=pa, incl=incl,
              vsys=vsys, dist=dist, sigma=sigma, rmax=rmax, vlim=vlim)
chan.fitting(Mstar_range=[0.1, 2.0],
             Rc_range=[30, 300],
             pI_range=[0, 2],
             fixed_params={'cs': 0.4, 'h1': 0, 'h2': -1,
                           'Rin': 0, 'Ienv': 0,
                           'xoff': 0, 'yoff': 0, 'voff': 0,
                           'incloff': 0, 'paoff':0},
             kwargs_emcee_corner={'nwalkers_per_ndim': 4,
                                  'nburnin': 10,
                                  'nsteps': 20,
                                  'rangelevel': 0.99},
             filename=filehead)
p = chan.popt
chan.modeltofits(**p, filehead=filehead)
for s in ['obs', 'model', 'residual']:
    chan.plotmom(mode=s, **p, filename=f'{filehead}.{s}mom01.png')
'-------------------------------------'