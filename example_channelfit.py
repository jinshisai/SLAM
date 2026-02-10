from channelfit import ChannelFit

'-------- INPUTS --------'
cubefits = './testfits/test.cube.fits'
center = '04h39m53.878s +26d03m09.43s'
pa = 2.0  # deg
incl = 85  # deg
vsys = 5.9  # km/s
dist = 140  # pc
sigma = 1.7e-3  # Jy/beam
rmax = 200  # au; The fitted area is [-rmax, ramx] x [-rmax, rmax].
vlim = [-3.6, -2.0, 2.0, 3.6]  # km/s; Relative to vsys.
'------------------------'


'-------- HOW TO DO EACH STEP --------'
filehead = 'test.channelfit'
# scaling can be 'uniform', 'mom0ft', or 'mom0clean'. mom0ft and mom0clean scales the model intensity by the deconvolved moment 0 map obtained by Fourier transform and clean, respectively.
chan = ChannelFit(scaling='uniform', progressbar=True)
chan.makegrid(cubefits=cubefits, center=center, pa=pa, incl=incl,
              vsys=vsys, dist=dist, sigma=sigma, rmax=rmax, vlim=vlim)
chan.fitting(Mstar_range=[0.1, 1.0],  # Msun; stellar mass
             Rc_range=[30, 300],  # au; disk radius
             pI_range=[0, 3],  # radial power-law index for the model intensity
             h1_range=[0, 0.5],  # scale height (H) / axial radius (R)
             h2_range=[0, 0.5],  # scale height (H) / axial radius (R)
             fixed_params={'cs': 0.4,  # km/s; line width
                           'Rin': 0,  # au; innermost radius
                           'Ienv': 0,  # intensity scaling for the >Rc region, relative to the inner region.
                           'xoff': 0, 'yoff': 0, 'voff': 0,  # au, au, km/s; offsets
                           'incloff': 0, 'paoff':0  # deg, deg; offsets
                           },
             kwargs_emcee_corner={'nwalkers_per_ndim': 2,
                                  'nburnin': 100,
                                  'nsteps': 100,
                                  'rangelevel': 0.99},
             filename=filehead)
p = chan.popt
chan.modeltofits(**p, filehead=filehead)
for s in ['obs', 'model', 'residual']:
    chan.plotmom(mode=s, **p, filename=f'{filehead}.{s}mom01.png')
'-------------------------------------'