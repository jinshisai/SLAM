from velgrad import VelGrad

'-------- INPUTS --------'
cubefits = './testfits/test.cube.fits'
center = '04h39m53.878s +26d03m09.43s'
pa = 2.0  # deg
incl = 85  # deg
vsys = 5.9  # km/s
dist = 140  # pc
sigma = 1.7e-3  # Jy/beam
cutoff = 5.0  # sigma
xmax = 200  # au; The fitted area is [-xmax, xamx] x [-ymax, ymax].
ymax = xmax  # au The fitted area is [-xmax, xamx] x [-ymax, ymax].
vmax = 3.6  # km/s; relative to vsys.
vmin = -3.6  # km/s; relative to vsys.
vmask = [-2.0, 2.0]  # km/s; relative to vsys. To exclude low velocity channels.
show_figs = True
minabserr = 0.1  # minimum absolute errorbar in the unit of bmaj or dv.
minrelerr = 0.01  # minimum relative errorbar.
method = 'mean'  # mean or gauss
'------------------------'


'-------- HOW TO DO EACH STEP --------'
filehead = 'test.velgrad'
vg = VelGrad()
vg.read_cubefits(cubefits=cubefits, center=center,
                 vsys=vsys, dist=dist, sigma=sigma,
                 xmin=-xmax, xmax=xmax,
                 ymin=-ymax, ymax=ymax,
                 vmin=vmin, vmax=vmax)
vg.get_2Dcenter(cutoff=cutoff, vmask=vmask,
                minrelerr=minrelerr, minabserr=minabserr, method=method)
vg.filtering(pa0=pa, filename=filehead)
vg.calc_mstar(incl=incl)
vg.plot_center(filehead=filehead, pa=pa, show_figs=show_figs)
'-------------------------------------'