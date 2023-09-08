from twodgrad import TwoDGrad

'-------- INPUTS --------'
cubefits = './twodgrad/IRAS16253_SBLB_C18O_robust_2.0.imsub.fits'
center = '16h28m21.61526785s -24d36m24.32538414s'
pa = 113  # deg
incl = 65  # deg
vsys = 4  # km/s
dist = 139  # pc
sigma = 1.9e-3  # Jy/beam; None means automatic calculation.
cutoff = 5.0  # sigma
tol_kep = 0.25  # beam; tolerance from the major axis
xmax = 1 * dist  # au
ymax = xmax  # au
vmax = 2.5  # km/s
vmin = -vmax  # km/s
xmax_plot = xmax  # au
ymax_plot = xmax_plot  # au
vmax_plot = vmax  # au
vmin_plot = vmax_plot / 50  # au
show_figs = True
minrelerr = 0.01
minabserr = 0.1
method = 'mean'  # mean or gauss
write_point = False  # True: write the 2D centers to a text file.
'------------------------'


'-------- HOW TO DO EACH STEP --------'
if __name__ == '__main__':
    filehead = cubefits.replace('.fits', '')
    tdg = TwoDGrad()
    tdg.get_2Dcenter(cubefits=cubefits, center=center,
                      vsys=vsys, dist=dist, sigma=sigma, cutoff=cutoff,
                      xmax=xmax, ymax=ymax, vmax=vmax, vmin=vmin,
                      minrelerr=minrelerr, minabserr=minabserr, method=method)
    tdg.find_rkep(pa=pa, tol_kep=tol_kep)
    tdg.get_mstar(incl=incl)
    if write_point: tdg.write_2Dcenter(filehead)
    tdg.plot_center(filehead=filehead, xmax=xmax_plot, ymax=ymax_plot,
                     vmax=vmax_plot, vmin=vmin_plot, show_figs=True)
'-------------------------------------'