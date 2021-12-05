# coding: utf-8


### ploting offset vs velocity along 1-D direction
### to derive rotation power law
### chi-square



### import modules
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.optimize
from scipy.stats import norm, uniform
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter




### setting for figures
plt.rcParams['font.family'] = 'Arial'    # font (Times New Roman, Helvetica, Arial)
plt.rcParams['xtick.direction'] = 'in'  # directions of x ticks ('in'), ('out') or ('inout')
plt.rcParams['ytick.direction'] = 'in'  # directions of y ticks ('in'), ('out') or ('inout')
#plt.rcParams['xtick.major.width'] = 1.0 # x ticks width
#plt.rcParams['ytick.major.width'] = 1.0 # y ticks width
plt.rcParams['font.size'] = 14           # fontsize
#plt.rcParams['axes.linewidth'] = 1.0    # edge linewidth



### setting parameters
Ggrav  = 6.67408e-8   # gravitational constant [dyn cm^2 g^-2]
Msun   = 1.9884e33    # solar mass [g]
clight = 2.99792458e10 # light speed [cm s^-1]
auTOkm = 1.50e8       # AU --> km
auTOcm = 1.50e13      # AU --> cm
pi     = np.pi




### defining functions
# chi in chi square method
def chiSPlaw(param, xdata, ydata, xsig, ysig):
    Vsys      = param[0]
    func,dydx = SPlaw(xdata,param)
    #sig       = np.sqrt((xsig*dydx)*(xsig*dydx) + ysig*ysig)
    sig       = ysig
    chi       = (np.abs(ydata - Vsys) - func)/sig
    return chi


# chi in chi square method
def chiDPlaw(param, xdata, ydata, xsig, ysig):
    # param: fitting paramters
    # Vsys: systemic velocity
    # func: estimated y-value
    # dydx: dy/dx
    func ,dydx = DPlaw(xdata, param)
    #sig        = np.sqrt((xsig*dydx)*(xsig*dydx) + ysig*ysig)
    sig        = ysig
    chi        = (ydata - func)/sig
    return chi


def chicheck(param, xdata, ydata, xsig, ysig):
    # param: fitting paramters
    # Vsys: systemic velocity
    # func: estimated y-value
    # dydx     = dy/dx
    func ,dydx = DPlaw(xdata, param)
    sig        = np.sqrt((xsig*dydx)*(xsig*dydx) + ysig*ysig)
    chi        = (np.abs(ydata - Vsys) - func)/sig
    return chi,sig


# double power law function
def DPlaw(radius, param):
    # Vb: rotational velocity at Rb [km/s]
    # Rb: break point raidus at which powers change [au]
    # pin: power at R < Rb
    # pout: power at R > Rb
    Vb, Rb, pin, pout = param
    Vrot = np.array([])
    dydx = np.array([])
    for r in radius:
        if r <= Rb:
            Vrot = np.append(Vrot,Vb*(r/Rb)**(-pin))
            dydx = Vb*(-pin)*(r/Rb)**(-pin-1)
        else:
            Vrot = np.append(Vrot, Vb*(r/Rb)**(-pout))
            dydx = Vb*(-pout)*(r/Rb)**(-pout-1)
    return Vrot, dydx


# single power law function
def SPlaw(r, param, r0=100.):
    Vsys, V0, p = param
    Vrot        = V0*(r/r0)**(-p)
    dydx        = V0*(-p)*(r0**p)**r**(-p-1)
    return Vrot, dydx


# Keplerian rotation velocity
def Kep_rot(r, M):
    # input parameter's unit
    # r: [au]
    # M: [g]

    # convering units
    r = r*auTOcm # au --> cm
    Vkep = np.sqrt(Ggrav*M/r) # cm/s
    Vkep = Vkep*1.e-5 # cm/s --> km/s
    return Vkep

# rotation velocity conserving angular momentum
def infall_rot(r, j):
    #input parameter's unit
    # r: [au]
    # j: specific angular momentum [g cm^2/s]

    # convering units
    r = r*auTOcm # au --> cm
    Vrot = j/r # cm/s
    Vrot = Vrot*1.e-5 # cm/s --> km/s
    return Vrot





def main(fred, fblue, param0, outname, mode='SP', inc=45., dist=140., vsys=None, paramerr_figout=True, offthr_red_in=None, offthr_blue_in=None, offthr_red_out=None, offthr_blue_out=None):
    '''
    fit single or double power-law function, and derive rotational profile.
    '''

    # read files
    offset_red, velocity_red, offerr_red, velerr_red = np.genfromtxt(fred, comments='#', unpack = True)
    offset_blue, velocity_blue, offerr_blue, velerr_blue = np.genfromtxt(fblue, comments='#', unpack = True)

    # offset threshold of used data point
    if offthr_red_in:
        thrindx      = np.where(np.abs(offset_red) >= offthr_red_in)
        offset_red   = offset_red[thrindx]
        velocity_red = velocity_red[thrindx]
        offerr_red   = offerr_red[thrindx]
        velerr_red   = velerr_red[thrindx]

    if offthr_blue_in:
        thrindx       = np.where(np.abs(offset_blue) >= offthr_blue_in)
        offset_blue   = offset_blue[thrindx]
        velocity_blue = velocity_blue[thrindx]
        offerr_blue   = offerr_blue[thrindx]
        velerr_blue   = velerr_blue[thrindx]

    # offset threshold of used data point
    if offthr_red_out:
        thrindx      = np.where(np.abs(offset_red) <= offthr_red_out)
        offset_red   = offset_red[thrindx]
        velocity_red = velocity_red[thrindx]
        offerr_red   = offerr_red[thrindx]
        velerr_red   = velerr_red[thrindx]

    if offthr_blue_out:
        thrindx       = np.where(np.abs(offset_blue) <= offthr_blue_out)
        offset_blue   = offset_blue[thrindx]
        velocity_blue = velocity_blue[thrindx]
        offerr_blue   = offerr_blue[thrindx]
        velerr_blue   = velerr_blue[thrindx]

    # arcsec --> au
    offset_red  = offset_red*dist
    offset_blue = - offset_blue*dist
    offerr_red  = offerr_red*dist
    offerr_blue = offerr_blue*dist
    #print offerr_red, offerr_blue

    # threshold by error bar
    # exclude data points having error bar > 0.3
    erindex      = np.where(velerr_red < 0.3)
    offset_red   = offset_red[erindex]
    velocity_red = velocity_red[erindex]
    offerr_red   = offerr_red[erindex]
    velerr_red   = velerr_red[erindex]

    erindex       = np.where(velerr_blue < 0.3)
    offset_blue   = offset_blue[erindex]
    velocity_blue = velocity_blue[erindex]
    offerr_blue   = offerr_blue[erindex]
    velerr_blue   = velerr_blue[erindex]

    # mergine offset & errors
    offset = np.r_[offset_red, offset_blue]
    offerr = np.r_[offerr_red, offerr_blue]
    velerr = np.r_[velerr_red, velerr_blue]



    ### chi-square fitting
    if mode == 'SP':
        # single-power-law
        # param = [systemic velocity, velocity at 100 au, power]
        # param0: initial value for fitting

        # margine red and blue points
        velocity = np.r_[velocity_red, velocity_blue]


        # fitting
        result       = scipy.optimize.leastsq(chiSPlaw, param0, args=(offset, velocity, offerr, velerr), full_output = True)
        param_SPout  = result[0]
        chi          = chiSPlaw(param_SPout, offset, velocity, offerr, velerr)
        chi2         = np.sum(chi*chi)
        ndata        = len(offset)
        nparam       = len(param_SPout)
        DOF          = ndata - nparam
        reduced_chi2 = chi2/DOF
        print 'chi2', chi2

        # output results
        print 'output parameters'
        print 'Vsys: ', param_SPout[0], ' km/s'
        print 'V100: ', param_SPout[1], ' km/s'
        print 'power-law index: ', param_SPout[2]
        print 'chi-square'
        print 'degree of freedum', DOF
        print 'reduced chi-square: ', reduced_chi2


        # systemic velocity --> relative velocity
        vsys_fixed = param_SPout[0]
        vbfixed    = vsys_fixed - velocity_blue
        vrfixed    = velocity_red - vsys_fixed

        # derive Mstar
        incrad = np.radians(inc)       # degree --> radian
        v100   = param_SPout[1] * 1.e5 # in cm
        vtrue  = v100/np.sin(incrad)   # correct inclination effect
        R100   = 100.                  # in au
        Mstar  = vtrue*vtrue*R100*auTOcm/Ggrav
        Mstar  = Mstar /Msun
        print 'inclination: ', inc, ' degree'
        print 'Mstar: ', Mstar, ' Msun'

        # result
        radius = np.array([r for r in range(1,1000)])
        Vrot   = SPlaw(radius, param_SPout)[0]



        ### error estimation
        # fitting
        param_esterr = np.empty((0,3), float)
        for n in range(3000):
            offest = norm.rvs(size = len(offset), loc = offset, scale = offerr)
            velest = norm.rvs(size = len(velocity), loc = velocity, scale = velerr) # assuming uniform distribution because the error of V is velocity resolution
            result = scipy.optimize.leastsq(chiSPlaw, param_SPout, args=(offest, velest, offerr, velerr), full_output = True)
            param_esterr = np.vstack((param_esterr, result[0]))
        #print param_esterr[:,0]

        # derive error
        vsyssig    = np.std(param_esterr[:,0])
        v100sig    = np.std(param_esterr[:,1])
        psig       = np.std(param_esterr[:,2])
        vsysmedian = np.median(param_esterr[:,0])
        v100median = np.median(param_esterr[:,1])
        pmedian    = np.median(param_esterr[:,2])
        #print vbmedian, '+/-', vbsig
        #print vbmean
        print 'estimated error (standard deviation)'
        print 'vsys \t v100 \t  p'
        print vsyssig, v100sig, psig
        print vsysmedian, v100median, pmedian
        print np.mean(param_esterr[:,0])


        ### plot the results of error estimation
        if paramerr_figout:
            fig_errest = plt.figure(figsize=(11.69,8.27), frameon = False)
            gs         = GridSpec(3, 2)
            xlabels    = [r'$V_\mathrm{sys}$', r'$V_\mathrm{100}$', r'$p$' ]
            for i in xrange(3):
                # histogram
                ax1 = fig_errest.add_subplot(gs[i,0])
                ax1.set_xlabel(xlabels[i])
                if i == 2:
                    ax1.set_ylabel('frequency')
                ax1.hist(param_esterr[:,i], bins = 50, cumulative = False) # density = True

                # cumulative histogram
                ax2 = fig_errest.add_subplot(gs[i,1])
                ax2.set_xlabel(xlabels[i])
                if i == 2:
                    ax2.set_ylabel('cumulative\n frequency')
                ax2.hist(param_esterr[:,i], bins = 50, density=True, cumulative = True)

            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            #plt.show()
            fig_errest.savefig(outname + '_errest.pdf', transparent=True)
            fig_errest.clf()




    elif mode == 'DP':
        ### chi-square fitting
        # double-power-law
        # param = [velocity at break point, break point radius, power-in, power-out]
        # param0: initial value for fitting
        if vsys:
            # margine red and blue points
            velocity = np.r_[velocity_red-vsys, vsys-velocity_blue]
            pass
        else:
            print 'ERROR: Systeimc velocity must be fixed if you fit double power-law function.\
                    Please input systemic velocity in the variable vsys.'

        # fitting
        result       = scipy.optimize.leastsq(chiDPlaw, param0, args=(offset, velocity, offerr, velerr), full_output = True)
        param_out    = result[0]
        chi2         = sum(np.power(chiDPlaw(param_out, offset, velocity, offerr, velerr), 2))
        ndata        = len(offset)
        nparam       = len(param_out)
        DOF          = ndata - nparam - 1
        reduced_chi2 = chi2/DOF


        # output results
        print 'output parameters'
        #print 'chi2', chi2
        print 'Vbreak: ', param_out[0], ' km/s'
        print 'Rbreak: ', param_out[1], ' au'
        print 'p at inner region: ', param_out[2]
        print 'p at outer region: ', param_out[3]
        print 'chi-square'
        print 'degree of freedum', DOF
        print 'reduced chi-square: ', reduced_chi2

        # systemic velocity --> relative velocity
        vsys_fixed = param_out[0]
        vbfixed = vsys - velocity_blue
        vrfixed = velocity_red - vsys

        # result
        radius = np.array([r for r in range(1,1000)])
        Vrot   = DPlaw(radius, param_out)[0]

        ### error estimation
        param_esterr = np.empty((0,4), float)
        for n in range(3000):
            offest = norm.rvs(size = len(offset), loc = offset, scale = offerr)
            velest = norm.rvs(size = len(velocity), loc = velocity, scale = velerr)
            result = scipy.optimize.leastsq(chiDPlaw, param_out, args=(offest, velest, offerr, velerr), full_output = True)
            param_esterr = np.vstack((param_esterr, result[0]))
            #print param_esterr[:,0]


        ### derive error
        vbsig    = np.std(param_esterr[:,0])
        rbsig    = np.std(param_esterr[:,1])
        pinsig   = np.std(param_esterr[:,2])
        poutsig  = np.std(param_esterr[:,3])
        vbmedian = np.median(param_esterr[:,0])
        vbmean   = np.mean(param_esterr[:,0])
        #print vbmedian, '+/-', vbsig
        #print vbmean
        print 'estimated error (standard deviation)'
        print 'vb \t rb \t  pin \t pout'
        print vbsig, rbsig, pinsig, poutsig



        ### plot the results of error estimation
        fig_errest = plt.figure(figsize=(11.69,8.27), frameon = False)
        plt.rcParams["font.size"] = 14
        gs = GridSpec(4, 2)
        xlabels = [r'$V_{break}$', r'$R_{break}$', r'$p_{in}$', r'$p_{out}$' ]
        for i in range(4):
            # histogram
            ax1 = fig_errest.add_subplot(gs[i,0])
            ax1.set_xlabel(xlabels[i])
            if i == 3:
                ax1.set_ylabel('frequency')
            ax1 = plt.hist(param_esterr[:,i], bins = 50, density=True, cumulative = False)

            # cumulative histogram
            ax2 = fig_errest.add_subplot(gs[i,1])
            ax2.set_xlabel(xlabels[i])
            if i == 3:
                ax2.set_ylabel('cumulative\n frequency')
            ax2 = plt.hist(param_esterr[:,i], bins = 50, density=True, cumulative = True)

        plt.subplots_adjust(wspace=0.2, hspace=0.6)
        #plt.show()
        fig_errest.savefig(outname + '_errest.pdf', transparent=True)
        fig_errest.clf()

    else:
        print 'ERROR: mode must be choosen in SP or DP. Please check your input.'



    ### plot; log r vs log v plot
    # set a figure
    fig = plt.figure(figsize=(11.69,8.27))
    ax  = fig.add_subplot(111)

    # axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(20,200)
    ax.set_ylim(0.5,5)
    ax.set_xlabel('radius (au)')
    ax.set_ylabel('velocity (km/s)')
    ax.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)
    #ax2.set_aspect(0.5)
    #ax2.grid()

    '''
    # Keplerian rotation
    Mstar = 1.5*Msun # [g]
    inc = 66.*pi/180. # [radian]
    # Yen et al. (2014)
    # Mstar: 1.6 +/- 0.5 Msun
    # inclination angle: 66 +/- 10 deg
    j = 1.13e21
    radius = np.array([r for r in range(1,1000)])
    Vkep = Kep_rot(radius, Mstar)
    Vrot = Vkep*np.sin(inc)
    Vrot_infall = infall_rot(radius, j)
    #print Vkep
    '''

    # fitting results
    ax.errorbar(offset_red, vrfixed, xerr = offerr_red, yerr = velerr_red, fmt= ' ', color = 'red', capsize = 2, capthick = 1)
    ax.errorbar(offset_blue, vbfixed, xerr = offerr_blue, yerr = velerr_blue, fmt= ' ', color = 'blue', capsize = 2, capthick = 1)
    ax.plot(radius, Vrot, 'k-', lw=1)

    #plt.show()
    fig.savefig(outname+'.pdf', transparent=True)
    return ax


if __name__ == '__main__':
    ### input parameters
    #imagename = '../../../l1489.c18o.contsub.gain01.rbp05.mlt100.taper500k.pbcor.pv54dmaj1wd.fits'
    rms       = 5.4e-3 # Jy/beam, for taper 500k

    fred  = 'l1489_red_chi2_pv_vfit.txt'
    fblue = 'l1489_blue_chi2_pv_vfit.txt'
    inc   = 73. # degree
    vsys  = 7.2 # km/s
    dist  = 140. # pc
    offthr_red_in  = 2. # arcsec
    offthr_red_out = None
    offthr_blue_in = 2. # arcsec
    offthr_blue_out = None


    outname   = 'rot_plawfit_result_54d_SP'
    mode      = 'SP'
    param0_sp = [7.3, 3., 0.5]
    param0_dp = [1.2,  800., 0.5, 1.] # for DP

    ax = main(fred, fblue, param0_sp, outname, mode=mode, inc=inc, dist=dist, vsys=vsys, offthr_red_in=offthr_red_in, offthr_blue_in=offthr_blue_in, offthr_red_out=offthr_red_out, offthr_blue_out=offthr_blue_out)
    ax.set_xlim(200,1000)
    ax.set_ylim(0.6,3)
    #ax.xaxis.set_major_formatter(ScalarFormatter())
    #ax.yaxis.set_major_formatter(ScalarFormatter())
    #ax.xaxis.set_minor_formatter(ScalarFormatter())
    #ax.yaxis.set_minor_formatter(ScalarFormatter())

    # ticks
    # y
    ax.set_yticks([1,2,3],minor=False)
    ax.set_yticks([0.6,0.7,0.8,0.9],minor=True)
    ax.set_yticklabels(['1','2'], minor=False)
    ax.set_yticklabels(['0.6'], minor=True)
    # x
    ax.set_xticks([1000],minor=False)
    ax.set_xticks([200,300,400,500,600,700,800,900],minor=True)
    ax.set_xticklabels(['200','','','','600','',''], minor=True)
    ax.set_xticklabels(['1000'], minor=False)
    ax.set_aspect(1)

    plt.savefig(outname+'.pdf', transparent=True)