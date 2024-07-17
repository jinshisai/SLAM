import numpy as np
import time
from astropy import constants, units
from scipy.signal import convolve

from grid import Nested3DGrid
from ulrichenvelope import *


au = units.au.to('m')
GG = constants.G.si.value
M_sun = constants.M_sun.si.value
deg = units.deg.to('radian')



class MockPVD(object):
    """
    MockPVD is a class to generate mock PV diagram.

    """
    def __init__(self, x:np.ndarray, z:np.ndarray, v:np.ndarray, 
        nsubgrid:int = 1, nnest:list = None, 
        xlim:list = None, ylim: list = None, zlim:list = None,
        beam:list = None):
        '''
        Initialize MockPVD with a given grid. z is the line of sight axis.

        Parameters
        ----------
        x, z, v (array): 1D arrays for x, z and v axes.
        nsubgrid (int): Number of sub pixels to which the original pixel is divided.
        nnest (list): Number of sub pixels of the nested grid, to which the parental pixel is divided.
         E.g., if nnest=[4], a nested grid having a resolution four finer is created. If [4, 4],
         the grid is nested to two levels and each has four times better resolution than the upper grid.
        xlim, zlim (list): x and z ranges for the nested grid. Must be given as [[xmin0, xmax0], [xmin1, xmax1]].
        '''
        super(MockPVD, self).__init__()

        # save input
        self._x, self._z = x, z
        self._nx, self._nz = len(x), len(z)
        # subgrid
        self.nsubgrid = nsubgrid
        if nsubgrid > 1:
            x, z = self.subgrid([x, z], nsubgrid)
            self.x, self.z = x, z
        else:
            self.x, self.z = x, z
        self.nx, self.nz = len(x), len(z)
        self.v = v
        # nested grid
        self.nnest = nnest
        # beam
        self.beam = beam

        # make grid
        self.makegrid(xlim, ylim, zlim)
        #print(self.grid.xnest)



    def generate_mockpvd(self, 
        Mstar: float, Rc: float, alphainfall: float = 1., 
        fflux: float = 1., frho: float = 1., ftau: float = 1.,
        incl: float = 89., withkepler: bool = True, 
        pls: float = -1., plh: float = 0.25, h0: float = 0.1, 
        pa: float = 0., linewidth: float = None, 
        rin: float = 0.1, rout: float = None,
        axis: str = 'both', tauscale: float = 1.,):
        '''
        Generate a mock PV diagram.

        Parameters
        ----------
        Mstar (float): Stellar mass (Msun)
        Rc (float): Centrifugal radius (au)
        alphainfall (float): Decelerating factor
        fflux (float): Factor to scale mock flux
        frho (float): Factor to scale density contrast between disk and envelope
        ftau (float): Factor to scale mock optical depth
        incl (float): Inclination angle (deg). Incl=90 deg corresponds to edge-on configuration.
        axis (str): Axis of the pv cut. Must be major, minor or both.
        '''
        # parameters
        self.Mstar = Mstar
        self.Rc = Rc
        self.alphainfall = alphainfall
        self.incl = incl

        # check
        if axis not in ['major', 'minor', 'both']:
            print("ERROR\tgenerate_mockpvd: axis input must be 'major', 'minor' or 'both'.")
            return 0

        # Generate PV diagram
        if axis == 'both':
            I_out = []
            for _axis in ['major', 'minor']:
                # build model
                rho, vlos = self.build(Mstar, Rc, incl, 
                    alphainfall = alphainfall, withkepler = withkepler, 
                    pls = pls, plh = plh, h0 = h0, frho = frho, rin = rin, rout = rout, 
                    axis = _axis, collapse = True)
                # PV cut
                I_pv = self.generate_pvd(rho, vlos, fflux, ftau, beam = self.beam,
                    linewidth = linewidth)
                I_out.append(I_pv)
            return I_out
        else:
            # build model
            rho, vlos = self.build(Mstar, Rc, incl, 
                alphainfall = alphainfall, withkepler = withkepler, 
                pls = pls, plh = plh, h0 = h0, frho = frho, rin = rin, rout = rout, 
                axis = axis, collapse = True)
            # PV cut
            return self.generate_pvd(rho, vlos, fflux, ftau, beam = self.beam,
                linewidth = linewidth)


    def subgrid(self, axes:list, nsubgrid:int):
        axes_out = []
        for x in axes:
            nx = len(x)
            dx = x[1] - x[0]
            x_e = np.linspace( x[0] - 0.5 * dx, x[-1] + 0.5 * dx, nx*nsubgrid + 1)
            x = 0.5 * (x_e[:-1] + x_e[1:])
            axes_out.append(x)
        return axes_out


    def makegrid(self, 
        xlim:list = None, ylim:list = None, zlim:list = None):
        # parental grid
        # x and z
        x = self.x
        z = self.z
        dx = x[1] - x[0]
        # y axis
        if self.beam is not None:
            bmaj, bmin, bpa = self.beam
            y = np.arange(
                -int(bmaj / dx * 3. / 2.35) -1, 
                int(bmaj / dx * 3. / 2.35) + 2, 
                1) * dx # +/- 3 sigma
            self.y = y
        else:
            y = np.array([0.])


        if self.nnest is not None:
            grid = Nested3DGrid(x, y, z, xlim, ylim, zlim,
                self.nnest, nlevels = len(self.nnest))
        else:
            grid = Nested3DGrid(x, y, z, None, None, None, [1],
                nlevels = 0)
        self.grid = grid


    def build(self, Mstar:float, Rc:float, incl:float,
        alphainfall:float = 1., withkepler: bool = True, 
        pls: float = -1., plh: float = 0.25, h0: float = 0.1, frho:float = 1.,
        rin:float = 0.1, rout:float = None, collapse = True,
        axis: str = 'major'):
        # parameters/units
        irad = np.radians(incl)
        vunit = np.sqrt(GG * Mstar * M_sun / Rc / au) * 1e-3
        elos = [0, -np.sin(irad), np.cos(irad)] # line of sight vector

        # for each nested level
        d_rho = [None] * self.grid.nlevels
        d_vlos = [None] * self.grid.nlevels
        for l in range(self.grid.nlevels):
            X, Y, Z = self.grid.xnest[l], self.grid.ynest[l], self.grid.znest[l]
            X /= Rc
            Y /= Rc
            Z /= Rc

            # along which axis
            x, y, z = XYZ2xyz(irad, 0., X, Y, Z) if axis == 'major' else XYZ2xyz(irad, 0, Y, X, Z)
            r, t, p = xyz2rtp(x, y, z)

            # get density and velocity
            _, _, _, rho = velrho(r, t, alphainfall, withkepler=withkepler,
                rc = Rc, pls = pls, plh = plh, h0 = h0, rho_jump = frho)
            rho /= np.nanmax(rho)   # normalize
            #if len(rho.shape) != 3: rho = rho.reshape(nx, ny, nz) # in 3D
            vlos = losvel(elos, r, t, p, alphainfall, kepler=False, withkepler=withkepler,
                rc = Rc, pls = pls, plh = plh, h0 = h0, rho_jump = frho) * vunit
            #if len(vlos.shape) != 3: vlos = vlos.reshape(nx, ny, nz) # in 3D

            # inner and outer edge
            rho[r * Rc <= rin] = 0.
            if rout is not None: rho[np.where(r * Rc > rout)] = np.nan

            d_rho[l] = rho
            d_vlos[l] = vlos

        # collapse
        if collapse * (self.grid.nlevels >= 2):
            d_rho = self.grid.collapse(d_rho)
            d_vlos = self.grid.collapse(d_vlos)
        else:
            d_rho = d_rho[0]
            d_vlos = d_vlos[0]
        d_rho = d_rho.reshape(self.grid.nx, self.grid.ny, self.grid.nz)
        d_vlos = d_vlos.reshape(self.grid.nx, self.grid.ny, self.grid.nz)

        return d_rho, d_vlos


    def generate_pvd(self, rho:np.ndarray | list, vlos:np.ndarray | list, 
        fflux:float, ftau:float, beam:list = None, linewidth: float = None, 
        pa: float = 0.):
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        # integrate along Z axis
        v = self.v.copy()
        nv = len(v)
        delv = v[1] - v[0]
        ve = np.hstack([v - delv * 0.5, v[-1] + 0.5 * delv])

        tau_v  = np.zeros((nv, ny, nx)) # mock tau
        for i in range(nv):
            _rho = np.where((ve[i] <= vlos) & (vlos < ve[i+1]), rho, np.nan)
            tau_v[i,:,:] = np.nansum(_rho, axis=2).T * ftau

        # convolution along the spectral direction
        if linewidth is not None:
            #gaussbeam = np.exp(- 0.5 * (v /(linewidth / 2.35))**2.)
            gaussbeam = np.exp(- (v - v[nv//2 - 1 + nv%2])**2. / linewidth**2.)
            gaussbeam /= np.sum(gaussbeam)
            tau_v = convolve(tau_v, 
            np.array([[gaussbeam]]).T, mode='same') # conserve integrated value

        I_cube = fflux * ( 1. - np.exp(-tau_v) ) # fscale = (Bv(Tex) - Bv(Tbg))

        # beam convolution
        if beam is not None:
            bmaj, bmin, bpa = beam
            xb, yb = rot(*np.meshgrid(self.x, self.y, indexing='ij'), np.radians(bpa - pa))
            gaussbeam = np.exp(
                - 0.5 * (yb /(bmin / 2.35))**2. \
                - 0.5 * (xb /(bmaj / 2.35))**2.)
            gaussbeam /= np.sum(gaussbeam)
            I_cube = convolve(I_cube, np.array([gaussbeam]), mode='same')

        # output
        I_pv = I_cube[:,ny//2,:]

        if self.nsubgrid > 1:
            I_pv = np.nanmean(
                np.array([
                I_pv[:, i::self.nsubgrid]
                for i in range(self.nsubgrid)
                ]),
                axis = 0)
        return I_pv