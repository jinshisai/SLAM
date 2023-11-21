import numpy as np
import time
from astropy import constants, units
from scipy.signal import convolve


au = units.au.to('m')
GG = constants.G.si.value
M_sun = constants.M_sun.si.value
deg = units.deg.to('radian')


def kepvel(radius, theta):
    radius = radius.clip(1e-10, None)
    R = radius * np.sin(theta)
    vr = np.zeros_like(radius)
    vt = np.zeros_like(radius)
    vp = 1 / np.sqrt(radius) * R / radius
    return vr, vt, vp

def velrho(radius, theta, alphainfall: float = 1, withkepler: bool = True):
    radius = radius.clip(1e-10, None)
    parity = np.sign(mu := np.cos(theta))
    mu = np.abs(mu).clip(1e-10, 1)
    p = (radius - 1) / 3.
    q = mu * radius / 2.
    mu0 = np.full_like(radius, np.nan)
    ### three cases for computational stability
    # r >= 1 (p >= 0, D >= 0)
    c = (radius >= 1)
    D = q[c]**2 + p[c]**3
    qD = q[c] * (D)**(-1/2)
    mu0[c] = D**(1/6) * ((1 + qD)**(1/3) - (1 - qD)**(1/3))
    # r < 1 (p < 0) and D >= 0
    p_minus = (-p).clip(0, None)
    D = q**2 - p_minus**3
    c = (radius < 1) * (D >= 0)
    mu0[c] = (q[c] + np.sqrt(D[c]))**(1/3) + (q[c] - np.sqrt(D[c]))**(1/3)
    # D < 0 (p < 0, r < 1)
    c = (D < 0)
    mu0[c] = 2 * np.sqrt(p_minus[c]) \
             * np.cos(np.arccos(q[c] * p_minus[c]**(-3/2)) / 3.)
    
    mm = mu / mu0
    vr = -np.sqrt(1 + mm) / np.sqrt(radius) * alphainfall
    vt = parity * np.sqrt(1 + mm) * (mu0 - mu) / np.sin(theta) / np.sqrt(radius)
    vp = np.sqrt(1 - mm) / np.sqrt(radius)
    rho = radius**(-3/2) / np.sqrt(1 + mm) / (2 / radius * mu0**2 + mm)
    
    if withkepler:
        R = radius * np.sin(theta)
        vkep = kepvel(radius, theta)
        vr[R < 1] = vkep[0][R < 1]
        vt[R < 1] = vkep[1][R < 1]
        vp[R < 1] = vkep[2][R < 1]
    return vr, vt, vp, rho

def xyz2rtp(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2).clip(1e-10, None)
    t = np.arccos(z / r)
    p = np.arctan2(x, -y)
    return r, t, p

def rotbase(t, p):
    """t is an inclination angle from +z to (x, y)=(sin p, -cos p).
       p is an azimuthal angle from -y to +x.
    """
    er = np.array([np.sin(t) * np.sin(p), -np.sin(t) * np.cos(p), np.cos(t)])
    et = np.array([np.cos(t) * np.sin(p), -np.cos(t) * np.cos(p), -np.sin(t)])
    ep = np.array([np.cos(p), np.sin(p), np.zeros_like(p)])
    return er, et, ep

def XYZ2xyz(incl, phi, X, Y, Z):
    """The observer coordinate (X,Y,Z) and the envelope coordinate (x,y,z)
       are linked as X * ep + Y * (-et) + Z * er = x, y, z.
    """
    shape = np.shape(X)
    er, et, ep = rotbase(incl, phi)
    x, y, z = np.outer(ep, X.ravel()) \
              + np.outer(-et, Y.ravel()) \
              + np.outer(er, Z.ravel())
    x = np.reshape(x, shape)
    y = np.reshape(y, shape)
    z = np.reshape(z, shape)
    #r2D = np.sqrt(x**2 + y**2)
    #x[r2D < 1] = np.nan
    #y[r2D < 1] = np.nan
    #z[r2D < 1] = np.nan
    return x, y, z

def losvel(elos, r, t, p, alphainfall: float = 1, 
    kepler: bool = False, withkepler: bool = True):
    """Line-of-sight velocity with a given vector, elos,
       which points to the observer in the envelople coordinate.
    """
    shape = np.shape(r)
    if kepler:
        vr, vt, vp = kepvel(r.ravel(), t.ravel())
    else:
        vr, vt, vp, _ = velrho(r.ravel(), t.ravel(), alphainfall, withkepler)
    er, et, ep = rotbase(t.ravel(), p.ravel())
    e = np.moveaxis(np.full((len(r.ravel()), 3), elos), -1, 0)
    vlos = -vr * np.sum(er * e, axis=0) \
           -vt * np.sum(et * e, axis=0) \
           -vp * np.sum(ep * e, axis=0)
    return np.reshape(vlos, shape)

def velmax(r: np.ndarray, Mstar: float, Rc: float,
           alphainfall: float = 1, incl: float = 90, withkepler: bool = True):
    irad = np.radians(incl)
    vunit = np.sqrt(GG * Mstar * M_sun / Rc / au) * 1e-3
    X, Z = np.meshgrid(r / Rc, r / Rc)  # X is inner, second axis
    zero = np.zeros_like(X)
    elos = [0, -np.sin(irad), np.cos(irad)]
    a = {}
    x, y, z = XYZ2xyz(irad, 0, X, zero, Z)
    r, t, p = xyz2rtp(x, y, z)
    vlos = losvel(elos, r, t, p, alphainfall, False, withkepler)
    #vlos[(t < np.pi / 4 ) + (np.pi * 3 / 4 < t)] = 0  # remove outflow cavity
    vlosmax = np.nanmax(vlos, axis=0)
    vlosmin = np.nanmin(vlos, axis=0)
    a['major'] = {'vlosmax':vlosmax * vunit, 'vlosmin':vlosmin * vunit}
    x, y, z = XYZ2xyz(irad, 0, zero, X, Z)
    r, t, p = xyz2rtp(x, y, z)
    vlos = losvel(elos, r, t, p, alphainfall, False, withkepler)
    #vlos[(t < np.pi / 4 ) + (np.pi * 3 / 4 < t)] = 0  # remove outflow cavity
    vlosmax = np.nanmax(vlos, axis=0)
    vlosmin = np.nanmin(vlos, axis=0)
    a['minor'] = {'vlosmax':vlosmax * vunit, 'vlosmin':vlosmin * vunit}
    return a


def mockpvd(xin: np.ndarray, zin: np.ndarray, v: np.ndarray, 
    Mstar: float, Rc: float, 
    alphainfall: float = 1., incl: float = 89., fscale: float = 1.,
    pa: float = 0., beam: list = None, linewidth: float = None, rout: float=None,
    axis: str = 'major'):
    """
    Generate a mock Position-Velocity (PV) diagram.

    XYZ: plane of sky coordinates with X axis as a major axis and Z axis as a line of sight.
    """
    # parameters/units
    if (incl > 89.) & (incl <= 90.):
        incl = 89.
    elif (incl > 90.) & (incl < 91.):
        incl = 91.
    irad = np.radians(incl)
    vunit = np.sqrt(GG * Mstar * M_sun / Rc / au) * 1e-3
    elos = [0, -np.sin(irad), np.cos(irad)] # line of sight vector
    nz, nx = len(zin), len(xin)

    # grid
    if beam is None:
        X, Z = np.meshgrid(xin/Rc, zin/Rc, indexing='ij') # in units of Rc
        Y = np.zeros_like(X)
        ny = 1
    else:
        if len(beam) == 3:
            bmaj, bmin, bpa = beam
            dely = bmin * 0.2
            yin = np.arange(-int(bmaj / dely * 2.), int(bmaj/dely * 2.) + 1, 1) * dely
        else:
            print('ERROR\tmockpvd: beam must be given as [bmaj, bmin, bpa].')
            return 0

        ny = len(yin)
        X, Y, Z = np.meshgrid(xin/Rc, yin/Rc, zin/Rc, indexing='ij')

    # along which axis
    x, y, z = XYZ2xyz(irad, 0., X, Y, Z) if axis == 'major' else XYZ2xyz(irad, 0., Y, X, Z)
    r, t, p = xyz2rtp(x, y, z)

    # get density and velocity
    _, _, _, rho = velrho(r, t, alphainfall, withkepler=False)
    #print(np.nanmax(rho))
    #print(r[np.where(rho == np.nanmax(rho))], t[np.where(rho == np.nanmax(rho))])
    rho /= np.nanmax(rho)   # normalize
    if len(rho.shape) != 3: rho = rho.reshape(nx, ny, nz) # in 3D
    vlos = losvel(elos, r, t, p, alphainfall, kepler=False, withkepler=False) * vunit
    if len(vlos.shape) != 3: vlos = vlos.reshape(nx, ny, nz) # in 3D

    # outer edge
    if rout is not None: rho[np.where(r.reshape(nx, ny, nz) > rout/Rc)] = np.nan

    # integrate along Z axis
    nv = len(v)
    delv = v[1] - v[0]
    ve = np.hstack([v - delv * 0.5, v[-1] + 0.5 * delv])
    #I_cube = np.array([[[
    #    np.nansum(rho[i,j, np.where((ve[k] <= vlos[i,j,:]) & (vlos[i,j,:] < ve[k+1]))])
    #    if len(np.where((ve[k] <= vlos[i,j,:]) & (vlos[i,j,:] < ve[k+1]))[0]) != 0
    #    else 0.
    #    for i in range(nx)]
    #    for j in range(ny)]
    #    for k in range(nv)
    #    ])
    I_cube = np.zeros((nv, ny, nx))
    for i in range(nv):
        _rho = np.where((ve[i] <= vlos) & (vlos < ve[i+1]), rho, np.nan)
        I_cube[i,:,:] = np.nansum(_rho, axis=2).T
    #I_cube = np.array([
    #    np.nansum(np.where((ve[i] <= vlos) & (vlos < ve[i+1]), rho, np.nan), axis=2).T
    #    for i in range(nv)
    #    ])

    # convolution along the spectral direction
    if linewidth is not None:
        gaussbeam = np.exp(-(v /(2. * linewidth / 2.35))**2.)
        I_cube = convolve(I_cube, np.array([[gaussbeam]]).T, mode='same')

    # beam convolution
    if beam is not None:
        xb, yb = rot(*np.meshgrid(xin, yin), np.radians(bpa - pa))
        gaussbeam = np.exp(-(yb /(2. * bmin / 2.35))**2. - (xb / (2. * bmaj / 2.35))**2.)
        I_cube = convolve(I_cube, np.array([gaussbeam]), mode='same')

    # output
    I_pv = I_cube[:,ny//2,:]
    I_pv /= np.nanmax(I_pv) # normalize
    I_pv *= fscale # scaling
    return I_pv


def rot(x, y, pa):
    s = x * np.cos(pa) - y * np.sin(pa)  # along minor axis
    t = x * np.sin(pa) + y * np.cos(pa)  # along major axis
    return np.array([s, t])
