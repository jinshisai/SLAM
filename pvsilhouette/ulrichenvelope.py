import numpy as np


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
    return x, y, z
