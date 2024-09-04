import numpy as np
from numba import jit
from numba import prange


class diskenvelope():
    def __init__(self, radius: np.ndarray = None,
                 theta: np.ndarray = None,
                 phi: np.ndarray = None, incl: float = 0,
                 H0: float = 0.2, plh: float = 0.25, pls: float = 1.0):
        """theta is a polar angle from +z to (x, y)=(sin phi, -cos phi).
           phi is an azimuthal angle from -y to +x.
        """
        self.H0 = H0
        self.plh = plh
        self.pls = pls
        if theta is not None:
            self.theta = theta
            mu = np.cos(theta)
            self.sign_mu = np.sign(mu)
            self.mu = np.abs(mu)
            self.sin_theta = np.sin(theta).clip(1e-10, None)
            if radius is not None:
                self.radius = radius
                self.R = self.radius * self.sin_theta
                self.z = self.radius * self.mu
                self.H = self.H0 * self.R**(1. + self.plh)
                self.where_disk = (self.R < 1) * (self.z < 3 * self.H)
            if phi is not None:
                self.phi = phi
                t = theta.ravel()
                cos_t = np.cos(t)
                sin_t = np.sin(t)
                p = phi.ravel()
                cos_p = np.cos(p)
                sin_p = np.sin(p)
                er = np.array([sin_t * sin_p, -sin_t * cos_p, cos_t])
                et = np.array([cos_t * sin_p, -cos_t * cos_p, -sin_t])
                ep = np.array([cos_p, sin_p, np.zeros_like(p)])
                elos = np.array([0, -np.sin(incl), np.cos(incl)])
                elos = elos[:, np.newaxis]
                elos_r = -np.sum(er * elos, axis=0)
                elos_t = -np.sum(et * elos, axis=0)
                elos_p = -np.sum(ep * elos, axis=0)
                shape = np.shape(self.theta)
                self.elos_r = np.reshape(elos_r, shape)
                self.elos_t = np.reshape(elos_t, shape)
                self.elos_p = np.reshape(elos_p, shape)

    def get_mu0(self, mu: np.ndarray):
        r = self.radius
        p = (r - 1) / 3.
        q = mu * r / 2.
        D = q**2 + p**3
        mu0 = np.full_like(r, np.nan)
        ### three cases for computational stability
        # r >= 1 (then p >= 0 and sqrt(D) >= q >= 0)
        c = (r >= 1)
        qD = q[c] * (D[c])**(-1/2)
        mu0[c] = D[c]**(1/6) * ((1 + qD)**(1/3) - (1 - qD)**(1/3))
        # r < 1 and D >= 0 (then p < 0 and sqrt(D) < q)
        c = (r < 1) * (D >= 0)
        mu0[c] = (q[c] + np.sqrt(D[c]))**(1/3) + (q[c] - np.sqrt(D[c]))**(1/3)
        # D < 0 (then p < 0)
        c = (D < 0)
        minus_p = -p[c]
        sqrt_minus_p = np.sqrt(minus_p.clip(0, None))
        mu0[c] = 2 * sqrt_minus_p \
                 * np.cos(np.arccos(q[c] * sqrt_minus_p**(-3)) / 3.)
        return mu0.clip(0, 1)
        

    def envelope(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mu0 = self.get_mu0(self.mu)
        sin_theta0 = np.sqrt(1 - mu0**2)
        r = self.radius
        mm = 1 - sin_theta0**2 / r  # mu / mu0
        vr = -np.sqrt(1 + mm) / np.sqrt(r)
        vt = self.sign_mu * np.sqrt(1 + mm) \
             * (mu0 - self.mu) / self.sin_theta / np.sqrt(r)
        vp = np.sqrt(1 - mm) * sin_theta0 / self.sin_theta / np.sqrt(r)
        clipped_mu = self.mu.clip(0.17, 1.0)  # theta < 80 deg
        clipped_mu0 = self.get_mu0(clipped_mu)
        clipped_mm = clipped_mu0 / clipped_mu
        rho = (np.sqrt((1 + clipped_mm) / 2) * (2 * clipped_mu0**2 / r + clipped_mm))**(-1) \
              / np.sqrt(r)**3
        c = self.where_disk
        vr[c] = 0
        vt[c] = 0
        vp[c] = 0
        rho[c] = 0
        return vr, vt, vp, rho

    def disk(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r = self.radius
        vp = self.sin_theta / np.sqrt(r)
        rho = r**(-self.pls) * self.H0 / self.H \
              * np.exp(-0.5 * (self.z / self.H)**2)
        c = self.where_disk
        vp[~c] = 0
        rho[~c] = 0
        return vp, rho

def rotbase(t: np.ndarray, p: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """t is an inclination angle from +z to (x, y)=(sin p, -cos p).
       p is an azimuthal angle from -y to +x.
    """
    er = np.array([np.sin(t) * np.sin(p), -np.sin(t) * np.cos(p), np.cos(t)])
    et = np.array([np.cos(t) * np.sin(p), -np.cos(t) * np.cos(p), -np.sin(t)])
    ep = np.array([np.cos(p), np.sin(p), np.zeros_like(p)])
    return er, et, ep

def XYZ2rtp(incl: float, phi: float,
            X: np.ndarray, Y: np.ndarray, Z: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The observer coordinate (X,Y,Z) and the envelope coordinate (x,y,z)
       are linked as X * ep + Y * (-et) + Z * er = x, y, z.
    """
    shape = np.shape(X)
    er, et, ep = rotbase(incl, phi)
    x, y, z = np.outer(ep, X.ravel()) \
              + np.outer(-et, Y.ravel()) \
              + np.outer(er, Z.ravel())
    r = np.linalg.norm([x, y, z], axis=0).clip(1e-10, None)
    t = np.arccos(z / r)
    p = np.arctan2(x, -y)
    r = np.reshape(r, shape)
    t = np.reshape(t, shape)
    p = np.reshape(p, shape)
    return r, t, p

gauss_xy = None
gauss_v = None
vedge = None
@jit(parallel=True)
def rho2tau(vlos: np.ndarray, rho: np.ndarray, dz: float) -> np.ndarray:
    nv = len(vedge) - 1
    nx, ny, _ = np.shape(vlos)
    tau = np.zeros((nv, ny, nx))
    for i in prange(nv):
        mask = (vedge[i] <= vlos) * (vlos < vedge[i + 1])
        tau[i] = np.sum(mask * rho * dz, axis=2).T
    return tau

Nr = 1600
lnr = np.linspace(np.log(1e-4), np.log(1e4), Nr)  # dr/r ~ dtheta ~ 0.01
lnr0 = lnr[0]
dlnr = np.exp(lnr[1] - lnr[0]) - 1
Ntheta = int(np.pi / dlnr + 0.5)
theta = np.linspace(0, np.pi, Ntheta)
theta0 = theta[0]
dtheta = theta[1] - theta[0]
lnr_mesh, theta_mesh = np.meshgrid(lnr, theta)

m = diskenvelope(radius=np.exp(lnr_mesh), theta=theta_mesh)
vr_env, vt_env, vp_env, rho_env = m.envelope()
vp_disk, rho_disk = m.disk()
vp_all = vp_env + vp_disk

lmax = 10
elos_r = {'major' : [None] * lmax, 'minor' : [None] * lmax}
elos_t = {'major' : [None] * lmax, 'minor' : [None] * lmax}
elos_p = {'major' : [None] * lmax, 'minor' : [None] * lmax}
t = {'major' : [None] * lmax, 'minor' : [None] * lmax}
idx_t = {'major' : [None] * lmax, 'minor' : [None] * lmax}
r_org = {'major' : [None] * lmax, 'minor' : [None] * lmax}
j_org = {'major' : [None] * lmax, 'minor' : [None] * lmax}
def update(radius_org: np.ndarray, theta: np.ndarray, phi: np.ndarray, incl: float,
           axis: str, l: int) -> None:
    m = diskenvelope(theta=theta, phi=phi, incl=incl)
    elos_r[axis][l] = m.elos_r
    elos_t[axis][l] = m.elos_t
    elos_p[axis][l] = m.elos_p
    t[axis][l] = m.theta
    i = (m.theta - theta0) / dtheta + 0.5
    idx_t[axis][l] = i.astype(int)
    r_org[axis][l] = radius_org
    j_org[axis][l] = (np.log(radius_org) - lnr0) / dlnr + 0.5

def get_rho_vlos(Rc: float, rho_jump: float, alphainfall: float,
                 axis: str, l: int) -> tuple[np.ndarray, np.ndarray]:
    i = idx_t[axis][l]
    j = j_org[axis][l] - np.log(Rc) / dlnr
    j = np.clip(j, 0, Nr - 1) 
    j = j.astype(int)
    rho = rho_env[i, j] + rho_disk[i, j] * rho_jump
    vr = vr_env[i, j] * alphainfall
    vt = vt_env[i, j]
    vp = vp_all[i, j]
    vlos = vr * elos_r[axis][l] + vt * elos_t[axis][l] + vp * elos_p[axis][l]
    return rho, vlos
