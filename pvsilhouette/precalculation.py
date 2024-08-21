import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI


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
                elos = np.moveaxis(np.full((len(t), 3), elos), -1, 0)
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


Nr = 1600
lnr = np.linspace(np.log(1e-4), np.log(1e4), Nr)  # dr/r ~ dtheta ~ 0.01
dlnr = np.exp(lnr[1] - lnr[0]) - 1
Nt = int(np.pi / dlnr + 0.5)
t = np.linspace(0, np.pi, Nt)
lnr_mesh, t_mesh = np.meshgrid(lnr, t)
m = diskenvelope(radius=np.exp(lnr_mesh), theta=t_mesh)

vr_env, vt_env, vp_env, rho_env = m.envelope()
vp_disk, rho_disk = m.disk()
f_vr_env = RGI((t, lnr), vr_env, bounds_error=False, fill_value=0)
f_vt_env = RGI((t, lnr), vt_env, bounds_error=False, fill_value=0)
f_vp = RGI((t, lnr), vp_env + vp_disk, bounds_error=False, fill_value=0)
f_rho_env = RGI((t, lnr), rho_env, bounds_error=False, fill_value=0)
f_rho_disk = RGI((t, lnr), rho_disk, bounds_error=False, fill_value=0)

lmax = 10
elos_r = {'major' : [None] * lmax, 'minor' : [None] * lmax}
elos_t = {'major' : [None] * lmax, 'minor' : [None] * lmax}
elos_p = {'major' : [None] * lmax, 'minor' : [None] * lmax}
t = {'major' : [None] * lmax, 'minor' : [None] * lmax}
def update(theta: np.ndarray, phi: np.ndarray, incl: float,
           axis: str, l: int) -> None:
    m = diskenvelope(theta=theta, phi=phi, incl=incl)
    elos_r[axis][l] = m.elos_r
    elos_t[axis][l] = m.elos_t
    elos_p[axis][l] = m.elos_p
    t[axis][l] = m.theta

def get_rho(radius: np.ndarray, rho_jump: float,
            axis: str, l: int) -> np.ndarray:
    theta = t[axis][l]
    rho = f_rho_env((theta, radius)) + f_rho_disk((theta, radius)) * rho_jump
    return rho

def get_vlos(radius: np.ndarray, alphainfall: float,
             axis: str, l: int) -> np.ndarray:
    theta = t[axis][l]
    vr = f_vr_env((theta, radius)) * alphainfall
    vt = f_vt_env((theta, radius))
    vp = f_vp((theta, radius))
    vlos = vr * elos_r[axis][l] + vt * elos_t[axis][l] + vp * elos_p[axis][l]
    return vlos
