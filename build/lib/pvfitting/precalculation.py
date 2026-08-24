import os
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
from numba import config, get_num_threads, jit, prange, set_num_threads


DEFAULT_NUMBA_THREAD_CAP = 4


def resolve_num_threads(num_threads: int | str | None = None) -> int:
    """Resolve the Numba thread budget used for PV integration.

    Args:
        num_threads (int, str, or None, optional): Positive thread count,
            ``'all'`` to use every Numba thread, or None for an automatic
            budget. Defaults to None.

    Returns:
        int: Number of Numba threads to use.

    Raises:
        TypeError: If ``num_threads`` is not an integer, ``'all'``, or None.
        ValueError: If an integer is outside Numba's available thread range.

    Notes:
        With None, an existing ``NUMBA_NUM_THREADS`` setting or runtime thread
        mask is respected. Otherwise, half the available threads are used, up
        to ``DEFAULT_NUMBA_THREAD_CAP``.
    """
    maximum = config.NUMBA_NUM_THREADS

    if num_threads is None:
        current = get_num_threads()
        if "NUMBA_NUM_THREADS" in os.environ or current < maximum:
            return current
        return min(DEFAULT_NUMBA_THREAD_CAP, max(1, maximum // 2))

    if num_threads == "all":
        return maximum
    if isinstance(num_threads, bool) or not isinstance(num_threads, int):
        raise TypeError("num_threads must be a positive integer, 'all', or None")
    if not 1 <= num_threads <= maximum:
        raise ValueError(
            f"num_threads must be between 1 and {maximum}, or 'all'"
        )
    return num_threads


@contextmanager
def numba_thread_limit(num_threads: int | str | None = None) -> Iterator[int]:
    """Temporarily apply a Numba thread budget.

    Args:
        num_threads (int, str, or None, optional): Thread budget accepted by
            :func:`resolve_num_threads`. Defaults to None.

    Yields:
        int: Resolved thread count active inside the context.

    Notes:
        The previously active Numba thread count is restored when the context
        exits, including when an exception is raised.
    """
    previous = get_num_threads()
    requested = resolve_num_threads(num_threads)
    if requested != previous:
        set_num_threads(requested)
    try:
        yield requested
    finally:
        if requested != previous:
            set_num_threads(previous)


class diskenvelope():
    """Precalculate geometry and dimensionless disk-envelope quantities.

    Args:
        radius (np.ndarray or None, optional): Spherical radius normalized by
            the centrifugal radius. Defaults to None.
        theta (np.ndarray or None, optional): Polar angle in radians, measured
            from +z toward ``(x, y) = (sin(phi), -cos(phi))``. Defaults to
            None.
        phi (np.ndarray or None, optional): Azimuth in radians, measured from
            -y toward +x. Defaults to None.
        incl (float, optional): Inclination in radians used to project velocity
            onto the line of sight. Defaults to 0.
        H0 (float, optional): Disk scale height at cylindrical radius 1.
            Defaults to 0.2.
        plh (float, optional): Flaring exponent in
            ``H = H0 * R ** (1 + plh)``. Defaults to 0.25.
        pls (float, optional): Radial density exponent of the disk. Defaults to
            1.
    """

    def __init__(self, radius: np.ndarray | None = None,
                 theta: np.ndarray | None = None,
                 phi: np.ndarray | None = None, incl: float = 0,
                 H0: float = 0.2, plh: float = 0.25,
                 pls: float = 1.0) -> None:
        """Initialize disk-envelope coordinates and projection factors."""
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

    def get_mu0(self, mu: np.ndarray) -> np.ndarray:
        """Calculate the initial polar-angle cosine of an infalling streamline.

        Args:
            mu (np.ndarray): Absolute cosine of the current polar angle.

        Returns:
            np.ndarray: Streamline initial cosine ``mu0``, clipped to [0, 1].

        Notes:
            The cubic streamline equation is evaluated in three regimes to
            improve numerical stability around changes in its discriminant.
        """
        r = self.radius
        p = (r - 1) / 3.
        q = mu * r / 2.
        D = q**2 + p**3
        mu0 = np.full_like(r, np.nan)
        # three cases for computational stability
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
        """Calculate dimensionless UCM-envelope velocity and density.

        Returns:
            tuple: Radial, polar, and azimuthal velocity components followed
                by density. Values inside the region assigned to the disk are
                set to zero.
        """
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

    def disk(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate dimensionless Keplerian disk velocity and density.

        Returns:
            tuple: Azimuthal velocity and density. Values outside the region
                assigned to the disk are set to zero.
        """
        r = self.radius
        vp = self.sin_theta / np.sqrt(r)
        rho = r**(-self.pls) * self.H0 / self.H \
            * np.exp(-0.5 * (self.z / self.H)**2)
        c = self.where_disk
        vp[~c] = 0
        rho[~c] = 0
        return vp, rho


def rotbase(t: float | np.ndarray, p: float | np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate spherical-coordinate basis vectors.

    Args:
        t (float or np.ndarray): Polar angle in radians, measured from +z
            toward ``(x, y) = (sin(p), -cos(p))``.
        p (float or np.ndarray): Azimuth in radians, measured from -y toward
            +x.

    Returns:
        tuple: Radial, polar, and azimuthal basis vectors ``(er, et, ep)``.
    """
    er = np.array([np.sin(t) * np.sin(p), -np.sin(t) * np.cos(p), np.cos(t)])
    et = np.array([np.cos(t) * np.sin(p), -np.cos(t) * np.cos(p), -np.sin(t)])
    ep = np.array([np.cos(p), np.sin(p), np.zeros_like(p)])
    return er, et, ep


def XYZ2rtp(incl: float, phi: float,
            X: np.ndarray, Y: np.ndarray, Z: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert observer Cartesian coordinates to envelope spherical ones.

    Args:
        incl (float): Polar viewing angle in radians.
        phi (float): Azimuthal viewing angle in radians.
        X (np.ndarray): Observer-frame X coordinates.
        Y (np.ndarray): Observer-frame Y coordinates with the same shape as
            ``X``.
        Z (np.ndarray): Observer-frame line-of-sight coordinates with the same
            shape as ``X``.

    Returns:
        tuple: Spherical radius, polar angle, and azimuth arrays with the same
            shape as ``X``.

    Notes:
        Observer and envelope coordinates are related by
        ``X * ep + Y * (-et) + Z * er = (x, y, z)``.
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
def _rho2tau_parallel(vlos: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Integrate density into global velocity bins in parallel.

    Args:
        vlos (np.ndarray): Line-of-sight velocity cube with shape
            ``(nx, ny, nz)``.
        rho (np.ndarray): Density cube with the same shape as ``vlos``.

    Returns:
        np.ndarray: Density summed along z for every velocity channel, with
            shape ``(nv, ny, nx)``.

    Notes:
        The module-level ``vedge`` array must contain ``nv + 1`` velocity-bin
        edges before this internal JIT-compiled function is called.
    """
    nv = len(vedge) - 1
    nx, ny, _ = np.shape(vlos)
    tau = np.zeros((nv, ny, nx))
    for i in prange(nv):
        mask = (vedge[i] <= vlos) * (vlos < vedge[i + 1])
        tau[i] = np.sum(mask * rho, axis=2).T
    return tau


def rho2tau(vlos: np.ndarray, rho: np.ndarray,
            num_threads: int | str | None = None) -> np.ndarray:
    """Integrate density by velocity channel with bounded parallelism.

    Args:
        vlos (np.ndarray): Line-of-sight velocity cube with shape
            ``(nx, ny, nz)``.
        rho (np.ndarray): Density cube with the same shape as ``vlos``.
        num_threads (int, str, or None, optional): Numba thread budget accepted
            by :func:`resolve_num_threads`. Defaults to None.

    Returns:
        np.ndarray: Density summed along z for every velocity channel, with
            shape ``(nv, ny, nx)``.

    Notes:
        Set the module-level ``vedge`` array to the velocity-bin edges before
        calling this function.
    """
    with numba_thread_limit(num_threads):
        return _rho2tau_parallel(vlos, rho)


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
elos_r = {'major': [None] * lmax, 'minor': [None] * lmax}
elos_t = {'major': [None] * lmax, 'minor': [None] * lmax}
elos_p = {'major': [None] * lmax, 'minor': [None] * lmax}
t = {'major': [None] * lmax, 'minor': [None] * lmax}
idx_t = {'major': [None] * lmax, 'minor': [None] * lmax}
r_org = {'major': [None] * lmax, 'minor': [None] * lmax}
j_org = {'major': [None] * lmax, 'minor': [None] * lmax}


def update(radius_org: np.ndarray, theta: np.ndarray, phi: np.ndarray, incl: float,
           axis: str, l: int) -> None:
    """Cache geometry and lookup indices for one nested model-grid level.

    Args:
        radius_org (np.ndarray): Physical spherical radii of grid samples.
        theta (np.ndarray): Polar angles in radians.
        phi (np.ndarray): Azimuthal angles in radians.
        incl (float): Inclination in radians used for line-of-sight projection.
        axis (str): PV-cut axis, normally ``'major'`` or ``'minor'``.
        l (int): Nested-grid level stored in the module-level caches.
    """
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
    """Interpolate cached disk-envelope density and line-of-sight velocity.

    Args:
        Rc (float): Centrifugal radius in the same unit as the radii cached by
            :func:`update`.
        rho_jump (float): Disk-to-envelope density scaling.
        alphainfall (float): Scaling applied to radial infall velocity.
        axis (str): PV-cut axis used in :func:`update`.
        l (int): Nested-grid level used in :func:`update`.

    Returns:
        tuple: Dimensionless density and line-of-sight velocity arrays for the
            requested grid level.
    """
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
