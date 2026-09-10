# -*- coding: utf-8 -*-
"""
This script makes model channel maps from the observed mom0 by assuming 2D velocity pattern.
The main class ChannelFit can be imported to do each steps separately.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy import constants, units, wcs
from scipy.signal import convolve
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import curve_fit
from scipy.special import erf
import warnings
from tqdm import tqdm
from utils import emcee_corner, ReadFits, rot

warnings.simplefilter('ignore', RuntimeWarning)

GG = constants.G.si.value
M_sun = constants.M_sun.si.value
au = units.au.to('m')
vunit = np.sqrt(GG * M_sun / au) * 1e-3


def avefour(a: np.ndarray) -> np.ndarray:
    b = (a[:, 0::2, 0::2] + a[:, 0::2, 1::2]
         + a[:, 1::2, 0::2] + a[:, 1::2, 1::2]) / 4.
    return b


def makemom012(d: np.ndarray, v: np.ndarray, sigma: float,
               threshold: float = 3) -> dict:
    dmasked = np.nan_to_num(d)
    dv = np.min(v[1:] - v[:-1])
    mom0 = np.sum(dmasked, axis=0) * dv
    sigma_mom0 = sigma * dv * np.sqrt(len(d))
    dmasked[dmasked < threshold * sigma] = 0
    dsum = np.sum(dmasked, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        mom1 = np.sum(dmasked * v[:, None, None], axis=0) / dsum
        Iv2 = np.sum(dmasked * (v[:, None, None] - mom1)**2, axis=0)
        mom2 = np.sqrt(Iv2 / dsum)
    mom1[mom0 < threshold * sigma_mom0] = np.nan
    mom2[mom0 < threshold * sigma_mom0] = np.nan
    return {'mom0': mom0, 'mom1': mom1, 'mom2': mom2,
            'sigma_mom0': sigma_mom0}


def clean(data: np.ndarray, beam: np.ndarray, sigma: float,
          threshold: float = 2, gain: float = 0.01,
          weakestcomponent: float = 0.3,
          savetxt: str | None = None, loadtxt: str | None = None) -> np.ndarray:
    if loadtxt is not None:
        print(f'Load deconvolved moment 0 from {loadtxt}.')
        cleancomponent = np.loadtxt(loadtxt)
        return cleancomponent

    shape = np.shape(data)
    cleancomponent = data * 0
    cleanresidual = data * 1
    beamarea = np.sum(beam)  # pixel/beam
    cc0 = np.zeros_like(data)
    rms = 10000 * sigma
    for i in range(10000000):
        if i == 10000000 - 1:
            print('\n10,000,000 iterations achived in CLEAN.')
            break
        if (peak := np.nanmax(cleanresidual)) < threshold * sigma:
            print('\nThreshold achieved in CLEAN. '
                  f'(rms={rms / sigma:.2f}sigma, '
                  f'peak={peak / sigma:.2f}sigma)')
            break
        print(f'\rCLEAN reached {peak / sigma:.2f}sigma in Moment 0.  ', end='')
        ip, jp = np.unravel_index(np.nanargmax(cleanresidual), shape)
        cc = cc0 * 1  # for deep copy
        lp = cleanresidual[ip, jp]
        cc[ip, jp] = max(gain * lp, weakestcomponent * sigma) / beamarea  # Jy/pixel
        newresidual = cleanresidual - convolve(cc, beam, mode='same')
        rms = np.sqrt(np.nanmean(newresidual**2))
        cleancomponent = cleancomponent + cc
        cleanresidual = newresidual
    cleancomponent = cleancomponent + cleanresidual / beamarea
    if savetxt is not None:
        np.savetxt(savetxt, cleancomponent)
    return cleancomponent


def modeldeconvolve(data: np.ndarray, x: np.ndarray, y: np.ndarray,
                    bmaj: float, bmin: float, bpa: float, sigma: float,
                    savetxt: str | None = None, loadtxt: str | None = None,
                    direct: bool = False, progressbar: bool = True) -> tuple:
    nx = len(x)
    ny = len(y)
    dx = np.abs(x[1] - x[0])
    dy = np.abs(y[1] - y[0])
    xskip = int(np.floor(bmin / 2 / dx))
    yskip = int(np.floor(bmaj / 2 / dy))
    nxh = int(np.floor((nx - 1) / 2 / xskip)) * xskip
    nyh = int(np.floor((ny - 1) / 2 / yskip)) * yskip
    nxnew = 2 * nxh + 1
    nynew = 2 * nyh + 1
    xoff = int((nx - nxnew) / 2)
    yoff = int((ny - nynew) / 2)
    xi = x[::-1]
    yi = y
    di = data[:, ::-1]
    xi = xi[xoff:nxnew + xoff]
    yi = yi[yoff:nynew + yoff]
    di = di[yoff:nynew + yoff, xoff:nxnew + xoff]
    Xi, Yi = np.meshgrid(xi, yi)
    Xg = (Xi - xi[nxh]) / (bmin / 2)
    Yg = (Yi - yi[nyh]) / (bmaj / 2)
    g = np.exp2(-Xg**2 - Yg**2)
    gsum = np.sum(g)
    xmodel = xi[::xskip]
    ymodel = yi[::yskip]
    xnpar = len(xmodel)
    ynpar = len(ymodel)
    f = RGI((yi, xi), di, method='linear',
            bounds_error=False, fill_value=0)
    drot = f(tuple(rot(Xi, Yi, -np.radians(bpa)))[::-1])
    Par0 = drot[::yskip, ::xskip].clip(0, None) / gsum
    if loadtxt is not None:
        popt = np.loadtxt(loadtxt)
        print(f'Load a deconvolved model of moment 0 from {loadtxt}.')
    elif direct:
        def model(x, *par):
            values = np.reshape(par, (ynpar, xnpar))
            f = RGI((ymodel, xmodel), values, method='linear',
                    bounds_error=False, fill_value=0)
            f = convolve(f(tuple(x)), g, mode='same')
            return np.ravel(f)
        p0 = np.ravel(Par0)
        bounds = [np.zeros_like(p0), np.full_like(p0, np.max(drot))]
        popt, _ = curve_fit(model, [Yi, Xi], np.ravel(drot),
                            p0=p0, bounds=bounds)
        print('Found a deconvolved solution.')
    else:
        niter = 20
        if progressbar:
            bar = tqdm(total=niter * ynpar * xnpar)
            bar.set_description('Deconvolution')
        for _ in range(niter):
            ilist = np.arange(ynpar)
            jlist = np.arange(xnpar)
            for i_p in ilist:
                i_d = i_p * yskip
                for j_p in jlist:
                    j_d = j_p * xskip
                    if progressbar:
                        bar.update(1)
                    p0 = Par0[i_p, j_p]
                    dd = drot[i_d, j_d]
                    if dd < 2 * sigma:
                        Par0[i_p, j_p] = 0
                    else:
                        bounds = [0, dd]

                        def model(x, par):
                            values = Par0 + 0
                            values[i_p, j_p] = par
                            f = RGI((ymodel, xmodel), values, method='linear',
                                    bounds_error=False, fill_value=0)
                            ff = f(tuple(x))
                            gg = np.roll(g, (i_d - nyh, j_d - nxh), axis=(0, 1))
                            return np.sum(ff * gg)
                        popt, _ = curve_fit(model, [Yi, Xi], dd, p0=p0,
                                            sigma=[sigma], absolute_sigma=True,
                                            bounds=bounds)
                        Par0[i_p, j_p] = popt[0]
        print('Found a deconvolved solution.')
        print('')
        popt = np.ravel(Par0)
    if savetxt is not None:
        np.savetxt(savetxt, popt)
    zmodel = np.reshape(popt, (ynpar, xnpar))
    f = RGI((ymodel, xmodel), zmodel,
            method='linear', bounds_error=False, fill_value=0)
    decon = f(tuple(rot(*np.meshgrid(x, y), np.radians(bpa)))[::-1])
    return decon, xmodel, ymodel, zmodel


def ftdeconvolve(data: np.ndarray, x: np.ndarray, y: np.ndarray,
                 bmaj: float, bmin: float, bpa: float,
                 tikhonov_threshold: float = 6.25e-2,
                 savetxt: str | None = None, loadtxt: str | None = None
                 ) -> np.ndarray:
    """
    Deconvolve an image with a Gaussian beam using zero-order Tikhonov
    regularization in Fourier space.

    Notes
    -----
    The FFT calculation assumes periodic (circular) convolution, so boundary
    pixels can be affected by wrap-around and other edge artifacts. The input
    image should contain an emission-free margin of at least one beam major
    axis around the scientifically useful region; a wider margin is
    preferable.

    A cosine taper is applied within approximately one beam major-axis width
    of the boundary after calculating the Tikhonov solution. The returned
    image is therefore a deliberately edge-tapered version of that solution,
    and pixels in the tapered region should not be interpreted quantitatively.

    For an even-sized axis, the first row or column is temporarily omitted to
    construct an odd-sized FFT grid and restored as zeros after deconvolution.
    This assumes that the input boundaries contain no significant emission.
    """
    if loadtxt is not None:
        print(f'Load deconvolved moment 0 from {loadtxt}.')
        dnew = np.loadtxt(loadtxt)
        return dnew

    xd = x[int(len(x) % 2 == 0):]
    yd = y[int(len(y) % 2 == 0):]
    d = data[int(len(y) % 2 == 0):, int(len(x) % 2 == 0):]
    ny, nx = np.shape(d)
    nyh, nxh = (ny - 1) // 2, (nx - 1) // 2
    dx, dy = x[1] - x[0], y[1] - y[0]
    xg = np.linspace(-nxh * dx, nxh * dx, nx)
    yg = np.linspace(-nyh * dy, nyh * dy, ny)
    s, t = rot(*np.meshgrid(xg, yg), np.radians(bpa))
    g = np.exp2(-4 * ((t / bmaj)**2 + (s / bmin)**2))
    u = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    v = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    u, v = np.meshgrid(u, v)
    phase0 = 2 * np.pi * (u * (xg[-1] + dx) + v * (yg[-1] + dy))
    FTg = np.fft.fftshift(np.fft.fft2(g)) * np.exp(-1j * phase0)
    phase0 = 2 * np.pi * (u * (xd[-1] + dx) + v * (yd[-1] + dy))
    FTd = np.fft.fftshift(np.fft.fft2(d)) * np.exp(-1j * phase0)
    abs_FTg = np.abs(FTg)
    thre = tikhonov_threshold * np.max(abs_FTg)
    inverse_filter = np.conj(FTg) / (abs_FTg**2 + thre**2)
    # ----------------------------------------------------------------
    # FTdnew is the zero-order Tikhonov solution on the periodic FFT grid:
    #
    #   argmin_X sum(|FTg * X - FTd|^2 + thre^2 * |X|^2).
    #
    # By Parseval's theorem, this is equivalent, up to the common FFT
    # normalization, to minimizing
    #
    #   sum(|g (*) dnew - d|^2 + thre^2 * |dnew|^2),
    #
    # where (*) denotes circular convolution. The image-domain edge taper
    # applied below is intentional post-processing and changes the returned
    # image from this exact minimizer.
    # ----------------------------------------------------------------
    FTdnew = FTd * inverse_filter
    print('Tikhonov regularization is used for Fourier-space deconvolution '
          + f'with a transition at {tikhonov_threshold:.4f} times'
          + ' the FT[beam] peak.')
    dnew = np.real(np.fft.ifft2(np.fft.ifftshift(FTdnew * np.exp(1j * phase0))))
    if len(x) % 2 == 0:
        dnew = np.concatenate((np.zeros((np.shape(dnew)[0], 1)), dnew), axis=1)
    if len(y) % 2 == 0:
        dnew = np.concatenate((np.zeros((1, np.shape(dnew)[1])), dnew), axis=0)
    edge_width = int(bmaj / min(abs(dx), abs(dy)) + 0.5)
    if edge_width > 0:
        # Suppress unreliable boundary behavior caused by circular
        # convolution. This post-processing assumes that scientifically useful
        # emission is separated from the boundary by a sufficiently wide,
        # emission-free margin.
        if 2 * edge_width >= min(dnew.shape):
            warnings.warn(
                'The edge-taper regions overlap or occupy the entire image. '
                'Use a wider input image for mom0ft deconvolution.'
            )
        iy = np.arange(np.shape(dnew)[0])
        ix = np.arange(np.shape(dnew)[1])
        ydist = np.minimum(iy, iy[::-1])
        xdist = np.minimum(ix, ix[::-1])
        dist = np.minimum(ydist[:, None], xdist[None, :])
        taper = np.ones_like(dnew)
        edge = dist < edge_width
        taper[edge] = 0.5 * (1 - np.cos(np.pi * dist[edge] / edge_width))
        dnew = dnew * taper
    if savetxt is not None:
        np.savetxt(savetxt, dnew)
    return dnew


def _periodic_distance_axis(n):
    """
    Periodic distance from index 0 on a length-n grid:
    [0, 1, 2, ..., floor(n/2), ..., 2, 1]
    """
    idx = np.arange(n)
    return np.minimum(idx, n - idx)


def make_periodic_gp_kernel(
    shape,
    kernel="rbf",
    sigma_f=1.0,
    length_scale_pix=3.0,
):
    """
    Create a stationary GP covariance kernel image on a periodic grid.

    This kernel image is used to build the covariance eigenvalues by FFT.

    Parameters
    ----------
    shape : tuple[int, int]
        Shape (ny, nx).
    kernel : {"rbf", "matern32", "matern52"}
        Choice of GP kernel.
    sigma_f : float
        Prior standard deviation of the latent image.
    length_scale_pix : float
        Correlation length in pixels.

    Returns
    -------
    kimg : ndarray
        Kernel image whose FFT gives the prior power per Fourier mode.
    """
    ny, nx = shape
    dy = _periodic_distance_axis(ny)
    dx = _periodic_distance_axis(nx)
    yy, xx = np.meshgrid(dy, dx, indexing="ij")
    r = np.sqrt(xx**2 + yy**2)

    ell = float(length_scale_pix)
    if ell <= 0:
        raise ValueError("length_scale_pix must be > 0")

    if kernel == "rbf":
        kimg = np.exp(-0.5 * (r / ell) ** 2)
    elif kernel == "matern32":
        z = np.sqrt(3.0) * r / ell
        kimg = (1.0 + z) * np.exp(-z)
    elif kernel == "matern52":
        z = np.sqrt(5.0) * r / ell
        kimg = (1.0 + z + z**2 / 3.0) * np.exp(-z)
    else:
        raise ValueError("kernel must be one of: 'rbf', 'matern32', 'matern52'")

    # normalize
    kimg /= kimg.sum()
    kimg *= sigma_f**2

    return kimg


def _pad_to_shape(arr, out_shape):
    """Zero-pad a 2D array to out_shape, centering the original array."""
    in_y, in_x = arr.shape
    out_y, out_x = out_shape
    if in_y > out_y or in_x > out_x:
        raise ValueError("Input array is larger than output shape.")

    y0 = (out_y - in_y) // 2
    x0 = (out_x - in_x) // 2

    out = np.zeros(out_shape, dtype=float)
    out[y0:y0 + in_y, x0:x0 + in_x] = arr
    return out, (y0, x0)


def _crop_center(arr, shape):
    """Crop the central region of arr to shape."""
    in_y, in_x = arr.shape
    out_y, out_x = shape
    y0 = (in_y - out_y) // 2
    x0 = (in_x - out_x) // 2
    return arr[y0:y0 + out_y, x0:x0 + out_x]


def gpdeconvolve(
    image,
    psf,
    noise_std,
    bmaj,
    bmin,
    dx,
    dy,
    kernel="rbf",
    sigma_f=None,
    scale_length=2.0,
    pad_factor=2,
    clip_positive=False,
    eps=1e-12,
    noise_clip_threshold = 3.,
):
    """
    Gaussian prior deconvolution of a 2D image using Fourier-domain posterior mean
    based on Gaussian Process (GP).

    Model
    -----
    y = H f + n,
    where f is the true image to guess, H is the beam convolution function,
    n is the intrinsic thermal noise, and y is the observed image.
    f ~ GP(0, K)
    n ~ N(0, noise_std^2 I)

    Parameters
    ----------
    image : ndarray
        Observed 2D image (ny, nx).
    psf : ndarray
        2D beam/PSF image. Should be centered in the middle of the array
        and normalized so that psf.sum() = 1.
    noise_std : float
        Standard deviation of the image noise in the same intensity units
        as the image.
    bmaj : float
        Beam major FWHM. Unit can be arbitral
        but must be the same as that of the pixel length.
    bmin : float
        Beam minor FWHM. Unit can be arbitral
        but must be the same as that of the pixel length.
    dx : float
        pixel size along x axis. Unit can be arbitral
        but must be the same as that of the beam size.
    dy : float
        pixel size along y axis. Unit can be arbitral
        but must be the same as that of the beam size.
    kernel : {"rbf", "matern32", "matern52"}
        GP kernel. Default is RBF.
    sigma_f : float or None
        Prior standard deviation of the latent image.
        If None, estimated from the image standard deviation.
    length_scale_pix : float
        GP correlation length in pixels.
    pad_factor : int
        Zero-padding factor to reduce FFT wrap-around artifacts.
        1 means no extra padding. 2 is a good default.
    clip_positive : bool
        If True, clip negative values in the deconvolved image to zero.
    eps : float
        Small floor to avoid division by zero.

    Returns
    -------
    result : dict
        Dictionary containing:
        - "deconvolved": posterior mean latent image
        - "reconvolved": posterior mean convolved back with the PSF
        - "residual": image - reconvolved
        - "posterior_filter": Fourier-space Wiener/GP filter
        - "prior_power": GP prior power spectrum
    """
    image = np.asarray(image, dtype=float)
    psf = np.asarray(psf, dtype=float)

    if image.ndim != 2 or psf.ndim != 2:
        raise ValueError("image and psf must both be 2D arrays")
    if noise_std <= 0:
        raise ValueError("noise_std must be > 0")
    if pad_factor < 1:
        raise ValueError("pad_factor must be >= 1")

    ny, nx = image.shape
    py = int(pad_factor * ny)
    px = int(pad_factor * nx)
    padded_shape = (py, px)

    # Pad image and PSF to a larger grid to reduce periodic wrap-around.
    image_pad, _ = _pad_to_shape(image, padded_shape)
    psf_pad, _ = _pad_to_shape(psf, padded_shape)

    # intrinsic noise
    beam_area = bmaj * bmin * np.pi / (4.*np.log(2.))    # in au^2 for default
    pix_area  = dx * dy                                  # in au for default
    sig_int = noise_std * np.sqrt(2. * pix_area / beam_area)    # deconvolved noise in Jy/pixel
    #sig_int *= np.sqrt(nx * ny)    # Jy/pixel to Jy in Fourier space
    omega_source = image[image >= 3 * noise_std].size
    sig_int *= np.sqrt(omega_source)    # Jy/pixel to Jy over the source in Fourier space
    #print('Input noise: %.2e Jy/beam'%noise_std)
    #print('Deconvolved noise: %.2e Jy/beam'%(sig_int * beam_area / pix_area))

    # Normalize PSF if needed.
    psf_sum = psf_pad.sum()
    if psf_sum <= 0:
        raise ValueError("PSF sum must be positive")
    psf_pad = psf_pad / psf_sum

    # Fourier-domain quantities.
    # PSF is assumed centered in the middle of the array.
    # ifftshift moves its center to [0,0] for correct FFT convolution.
    Hhat = np.fft.fft2(np.fft.ifftshift(psf_pad))
    Yhat = np.fft.fft2(image_pad)

    if sigma_f is None:
        sigma_f = np.nanmax(np.abs(Yhat)) * pix_area / beam_area    # flux in Jy
    if not np.isfinite(sigma_f) or sigma_f <= 0:
        print('WARNING\tgpdeconvolve: Flux is infinity or negative.')
        print('WARNING\tgpdeconvolve: Set sigma_f to sigma_int.')
        sigma_f = sig_int

    # Build GP kernel on the padded grid.
    length_scale_pix = 0.5 * (bmaj + bmin) / (2.0 * np.sqrt(2.0 * np.log(2.0))) / scale_length
    kimg = make_periodic_gp_kernel(
        padded_shape,
        kernel=kernel,
        sigma_f=sigma_f,
        length_scale_pix=length_scale_pix,
    )     #/ (px * py)**0.5    # to Jy/pixel in the image domain

    # Prior power for each Fourier mode.
    # Numerical round-off can make tiny negative values; clip them.
    Shat = np.real(np.fft.fft2(kimg))
    Shat = np.maximum(Shat, 0.0)
    #print('Source flux: %.2e Jy'%sigma_f)
    #print('Deconvolved visibility noise: %.2e Jy'%(sig_int))

    # Posterior mean in Fourier space:
    # F_post = Shat / (Shat + sig_int^2) / Hhat * Y
    denom = Shat + sig_int**2
    denom = np.maximum(denom, eps)
    Ghat  = Shat / denom
    Fhat_post = Ghat / Hhat * Yhat

    # Back to image space.
    f_post_pad = np.real(np.fft.ifft2(Fhat_post))

    if clip_positive:
        f_post_pad = np.maximum(f_post_pad, 0.0)

    if noise_clip_threshold > 0.:
        noise_dec = estimate_noise(f_post_pad)
        f_post_pad[f_post_pad < noise_clip_threshold * noise_dec] = 0.

    # Reconvolution for consistency check.
    reconv_pad = np.real(np.fft.ifft2(Hhat * np.fft.fft2(f_post_pad)))

    # Crop back to original image size.
    deconvolved = _crop_center(f_post_pad, image.shape)
    reconvolved = _crop_center(reconv_pad, image.shape)
    residual = image - reconvolved

    # for check
    yfreq = np.fft.fftfreq(py)    # unshifted; np.fft.fftshift if shifted
    xfreq = np.fft.fftfreq(px)    # unshifted; np.fft.fftshift if shifted

    return {
        "deconvolved": deconvolved * pix_area / beam_area,    # in Jy/pixel
        "reconvolved": reconvolved,
        "residual": residual,
        "FT_image": Yhat * pix_area / beam_area,    # in Jy
        "FT_beam": Hhat,
        "posterior_filter": Ghat,
        "yfreq": yfreq,
        "xfreq": xfreq,
    }

def _diagnose_gpdeconvolution(result, data, outname = None):
    '''
    Make diagnostic plots for GP deconvolution.
    '''
    # radial plot
    qy, qx   = np.meshgrid(result['yfreq'], result['xfreq'], indexing="ij")
    q_radius = np.sqrt(qx**2 + qy**2)
    n_bins   = np.max([len(result['yfreq']), len(result['xfreq'])]) // 2
    k_max    = np.nanmax(q_radius)

    outname_prof = 'gpdeconv_diagnostic_profiles.png'
    outname_maps = 'gpdeconv_diagnostic_maps.png'
    if outname is not None:
        outname_prof = outname + '_' + outname_prof
        outname_maps = outname + '_' + outname_maps

    # figure
    fig, axes = plt.subplots(1,2, figsize = (8.27, 3.6))
    ax1, ax2 = axes
    cmap = plt.get_cmap('viridis')

    amp_profs = []
    for _d, label, ci in zip(
        [result['FT_beam'], result['posterior_filter']],
        ['Beam', 'GP filter'],
        [0.3, 0.6]):
        k_profile, amp_profile = radial_profile(
            np.abs(_d),
            q_radius,
            n_bins=n_bins,
            r_max=k_max,
        )
        ax1.plot(k_profile, amp_profile, label = label,
            color = cmap(ci), lw = 2.)
        amp_profs.append(amp_profile)
    for _d, label in zip([result['FT_image']], ['FT[img]']):
        k_profile, amp_profile = radial_profile(
            np.abs(_d),
            q_radius,
            n_bins=n_bins,
            r_max=k_max,
        )
        ax1.plot(k_profile, amp_profile / np.nanmax(amp_profile), label = label,
            color = cmap(0.), lw=2)
        amp_profs.append(amp_profile)
    ax1.legend()
    ax1.set_ylabel('Normalized amplitude')

    ax2.set_ylabel('Amplitude')
    ax2.plot(k_profile, amp_profs[-1],
        color = cmap(0.), lw = 2, label = 'FT[img] (Before deconv.)')
    ax2.plot(k_profile, amp_profs[1] / amp_profs[0] * amp_profs[-1],
        color = cmap(0.3), lw = 2, label = 'FT[img] (Deconvolved)')
    ax2.legend()

    k_plt_max = np.nanmin(k_profile[amp_profs[0] <= 4.e-5])
    for ax in axes:
        ax.set_xlim(0, k_plt_max)
        ax.set_xlabel('Frequency')
    fig.tight_layout()
    fig.savefig(outname_prof, dpi = 300)
    #plt.show()

    # 2D plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axs = axes.ravel()

    # data
    im0 = axs[0].imshow(data, origin="lower")
    axs[0].set_title("Observed data")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(result['deconvolved'], origin="lower")
    axs[1].set_title("Model (deconvolved)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046)

    im2 = axs[2].imshow(result['reconvolved'], origin="lower")
    axs[2].set_title("Model (convolved)")
    plt.colorbar(im2, ax=axs[2], fraction=0.046)

    im3 = axs[3].imshow(result['residual'], origin="lower")
    axs[3].set_title("Residual")
    plt.colorbar(im3, ax=axs[3], fraction=0.046)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(outname_maps, dpi = 300)
    #plt.show()


def estimate_noise(_d, nitr=1000, thr=2.):
    '''
    Estimate map noise

    _d (array): Data
    nitr (int): Number of the maximum iteration
    thr (float): Threshold of the each iteration
    '''

    d = _d.copy().ravel()
    rms = np.sqrt(np.nanmean(d*d))
    for i in range(nitr):
        rms_p = rms
        d[d >= thr*rms] = np.nan
        rms = np.sqrt(np.nanmean(d*d))

        if (rms - rms_p)*(rms - rms_p) < 1e-20:
            return rms

    print('Reach maximum number of iteration.')
    return rms


def radial_profile(
    values: np.ndarray,
    radius: np.ndarray,
    n_bins: int,
    r_max: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Return an azimuthally averaged radial profile."""
    valid = np.isfinite(values) & np.isfinite(radius)
    if r_max is None:
        r_max = float(np.nanmax(radius[valid]))
    edges = np.linspace(0.0, r_max, n_bins + 1)
    which = np.digitize(radius[valid], edges) - 1
    in_range = (0 <= which) & (which < n_bins)
    which = which[in_range]
    vals = values[valid][in_range]

    sums = np.bincount(which, weights=vals, minlength=n_bins)
    counts = np.bincount(which, minlength=n_bins)
    profile = np.full(n_bins, np.nan, dtype=float)
    filled = counts > 0
    profile[filled] = sums[filled] / counts[filled]
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, profile


class ChannelFit(ReadFits):
    """Fit a Keplerian disk or infalling envelope model directly to channel maps.

    Args:
        disk (bool, optional): Whether to include the disk component. Defaults
            to True.
        envelope (bool, optional): Whether to include the envelope component.
            Defaults to False.
        scaling (str, optional): Method used to set the model intensity.
            ``'uniform'`` does not use the observed moment-0 map as a spatial
            template: the intensity follows the ``pI`` and ``Ienv`` model,
            and one global scale factor is fitted to the cube. Each method
            whose name begins with ``'mom0'`` instead normalizes the
            velocity-integrated model at every sky position to a deconvolved
            observed moment-0 map. ``'mom0clean'`` obtains that map with
            CLEAN, ``'mom0model'`` uses a nonnegative model-grid
            deconvolution, and ``'mom0ft'`` uses zero-order Tikhonov
            regularization in Fourier space. Defaults to ``'uniform'``.
        progressbar (bool, optional): Whether to display progress bars during
            deconvolution and fitting. Defaults to True.
    """

    def __init__(self,
                 disk: bool = True,
                 envelope: bool = False,
                 scaling: str = 'uniform',
                 progressbar: bool = True,
                 scaling_gp_args: dict = {
                 'scale_length': 2.,
                 'kernel': 'rbf',
                 'sigma_f': None,
                 'pad_factor': 2,
                 'noise_clip_threshold': 3.}) -> None:
        """Initialize channel-map fitting options."""
        self.paramkeys = ['Mstar', 'Rc', 'cs', 'h1', 'h2',
                          'pI', 'Rin', 'Ienv',
                          'xoff', 'yoff', 'voff', 'incloff', 'paoff']
        self.disk = disk
        self.envelope = envelope
        self.scaling = scaling
        self.progressbar = progressbar
        self.scaling_gp_args = scaling_gp_args

    def makegrid(self, cubefits: str | None = None,
                 pa: float = 0, incl: float = 90, dist: float = 1,
                 center: str | None = None, vsys: float = 0,
                 rmax: float = 1e4,
                 vlim: tuple[float, float, float, float] = (-100, 0, 0, 100),
                 sigma: float | None = None, nlayer: int = 3,
                 xskip: int = 1, yskip: int = 1,
                 skipto: int | bool | None = False,
                 gaussmargin: float = 1.6,
                 tikhonov_threshold: float = 6.25e-2,
                 savedeconvolved: str | None = None,
                 loaddeconvolved: str | None = None,
                 signmajor: int | None = None,
                 signminor: int | None = None) -> None:
        """Read a cube and prepare the observational and nested model grids.

        Args:
            cubefits (str or None, optional): Input channel-map FITS file.
                Defaults to None.
            pa (float, optional): Position angle of the disk major axis in
                degrees. Defaults to 0.
            incl (float, optional): Disk inclination in degrees. Defaults to
                90.
            dist (float, optional): Source distance in pc. Defaults to 1.
            center (str or None, optional): Sky coordinates of the model
                center. Defaults to None.
            vsys (float, optional): Systemic velocity in km/s. Defaults to 0.
            rmax (float, optional): Half-width of the fitted area in au.
                Defaults to 1e4.
            vlim (tuple, optional): Boundaries of the fitted blue and red
                velocity ranges, relative to ``vsys``, in km/s. Defaults to
                (-100, 0, 0, 100).
            sigma (float or None, optional): RMS noise of the cube. None means
                automatic estimation. Defaults to None.
            nlayer (int, optional): Number of nested model-grid layers.
                Defaults to 3.
            xskip (int, optional): Pixel stride along the x axis. Defaults to
                1.
            yskip (int, optional): Pixel stride along the y axis. Defaults to
                1.
            skipto (int, bool, or None, optional): Approximate number of
                pixels per beam minor axis after resampling. False or None
                disables automatic resampling. Defaults to False.
            gaussmargin (float, optional): Beam-kernel margin in units of the
                beam major axis. Defaults to 1.6.
            tikhonov_threshold (float, optional): Regularization threshold for
                Fourier deconvolution. Defaults to 6.25e-2.
            savedeconvolved (str or None, optional): File in which to save the
                deconvolved moment-0 map. Defaults to None.
            loaddeconvolved (str or None, optional): File from which to load a
                previously deconvolved moment-0 map. Defaults to None.
            signmajor (int or None, optional): Sign of the rotational
                line-of-sight velocity. +1 makes the positive major-axis side (i.e., pa)
                redshifted and -1 makes it blueshifted. Zero suppresses the
                rotational contribution. None determines the sign from the
                observed moment-1 map. Defaults to None.
            signminor (int or None, optional): Sign of the radial-infall
                line-of-sight velocity. For inward motion, +1 makes the
                positive minor-axis side blueshifted (i.e., pa+90) and -1 makes it
                redshifted. Zero suppresses the radial contribution. This
                option affects only a model with radial motion, such as when
                ``envelope=True``. None determines the sign from the observed
                moment-1 map. Defaults to None.
        """
        if cubefits is not None:
            self.read_cubefits(cubefits, center, dist, vsys,
                               -rmax, rmax, -rmax, rmax, None, None,
                               xskip, yskip, sigma)
            dpix = min([np.abs(self.dx), np.abs(self.dy)])
            if type(skipto) is int:
                iskip = int(self.bmin / (dpix / xskip) / skipto)
                if iskip == 0:
                    print('WARNING: \'skipto\' is ignored because the beam minor axis is smaller than \'skipto\' pixels.')
                    iskip = 1
                self.read_cubefits(cubefits, center, dist, vsys,
                                   -rmax, rmax, -rmax, rmax, None, None,
                                   iskip, iskip, sigma)
                dpix = min([np.abs(self.dx), np.abs(self.dy)])
                ibmaj = self.bmaj / dpix
                ibmin = self.bmin / dpix
                print(f'Adopt xskip={iskip:d} and yskip={iskip:d}.')
                print(f'Beam major/minor axis is {ibmaj:.1f}/{ibmin:.1f} pixels.')
            v = self.v
        self.incl0 = incl
        pa_rad = np.radians(pa)
        self.pa_rad = pa_rad
        self.cospa = np.cos(pa_rad)
        self.sinpa = np.sin(pa_rad)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.v_nanblue = v[v < vlim[0]]
        self.v_blue = v[(vlim[0] <= v) * (v <= vlim[1])]
        self.v_nanmid = v[(vlim[1] < v) * (v < vlim[2])]
        self.v_red = v[(vlim[2] <= v) * (v <= vlim[3])]
        self.v_nanred = v[vlim[3] < v]
        self.v_valid = np.r_[self.v_blue, self.v_red]

        self.data_blue = self.data[(vlim[0] <= v) * (v <= vlim[1])]
        self.data_red = self.data[(vlim[2] <= v) * (v <= vlim[3])]
        self.data_valid = np.append(self.data_blue, self.data_red, axis=0)

        m = makemom012(self.data_valid, self.v_valid, sigma)
        self.mom0 = m['mom0']
        self.mom1 = m['mom1']
        self.mom2 = m['mom2']
        self.sigma_mom0 = m['sigma_mom0']
        X, Y = rot(self.X, self.Y, pa_rad)
        if signmajor is None:
            self.signmajor = np.sign(np.nansum(self.mom1 * Y))
        else:
            self.signmajor = signmajor
        if signminor is None:
            self.signminor = np.sign(np.nansum(self.mom1 * X)) * (-1)
        else:
            self.signminor = signminor

        # 2d nested grid on the disk plane.
        # x and y are minor and major axis coordinates before projection.
        r_need = rmax + gaussmargin * self.bmaj
        npix = int(2 * r_need / dpix + 0.5)
        npix = int(4 * np.ceil(npix / 4))
        self.nq1 = npix // 2 - npix // 2 // 2
        self.nq3 = self.nq1 + npix // 2
        self.nlayer = nlayer  # down to dpix / 2**(nlayer-1)
        xnest = [None] * nlayer
        ynest = [None] * nlayer
        Xnest = [None] * nlayer
        Ynest = [None] * nlayer
        Rnest = [None] * nlayer
        for l in range(nlayer):
            n = npix // 2 - 0.5
            s = np.linspace(-n, n, npix) * dpix / 2**l
            X, Y = np.meshgrid(s, s)
            R = np.hypot(X, Y)
            xnest[l] = s
            ynest[l] = s
            Xnest[l] = X
            Ynest[l] = Y
            Rnest[l] = R
        self.xnest = np.array(xnest)
        self.ynest = np.array(ynest)
        self.Xnest = np.array(Xnest)
        self.Ynest = np.array(Ynest)
        self.Xnest, self.Ynest = rot(self.Xnest, self.Ynest, pa_rad)
        self.Xnest0 = self.Xnest * 1
        self.Ynest0 = self.Ynest * 1
        print(f'{len(self.v_valid):d} channels will be fitted.')
        print('-------- nested grid --------')
        for l in range(len(xnest)):
            print(f'x, dx, npix: +/-{xnest[l][-1]:.2f},'
                  + f' {xnest[l][1]-xnest[l][0]:.2f} au,'
                  + f' {npix:d}')
        print('-----------------------------')

        ngauss = int(gaussmargin * self.bmaj / dpix + 0.5)  # 0.5 is for rounding
        xb = (np.arange(2 * ngauss + 1) - ngauss) * dpix
        yb = (np.arange(2 * ngauss + 1) - ngauss) * dpix
        xb, yb = rot(*np.meshgrid(xb, yb), np.radians(self.bpa))
        gaussbeam = np.exp2(-4 * ((yb / self.bmaj)**2 + (xb / self.bmin)**2))
        self.pixperbeam = np.sum(gaussbeam)
        self.gaussbeam = gaussbeam  # The 1st (x) axis is in the model order.

        n_need = int(r_need / dpix + 0.5)
        self.ineed0 = npix // 2 - n_need
        self.ineed1 = npix // 2 + n_need
        self.xneed = self.xnest[0][self.ineed0:self.ineed1]
        self.yneed = self.ynest[0][self.ineed0:self.ineed1]

        if 'mom0' in self.scaling:
            # Change the 1st (x) axis to the observational order.
            self.gaussbeam = self.gaussbeam[:, ::-1]
        if self.scaling == 'mom0clean':
            self.mom0decon = clean(data=self.mom0, beam=self.gaussbeam,
                                   sigma=self.sigma_mom0, threshold=2,
                                   savetxt=savedeconvolved,
                                   loadtxt=loaddeconvolved)
        elif self.scaling == 'mom0model':
            d = modeldeconvolve(x=self.x, y=self.y, data=self.mom0,
                                bmaj=self.bmaj, bmin=self.bmin, bpa=self.bpa,
                                sigma=self.sigma_mom0,
                                savetxt=savedeconvolved,
                                loadtxt=loaddeconvolved,
                                progressbar=self.progressbar)
            self.mom0decon, self.xdecon, self.ydecon, self.zdecon = d
        elif self.scaling == 'mom0ft':
            self.mom0decon = ftdeconvolve(x=self.x, y=self.y, data=self.mom0,
                                          bmaj=self.bmaj, bmin=self.bmin,
                                          bpa=self.bpa,
                                          tikhonov_threshold=tikhonov_threshold,
                                          savetxt=savedeconvolved,
                                          loadtxt=loaddeconvolved)
        elif self.scaling == 'mom0gp':
            res = gpdeconvolve(self.mom0, self.gaussbeam, self.sigma_mom0,
                self.bmaj, self.bmin, np.abs(self.dx), np.abs(self.dy),
                **self.scaling_gp_args)
            self.mom0decon = res["deconvolved"]
        if 'mom0' in self.scaling:
            c = convolve(self.mom0decon, self.gaussbeam, mode='same')
            self.resdecon = self.mom0 - c
            maxres = np.max(self.resdecon) / self.sigma_mom0
            rmsres = np.sqrt(np.mean(self.resdecon**2)) / self.sigma_mom0
            print(f'Max and rms are {maxres:.1f}sigma '
                  + f'and {rmsres:.1f}sigma in Moment 0 residual.')

    def diagnose_gpdeconvolution(self, outname = None):
        res = gpdeconvolve(self.mom0, self.gaussbeam, self.sigma_mom0,
                self.bmaj, self.bmin, np.abs(self.dx), np.abs(self.dy),
                **self.scaling_gp_args)
        _diagnose_gpdeconvolution(res, self.mom0, outname = outname)

    def update_pa(self, pa: float):
        self.Xnest, self.Ynest = rot(self.Xnest0, self.Ynest0, np.radians(pa))

    def update_incl(self, incl: float):
        i = np.radians(self.incl0 + incl)
        self.sini = np.sin(i)
        self.cosi = np.cos(i)
        self.tani = np.tan(i)

    def update_xdisk(self, h1: float, h2: float = -1):
        x = [None] * 4
        for i, hdisk in zip([0, 2], [h1, h2]):
            if hdisk < 0:
                x1 = None
                x2 = None
            elif hdisk < 0.01:
                x1 = self.Xnest / self.cosi
                x2 = self.Xnest / self.cosi
            else:
                Xcosi = self.Xnest * self.cosi
                a = self.tani**(-2) - hdisk**2
                b = (1 + hdisk**2) * Xcosi
                c = (self.tani**2 - hdisk**2) * Xcosi**2 \
                    - hdisk**2 * self.Ynest**2
                if -1e-3 < a < 1e-3:
                    x1 = Xcosi + c / b / 2
                    x2 = None
                else:
                    zsini1 = np.full_like(self.Xnest, np.nan)
                    zsini2 = np.full_like(self.Xnest, np.nan)
                    c = (D := b**2 - a * c) >= 0
                    sqrtD = np.sqrt(D[c])
                    zsini1[c] = (b[c] + sqrtD) / a
                    zsini2[c] = (b[c] - sqrtD) / a
                    x1 = Xcosi + zsini1
                    x2 = Xcosi + zsini2
            x[i], x[i + 1] = x1, x2
        self.xdisk = x

    def update_prof(self, cs: float):
        cs_over_dv = cs / self.dv
        w = max([cs_over_dv * 2.35482, 1])  # 2.35482 ~ sqrt(8ln2)
        vmax_over_w = 2  # in the unit of max(FWHM, dv)
        w_over_d = 11
        vmax = vmax_over_w * w
        d = w / w_over_d
        n = 2 * int(vmax_over_w * w_over_d) + 1
        v = np.linspace(-vmax, vmax, n)
        if cs_over_dv < 0.01:
            p = (1 + np.sign(v + 0.5)) * (1 - np.sign(v - 0.5)) / 4
        else:
            p = erf((v + 0.5) / np.sqrt(2) / cs_over_dv) \
                - erf((v - 0.5) / np.sqrt(2) / cs_over_dv)
            p[0] = p[n - 1] = 0
        self.prof, self.prof_n, self.prof_d = p, n - 1, d

    def update_getvlos(self, Rc: float, Rin: float):
        def getvlos(x_in: np.ndarray | None, h_in: float):
            if x_in is None:
                return None
            r = np.hypot(x_in, self.Ynest)
            c = r > Rc
            vp = r**(-1/2) * (1 + h_in**2)**(-3/4)
            vr = r * 0
            if self.envelope:
                vp[c] = np.sqrt(Rc) / r[c]
                vr[c] = -r[c]**(-1/2) * np.sqrt(2 - Rc / r[c])
            erot = self.Ynest * self.signmajor / r
            erad = x_in * self.signminor / r
            vlos = (vp * erot + vr * erad) * self.sini * vunit
            vlos[r < Rin] = np.nan
            if not self.envelope:
                vlos[c] = np.nan
            if not self.disk:
                vlos[~c] = np.nan
            return vlos
        self.getvlos = getvlos

    def update_vlos(self, h1: float, h2: float):
        self.vlos = [self.getvlos(x, h) for x, h in zip(self.xdisk, [h1, h1, h2, h2])]

    def get_Iunif(self, Mstar: float, Rc: float, pI: float,
                  Ienv: float, offvsys: float) -> np.ndarray:
        Iunif = 0
        for vlos_in, x_in in zip(self.vlos, self.xdisk):
            if vlos_in is None:
                continue
            vlos = vlos_in * np.sqrt(Mstar)
            v = np.subtract.outer(self.v_valid, vlos) - offvsys  # v, layer, y, x
            iv = v / self.dv / self.prof_d + self.prof_n // 2 + 0.5  # 0.5 is for rounding
            p = self.prof[iv.astype('int').clip(0, self.prof_n)]
            r = np.hypot(x_in, self.Ynest)
            p = np.where(r < Rc, p, p * Ienv) / (Ienv + 1)
            if pI != 0:
                p = p * r**(-pI)
            Iunif = Iunif + np.nan_to_num(p)
        for l in range(self.nlayer - 1, 0, -1):
            Iunif[:, l - 1, self.nq1:self.nq3, self.nq1:self.nq3] \
                = avefour(Iunif[:, l, :, :])
        Iunif = Iunif[:, 0, self.ineed0:self.ineed1, self.ineed0:self.ineed1]  # v, y, x
        return Iunif

    def rgi2d(self, xoff: float, yoff: float,
              I_in: np.ndarray) -> np.ndarray:
        Iout = [None] * len(I_in)
        for i, c in enumerate(I_in):
            interp = RGI((self.yneed, self.xneed), c, method='linear',
                         bounds_error=False, fill_value=0)
            Iout[i] = interp((self.Y - yoff, self.X - xoff))
        Iout = np.array(Iout)
        return Iout

    def get_scale(self, Iout) -> float:
        fg = np.sum(Iout * self.data_valid)
        ff = np.sum(Iout * Iout)
        scale = 0.0 if ff == 0 else fg / ff
        return scale

    def cubemodel(self, Mstar: float, Rc: float, cs: float,
                  h1: float = 0, h2: float = -1, pI: float = 0,
                  Rin: float = 0, Ienv: float = 0,
                  xoff: float = 0, yoff: float = 0, voff: float = 0,
                  incloff: float = 90, paoff: float = 0,
                  convolving: bool = True):
        if self.free['paoff']:
            self.update_pa(paoff)
        if self.free['incloff']:
            self.update_incl(incloff)
        if self.free['cs']:
            self.update_prof(cs)
        if self.free['paoff'] or self.free['h1'] or self.free['h2']:
            self.update_xdisk(h1, h2)
        if self.free['paoff'] or self.free['Rc'] or self.free['Rin']:
            self.update_getvlos(Rc, Rin)
        if self.free['paoff'] or self.free['h1'] or self.free['h2'] \
                or self.free['Rc'] or self.free['Rin']:
            self.update_vlos(h1, h2)

        Iunif = self.get_Iunif(Mstar, Rc, pI, Ienv, voff)
        if 'mom0' in self.scaling:
            Iunif = self.rgi2d(xoff, yoff, Iunif)  # 1st axis in the observational order
            mom0unif = np.sum(Iunif, axis=0) * self.dv
            mom0unif[mom0unif < 0] = np.nan
            Iunif = Iunif * self.mom0decon / mom0unif
            Iunif = np.nan_to_num(Iunif)
        # The 1st (x) axis of Iunif is in the observational order
        # if 'mom0' in self.scaling because of rgi2d.
        # For this reason, self.gaussbeam is inverted in the x direction
        # in advance.
        Iout = convolve(Iunif, [self.gaussbeam], mode='same')
        if 'mom0' not in self.scaling:
            Iout = self.rgi2d(xoff, yoff, Iout)  # 1st axis in the observational order
            scale = self.get_scale(Iout)
            Iout = Iout * scale
            if not convolving:
                Iunif = self.rgi2d(xoff, yoff, Iunif)
                Iunif = Iunif * scale
        if not convolving:
            Iout = Iunif
        return Iout

    def fitting(self, Mstar_range: list = [0.01, 10],
                Rc_range: list = [1, 1000],
                cs_range: list = [0.01, 1],
                h1_range: list = [0.01, 1],
                h2_range: list = [0.01, 1],
                pI_range: list = [-2, 2],
                Rin_range: list = [0, 1000],
                Ienv_range: list = [0.01, 100],
                xoff_range: list = [-100, 100],
                yoff_range: list = [-100, 100],
                voff_range: list = [-0.2, 0.2],
                incl_range: list = [-45, 45],
                pa_range: list = [-45, 45],
                fixed_params: dict = {},
                filename: str = 'channelfit',
                show: bool = False,
                save_result: bool = True,
                save_corner: bool = True,
                print_result: bool = True,
                kwargs_emcee_corner: dict = {}) -> dict:
        """Fit the channel-map model parameters with MCMC.

        Args:
            Mstar_range (list, optional): Prior range of stellar mass in solar
                masses. Defaults to [0.01, 10].
            Rc_range (list, optional): Prior range of disk radius in au.
                Defaults to [1, 1000].
            cs_range (list, optional): Prior range of line width in km/s.
                Defaults to [0.01, 1].
            h1_range (list, optional): Prior range of the first disk scale
                height divided by radius. Defaults to [0.01, 1].
            h2_range (list, optional): Prior range of the second disk scale
                height divided by radius. Always h1 < h2, regardless of h1_range and h2_range. Defaults to [0.01, 1].
            pI_range (list, optional): Prior range of the radial intensity
                power-law index. Defaults to [-2, 2].
            Rin_range (list, optional): Prior range of inner radius in au.
                Defaults to [0, 1000].
            Ienv_range (list, optional): Prior range of ``Ienv``, the
                intrinsic intensity immediately outside ``Rc`` divided by
                that immediately inside ``Rc``, before final intensity
                scaling. ``Ienv=0`` gives no envelope emission, ``Ienv=1``
                gives equal intensity across ``Rc``, values between 0 and 1
                make the envelope fainter than the disk, and values greater
                than 1 make it brighter. This parameter matters only when the
                envelope component is enabled. Defaults to [0.01, 100].
            xoff_range (list, optional): Prior range of x offsets in au.
                Defaults to [-100, 100].
            yoff_range (list, optional): Prior range of y offsets in au.
                Defaults to [-100, 100].
            voff_range (list, optional): Prior range of velocity offsets (i.e., the offset of systemic velocity) in
                km/s. Defaults to [-0.2, 0.2].
            incl_range (list, optional): Prior range of inclination offsets in
                degrees, from ``incl`` givne in ``makegrid``. Defaults to [-45, 45].
            pa_range (list, optional): Prior range of position-angle offsets
                in degrees, from ``pa`` given in ``makegrid``. Defaults to [-45, 45].
            fixed_params (dict, optional): Values of parameters to hold fixed.
                Unspecified parameters remain free. Defaults to {}.
            filename (str, optional): Prefix for fitting products. Defaults to
                ``'channelfit'``.
            show (bool, optional): Whether to show the corner plot. Defaults to
                False.
            save_result (bool, optional): Whether to save fitted parameter
                values. Defaults to True.
            save_corner (bool, optional): Whether to save the corner plot.
                Defaults to True.
            print_result (bool, optional): Whether to print fitted values.
                Defaults to True.
            kwargs_emcee_corner (dict, optional): Additional arguments passed
                to ``emcee_corner``. Defaults to {}.

        Returns:
            dict: Best-fit, lower, median, and upper parameter dictionaries,
            together with the reduced chi-square value.
        """

        p_fixed = {k: fixed_params[k] if k in fixed_params else None for k in self.paramkeys}
        self.free = {k: p_fixed[k] is None for k in self.paramkeys}

        if not self.free['paoff']:
            self.update_pa(p_fixed['paoff'])
        if not self.free['incloff']:
            self.update_incl(p_fixed['incloff'])
        if not self.free['cs']:
            self.update_prof(p_fixed['cs'])
        if not (self.free['paoff'] or self.free['h1'] or self.free['h2']):
            self.update_xdisk(p_fixed['h1'], p_fixed['h2'])
        if not (self.free['paoff'] or self.free['Rc'] or self.free['Rin']):
            self.update_getvlos(p_fixed['Rc'], p_fixed['Rin'])
        if not (self.free['paoff'] or self.free['h1'] or self.free['h2']
                or self.free['Rc'] or self.free['Rin']):
            self.update_vlos(p_fixed['h1'], p_fixed['h2'])

        p_fixed = np.array([p_fixed[k] for k in self.paramkeys])
        self.chain = None
        self.lnp = None
        notfixed = np.equal(p_fixed, None)
        runfit = None in p_fixed

        def chi2(q):
            model = self.cubemodel(*q)
            if not np.all(np.isfinite(model)):
                return np.inf
            return np.nansum((self.data_valid - model)**2) \
                / self.sigma**2 / self.pixperbeam

        def reduced_chi2(q):
            n_data = np.count_nonzero(np.isfinite(self.data_valid)) \
                / self.pixperbeam
            n_free = np.count_nonzero(notfixed)
            if 'mom0' not in self.scaling:
                n_free += 1
            dof = n_data - n_free
            return chi2(q) / dof if dof > 0 else np.nan

        if runfit:
            ilog = np.array([0, 1, 7])
            i = ilog[np.not_equal(p_fixed[ilog], None)]
            p_fixed[i] = np.log10(p_fixed[i].astype('float'))
            labels = np.array(self.paramkeys).copy()
            labels[ilog] = ['log'+labels[i] for i in ilog]
            labels = labels[notfixed]
            kwargs0 = {'nwalkers_per_ndim': 16, 'nburnin': 200,
                       'nsteps': 500, 'labels': labels,
                       'rangelevel': None, 'range_corner': None,
                       'figname': f'{filename}.corner.png',
                       'show_corner': show}
            kw = dict(kwargs0, **kwargs_emcee_corner)
            if not save_corner:
                kw['figname'] = None
            if self.progressbar:
                total = kw['nwalkers_per_ndim'] * len(p_fixed[notfixed])
                total *= kw['nburnin'] + kw['nsteps'] + 2
                bar = tqdm(total=total)
                bar.set_description('Within the ranges')

            def lnprob(p):
                if self.progressbar:
                    bar.update(1)
                q = p_fixed.copy()
                q[notfixed] = p
                q[ilog] = 10**q[ilog]
                h1, h2 = q[3], q[4]
                if min(h1, h2) >= 0 and h1 > h2:
                    return -np.inf

                return -0.5 * chi2(q)

            plim = np.array([Mstar_range, Rc_range,
                             cs_range, h1_range, h2_range,
                             pI_range, Rin_range, Ienv_range,
                             xoff_range, yoff_range, voff_range,
                             incl_range, pa_range])
            plim[ilog] = np.log10(plim[ilog])
            plim = plim[notfixed].T
            if type(r_c := kw['range_corner']) is dict:
                r_c = [r_c[k] if k in r_c else 0.8 for k in self.paramkeys]
                for i in ilog:
                    r_c[i] = r_c[i] if type(r_c[i]) is float else np.log10(r_c[i])
                r_c = [a for a, k in zip(r_c, self.paramkeys) if self.free[k]]
                kw['range_corner'] = r_c

            mcmc = emcee_corner(plim, lnprob, simpleoutput=False, **kw)
            i_mcmc = 4
            if kw.get('return_chain', False):
                chain_free = mcmc[i_mcmc]
                i_mcmc += 1
                chain = np.empty((len(p_fixed), chain_free.shape[1]), dtype=float)
                chain[notfixed] = chain_free
                chain[~notfixed] = p_fixed[~notfixed].astype(float)[:, None]
                chain[ilog] = 10**chain[ilog]
                self.chain = chain
            if kw.get('return_lnp', False):
                self.lnp = mcmc[i_mcmc]

            def get_p(i: int):
                p = p_fixed.copy()
                p[notfixed] = mcmc[i]
                p[ilog] = 10**p[ilog]
                return p
            self.popt = get_p(0)
            self.plow = get_p(1)
            self.pmid = get_p(2)
            self.phigh = get_p(3)
        else:
            self.popt = p_fixed
            self.plow = p_fixed
            self.pmid = p_fixed
            self.phigh = p_fixed

        self.chi2r = reduced_chi2(self.popt)

        self.pa_rad = self.pa_rad + np.radians(self.popt[12])
        self.sinpa = np.sin(self.pa_rad)
        self.cospa = np.cos(self.pa_rad)
        ulist = ['Msun', 'au', 'km/s', '', '', '', 'au', '',
                 'au', 'au', 'km/s', 'deg', 'deg']
        digits = [2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        if print_result:
            print('Parameter values (opt, low, mid, high):')
            for i, (k, d, u) in enumerate(zip(self.paramkeys, digits, ulist)):
                p = [self.popt[i], self.plow[i], self.pmid[i], self.phigh[i]]
                print(f'{k} = {p[0]:.{d}f}, {p[1]:.{d}f},'
                      + f' {p[2]:.{d}f}, {p[3]:.{d}f} {u}')
        if runfit and save_result:
            plist = [self.popt, self.plow, self.pmid, self.phigh]
            with open(filename+'.popt.txt', 'w') as f:
                f.write('#Rows:' + ','.join(self.paramkeys) + '\n')
                f.write('#Columns:' + ','.join(['popt', 'plow', 'pmid', 'phigh']) + '\n')
                np.savetxt(f, np.transpose(plist))
        self.popt = dict(zip(self.paramkeys, self.popt))
        self.plow = dict(zip(self.paramkeys, self.plow))
        self.pmid = dict(zip(self.paramkeys, self.pmid))
        self.phigh = dict(zip(self.paramkeys, self.phigh))
        return {'popt': self.popt, 'plow': self.plow, 'pmid': self.pmid,
                'phigh': self.phigh, 'chi2r': getattr(self, 'chi2r', None)}

    def make_model_products(self, **kwargs):
        h = self.header.copy()
        h['NAXIS1'] = len(self.x)
        h['NAXIS2'] = len(self.y)
        h['NAXIS3'] = len(self.v)
        h['CRPIX1'] = h['CRPIX1'] - self.offpix[0]
        h['CRPIX2'] = h['CRPIX2'] - self.offpix[1]
        h['CRPIX3'] = h['CRPIX3'] - self.offpix[2]
        nx = h['NAXIS1']
        ny = h['NAXIS2']
        for k in self.free.keys():
            self.free[k] = True
        p = self.popt if kwargs == {} else kwargs
        m = self.cubemodel(**p)
        m0 = self.cubemodel(**p, convolving=False)

        def concat(m):
            if len(self.v_red) > 0:
                m_blue = m[self.v_valid < np.min(self.v_red)]
            else:
                m_blue = m * 1
                m_red = np.full((0, ny, nx), np.nan)
            if len(self.v_blue) > 0:
                m_red = m[np.max(self.v_blue) < self.v_valid]
            else:
                m_red = m * 1
                m_blue = np.full((0, ny, nx), np.nan)
            nanblue = np.full((len(self.v_nanblue), ny, nx), np.nan)
            nanmid = np.full((len(self.v_nanmid), ny, nx), np.nan)
            nanred = np.full((len(self.v_nanred), ny, nx), np.nan)
            model = nanblue
            if len(m_blue) > 0:
                model = np.append(model, m_blue, axis=0)
            if len(nanmid) > 0:
                model = np.append(model, nanmid, axis=0)
            if len(m_red) > 0:
                model = np.append(model, m_red, axis=0)
            if len(nanred) > 0:
                model = np.append(model, nanred, axis=0)
            return model

        model = concat(m)
        return {'model': model,
                'residual': self.data - model,
                'beforeconvolving': concat(m0),
                'header': h}

    def modeltofits(self, filehead: str = 'best', **kwargs) -> None:
        """Write the best-fit model and residual cubes to FITS files.

        Args:
            filehead (str, optional): Prefix of the output FITS files.
                Defaults to ``'best'``.
            **kwargs: Model parameters. The stored best-fit parameters are
                used when no values are supplied.
        """
        w = wcs.WCS(naxis=3)
        products = self.make_model_products(**kwargs)

        def tofits(d: np.ndarray, ext: str):
            h = products['header'].copy()
            if ext == 'beforeconvolving':
                h['BUNIT'] = 'Jy/pixel'
                for k in ['BMAJ', 'BMIN', 'BPA']:
                    if k in h:
                        del h[k]
            header = w.to_header()
            hdu = fits.PrimaryHDU(d, header=header)
            for k in h.keys():
                if not ('COMMENT' in k or 'HISTORY' in k):
                    hdu.header[k] = h[k]
            hdu = fits.HDUList([hdu])
            hdu.writeto(f'{filehead}.{ext}.fits', overwrite=True)

        tofits(products['model'], 'model')
        tofits(products['residual'], 'residual')
        tofits(products['beforeconvolving'], 'beforeconvolving')

    def plotmom(self, mode: str, filename: str = 'mom01.png',
                save: bool = True, show: bool = False,
                **kwargs: float) -> None:
        """Plot moment-0 contours over a moment-1 image.

        Args:
            mode (str): Data product to plot: ``'obs'``, ``'model'``, or
                ``'residual'``.
            filename (str, optional): Output figure name. Defaults to
                ``'mom01.png'``.
            save (bool, optional): Whether to save the figure. Defaults to
                True.
            show (bool, optional): Whether to show the figure. Defaults to
                False.
            **kwargs: Model parameters used for model or residual plots.
        """
        if 'mod' in mode or 'res' in mode or 'clean' in mode:
            if kwargs != {}:
                self.popt = kwargs
            d = self.cubemodel(**self.popt)
            m = makemom012(d, self.v_valid, self.sigma)
        if 'obs' in mode:
            mom0 = self.mom0
            mom1 = self.mom1
            label = 'Obs.'
        elif 'mod' in mode:
            mom0 = m['mom0']
            mom1 = m['mom1']
            label = 'Model'
        elif 'res' in mode:
            mom0 = self.mom0 - m['mom0']
            mom1 = self.mom1 - m['mom1']
            label = r'Obs. $-$ model'
        levels = np.arange(1, 20) * 3 * self.sigma_mom0
        levels = np.sort(np.r_[-levels, levels])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        vplot = (np.nanpercentile(self.mom1, 99)
                 - np.nanpercentile(self.mom1, 1)) / 2.
        m = ax.pcolormesh(self.x, self.y, mom1, cmap='jet',
                          shading='nearest', vmin=-vplot, vmax=vplot)
        fig.colorbar(m, ax=ax, label=label + r' mom1 (km s$^{-1}$)')
        ax.contour(self.x, self.y, mom0, colors='gray', levels=levels)
        r = np.linspace(-1, 1, 3) * self.x.max() * 1.42
        ax.plot(r * self.sinpa, r * self.cospa, 'k:')
        ax.plot(r * self.cospa, -r * self.sinpa, 'k:')
        bpos = np.max(self.x) - 0.7 * self.bmaj
        e = Ellipse((bpos, -bpos), width=self.bmin, height=self.bmaj,
                    angle=self.bpa * np.sign(self.dx), facecolor='gray')
        ax.add_patch(e)
        ax.set_xlabel('R.A. offset (au)')
        ax.set_ylabel('Dec. offset (au)')
        ax.set_xlim(self.x.max() * 1.01, self.x.min() * 1.01)
        ax.set_ylim(self.y.min() * 1.01, self.y.max() * 1.01)
        ax.set_aspect(1)
        if save:
            fig.savefig(filename)
        if show:
            plt.show()
        plt.close()

    def plotdecon(self, filehead: str = 'test', save: bool = True,
                  show: bool = False):
        if not (hasattr(self, 'mom0decon') and hasattr(self, 'resdecon')):
            print('No deconvolution solutions and residual generated.')
            return
        cc = self.mom0decon / self.sigma_mom0
        cr = self.resdecon / self.sigma_mom0
        ccmax = np.max(cc)
        ccmin = np.min(cc)
        for c, vmin, vmax, s, ext in zip([cc, cr],
                                         [ccmin, -6],
                                         [ccmax, 6],
                                         ['deconvolved mom0', 'mom0 residual'],
                                         ['decon', 'resdecon']):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            m = ax.pcolormesh(self.x, self.y, c, cmap='jet',
                              shading='nearest', vmin=vmin, vmax=vmax)
            fig.colorbar(m, ax=ax, label=f'{s} / ' + r'$\sigma$')
            r = np.linspace(-1, 1, 3) * self.x.max() * 1.42
            ax.plot(r * self.sinpa, r * self.cospa, ':', color='gray')
            ax.plot(r * self.cospa, -r * self.sinpa, ':', color='gray')
            bpos = np.max(self.x) - 0.7 * self.bmaj
            e = Ellipse((bpos, -bpos), width=self.bmin, height=self.bmaj,
                        angle=self.bpa * np.sign(self.dx), facecolor='gray')
            ax.add_patch(e)
            ax.set_xlabel('R.A. offset (au)')
            ax.set_ylabel('Dec. offset (au)')
            ax.set_xlim(self.x.max() * 1.01, self.x.min() * 1.01)
            ax.set_ylim(self.y.min() * 1.01, self.y.max() * 1.01)
            ax.set_aspect(1)
            if save:
                fig.savefig(f'{filehead}.{ext}.png')
            if show:
                plt.show()
            plt.close()
