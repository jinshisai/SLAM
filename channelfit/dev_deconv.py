import numpy as np
from scipy.signal import fftconvolve


def fwhm_to_sigma(fwhm):
    """Convert FWHM to Gaussian sigma."""
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def make_gaussian_psf(
    shape,
    fwhm_major_pix,
    fwhm_minor_pix=None,
    pa_deg=0.0,
    normalize=True,
):
    """
    Create a 2D elliptical Gaussian PSF on a pixel grid.

    Parameters
    ----------
    shape : tuple[int, int]
        Output shape as (ny, nx).
    fwhm_major_pix : float
        Major-axis FWHM in pixels.
    fwhm_minor_pix : float or None
        Minor-axis FWHM in pixels. If None, circular beam is assumed.
    pa_deg : float
        Position angle in degrees, measured counterclockwise from +x axis.
    normalize : bool
        If True, normalize PSF sum to 1.

    Returns
    -------
    psf : ndarray
        2D PSF array centered in the middle of the image.
    """
    ny, nx = shape
    if fwhm_minor_pix is None:
        fwhm_minor_pix = fwhm_major_pix

    sigma_x = fwhm_to_sigma(fwhm_major_pix)
    sigma_y = fwhm_to_sigma(fwhm_minor_pix)

    y, x = np.indices((ny, nx), dtype=float)
    x0 = (nx - 1) / 2.0
    y0 = (ny - 1) / 2.0
    x = x - x0
    y = y - y0

    theta = np.deg2rad(pa_deg)
    ct = np.cos(theta)
    st = np.sin(theta)

    xp =  ct * x + st * y
    yp = -st * x + ct * y

    psf = np.exp(-0.5 * ((xp / sigma_x) ** 2 + (yp / sigma_y) ** 2))

    if normalize:
        s = psf.sum()
        if s > 0:
            psf /= s

    return psf


def _periodic_distance_axis(n):
    """
    Periodic distance from index 0 on a length-n grid:
    [0, 1, 2, ..., floor(n/2), ..., 2, 1]
    """
    idx = np.arange(n)
    return np.minimum(idx, n - idx)


def make_periodic_gp_kernel(
    shape,
    kernel="matern32",
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
        kimg = sigma_f**2 * np.exp(-0.5 * (r / ell) ** 2)
    elif kernel == "matern32":
        z = np.sqrt(3.0) * r / ell
        kimg = sigma_f**2 * (1.0 + z) * np.exp(-z)
    elif kernel == "matern52":
        z = np.sqrt(5.0) * r / ell
        kimg = sigma_f**2 * (1.0 + z + z**2 / 3.0) * np.exp(-z)
    else:
        raise ValueError("kernel must be one of: 'rbf', 'matern32', 'matern52'")

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


def gp_deconvolve_2d(
    image,
    psf,
    noise_std,
    kernel="matern32",
    sigma_f=None,
    length_scale_pix=3.0,
    pad_factor=2,
    clip_positive=False,
    eps=1e-12,
    noise_clip_threshold = 3.,
):
    """
    GP-prior deconvolution of a 2D image using Fourier-domain posterior mean.

    Model
    -----
    y = H f + n
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
    kernel : {"rbf", "matern32", "matern52"}
        GP kernel.
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

    if sigma_f is None:
        sigma_f = np.nanstd(image)
        if not np.isfinite(sigma_f) or sigma_f <= 0:
            sigma_f = 1.0

    # Pad image and PSF to a larger grid to reduce periodic wrap-around.
    image_pad, _ = _pad_to_shape(image, padded_shape)
    psf_pad, _ = _pad_to_shape(psf, padded_shape)

    # Normalize PSF if needed.
    psf_sum = psf_pad.sum()
    if psf_sum <= 0:
        raise ValueError("PSF sum must be positive")
    psf_pad = psf_pad / psf_sum

    # Build GP kernel on the padded grid.
    kimg = make_periodic_gp_kernel(
        padded_shape,
        kernel=kernel,
        sigma_f=sigma_f,
        length_scale_pix=length_scale_pix,
    )

    # Fourier-domain quantities.
    #
    # PSF is assumed centered in the middle of the array.
    # ifftshift moves its center to [0,0] for correct FFT convolution.
    Hhat = np.fft.fft2(np.fft.ifftshift(psf_pad))
    Yhat = np.fft.fft2(image_pad)

    # Prior power for each Fourier mode.
    # Numerical round-off can make tiny negative values; clip them.
    Shat = np.real(np.fft.fft2(kimg))
    Shat = np.maximum(Shat, 0.0)

    noise_var = noise_std**2

    # Posterior mean in Fourier space:
    # F_post = S H* / (|H|^2 S + sigma_n^2) * Y
    denom = (np.abs(Hhat) ** 2) * Shat + noise_var
    denom = np.maximum(denom, eps)

    Ghat = Shat * np.conj(Hhat) / denom
    Fhat_post = Ghat * Yhat

    # Back to image space.
    f_post_pad = np.real(np.fft.ifft2(Fhat_post))

    if clip_positive:
        f_post_pad = np.maximum(f_post_pad, 0.0)

    if noise_clip_threshold > 0.:
        #noise_dec = noise_std * np.sqrt(np.mean(np.abs(Ghat)**2))
        noise_dec = estimate_noise(f_post_pad)
        f_post_pad[f_post_pad < noise_clip_threshold * noise_dec] = 0.

    # Reconvolution for consistency check.
    reconv_pad = np.real(np.fft.ifft2(Hhat * np.fft.fft2(f_post_pad)))

    # Crop back to original image size.
    deconvolved = _crop_center(f_post_pad, image.shape)
    reconvolved = _crop_center(reconv_pad, image.shape)
    residual = image - reconvolved

    return {
        "deconvolved": deconvolved,
        "reconvolved": reconvolved,
        "residual": residual,
        "posterior_filter": Ghat,
        "prior_power": Shat,
    }


def convolve_with_psf(image, psf):
    """
    Convenience function to beam-convolve an image using FFT convolution.
    """
    psf = np.asarray(psf, dtype=float)
    psf = psf / psf.sum()
    return fftconvolve(image, psf, mode="same")



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