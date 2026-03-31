import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append('.')
from dev_deconv import *


def main():
    # --------------------------------------------------
    # Example: create a smooth mock disk-like image
    # --------------------------------------------------
    ny, nx = 128, 128
    y, x = np.indices((ny, nx), dtype=float)
    x0 = (nx - 1) / 2
    y0 = (ny - 1) / 2
    x = x - x0
    y = y - y0

    # Inclined smooth disk-like structure
    q = 0.6
    pa = np.deg2rad(30.0)
    xp =  np.cos(pa) * x + np.sin(pa) * y
    yp = -np.sin(pa) * x + np.cos(pa) * y
    r_ell = np.sqrt(xp**2 + (yp / q)**2)

    true_image = np.exp(-(r_ell / 18.0) ** 2)
    true_image += 0.35 * np.exp(-((r_ell - 22.0) / 7.0) ** 2)  # broad ring-like enhancement

    # --------------------------------------------------
    # Beam / PSF
    # --------------------------------------------------
    psf = make_gaussian_psf(
        shape=(41, 41),
        fwhm_major_pix=8.0,
        fwhm_minor_pix=5.0,
        pa_deg=20.0,
    )

    # --------------------------------------------------
    # Observed image
    # --------------------------------------------------
    conv_image = convolve_with_psf(true_image, psf)

    rng = np.random.default_rng(0)
    noise_std = 0.02
    obs_image = conv_image + rng.normal(0.0, noise_std, size=conv_image.shape)

    # --------------------------------------------------
    # GP deconvolution
    # --------------------------------------------------
    result = gp_deconvolve_2d(
        image=obs_image,
        psf=psf,
        noise_std=noise_std,
        kernel="matern32",
        sigma_f=np.std(obs_image),
        length_scale_pix=4.0,
        pad_factor=2,
        clip_positive=False,
    )

    deconv = result["deconvolved"]
    reconv = result["reconvolved"]
    resid = result["residual"]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axs = axes.ravel()

    im0 = axs[0].imshow(true_image, origin="lower")
    axs[0].set_title("True image")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(obs_image, origin="lower")
    axs[1].set_title("Observed image")
    plt.colorbar(im1, ax=axs[1], fraction=0.046)

    im2 = axs[2].imshow(deconv, origin="lower")
    axs[2].set_title("GP deconvolved")
    plt.colorbar(im2, ax=axs[2], fraction=0.046)

    im3 = axs[3].imshow(conv_image, origin="lower")
    axs[3].set_title("True convolved")
    plt.colorbar(im3, ax=axs[3], fraction=0.046)

    im4 = axs[4].imshow(reconv, origin="lower")
    axs[4].set_title("Deconvolved -> reconvolved")
    plt.colorbar(im4, ax=axs[4], fraction=0.046)

    im5 = axs[5].imshow(resid, origin="lower")
    axs[5].set_title("Residual (obs - reconvolved)")
    plt.colorbar(im5, ax=axs[5], fraction=0.046)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()