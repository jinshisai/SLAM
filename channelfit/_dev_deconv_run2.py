import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append('.')
sys.path.append('/Users/jinshi/Works/Git/jinshisai/SLAM/')
from dev_deconv import *
from channelfit import ChannelFit

def main():
    # -------- INPUTS --------
    cubefits = '../testfits/test.cube.fits'
    center = '04h39m53.878s +26d03m09.43s'
    pa = 2.0  # deg
    incl = 85  # deg
    vsys = 5.9  # km/s
    dist = 140  # pc
    sigma = 1.7e-3  # Jy/beam
    rmax = 200  # au; The fitted area is [-rmax, ramx] x [-rmax, rmax].
    vlim = [-3.6, -2.0, 2.0, 3.]
    outname = 'testfits'

    # image 2
    cubefits = '/Users/jinshi/Works/Myproject/V883Ori/Linecentroid/fitsimages/CH3OH/V883Ori_spw39_CH3OH_stacked_contsub_selfcal_rb05_cl.pbcor.subim.fits'
    center = '5h38m18.100s -7d02m25.99s'
    pa = 32.0  # deg
    incl = 38.  # deg
    vsys = 4.3  # km/s
    dist = 388.  # pc, Lee+19
    sigma = 0.71e-3  # Jy/beam
    rmax = 230  # au; The fitted area is [-rmax, ramx] x [-rmax, rmax].
    vlim = [-4.3, 0, 0, 4.3]  # km/s; Relative to vsys.
    outname = 'V883Ori'
    # ------------------------

    chan = ChannelFit(scaling='uniform', progressbar=True)
    chan.makegrid(cubefits=cubefits, center=center, pa=pa, incl=incl,
              vsys=vsys, dist=dist, sigma=sigma, rmax=rmax, vlim=vlim)

    # --------------------------------------------------
    # GP deconvolution
    # --------------------------------------------------
    mom0 = chan.mom0.copy()
    mom0_masked = chan.mom0.copy()
    mom0_masked[chan.mom0 < 3.*chan.sigma_mom0] = 0.
    one_third_beam = int(np.sqrt(chan.pixperbeam) // 1.)
    #print(one_third_beam)
    result = gp_deconvolve_2d(
        image=mom0,
        psf=chan.gaussbeam[:, ::-1],
        noise_std=sigma,
        kernel="matern32", # kernel doesn't change results much
        sigma_f=np.std(chan.mom0),
        length_scale_pix=one_third_beam,
        pad_factor=2,
        clip_positive=False,
        noise_clip_threshold = 3., #one_third_beam,
    )

    deconv = result["deconvolved"]
    reconv = result["reconvolved"]
    resid = result["residual"]


    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axs = axes.ravel()

    im0 = axs[0].imshow(chan.mom0, origin="lower")
    axs[0].set_title("Observed data")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(deconv, origin="lower")
    axs[1].set_title("Model (deconvolved)")
    #axs[1].contour(deconv, origin="lower", 
    #    levels = np.array([-6,-3,3,6,9])*chan.sigma_mom0, colors = 'white')
    axs[1].set_title("Model (deconvolved)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046)

    im2 = axs[2].imshow(reconv, origin="lower")
    axs[2].set_title("Model (convolved)")
    plt.colorbar(im2, ax=axs[2], fraction=0.046)

    im3 = axs[3].imshow(resid, origin="lower")
    axs[3].set_title("Residual")
    plt.colorbar(im3, ax=axs[3], fraction=0.046)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(outname + '_GPdeconv.pdf', transparent = True)
    plt.show()


    # comparison 2
    chan = ChannelFit(scaling='mom0ft', progressbar=True)
    chan.makegrid(cubefits=cubefits, center=center, pa=pa, incl=incl,
              vsys=vsys, dist=dist, sigma=sigma, rmax=rmax, vlim=vlim)
    deconv_ft = chan.mom0decon
    chan = ChannelFit(scaling='mom0clean', progressbar=True)
    chan.makegrid(cubefits=cubefits, center=center, pa=pa, incl=incl,
              vsys=vsys, dist=dist, sigma=sigma, rmax=rmax, vlim=vlim)
    deconv_cl = chan.mom0decon
    chan = ChannelFit(scaling='mom0model', progressbar=True)
    chan.makegrid(cubefits=cubefits, center=center, pa=pa, incl=incl,
              vsys=vsys, dist=dist, sigma=sigma, rmax=rmax, vlim=vlim)
    deconv_md = chan.mom0decon


    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axs = axes.ravel()

    im0 = axs[0].imshow(chan.mom0, origin="lower")
    axs[0].set_title("Observed data")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(deconv_ft, origin="lower")
    axs[1].set_title("Deconvolved (FT)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046)

    im2 = axs[2].imshow(deconv_cl, origin="lower")
    axs[2].set_title("Deconvolved (CLEAN)")
    plt.colorbar(im2, ax=axs[2], fraction=0.046)

    im3 = axs[3].imshow(deconv_md, origin="lower")
    axs[3].set_title("Deconvolved (Model)")
    plt.colorbar(im3, ax=axs[3], fraction=0.046)

    im3 = axs[4].imshow(deconv, origin="lower")
    axs[4].set_title("Deconvolved (GP)")
    plt.colorbar(im3, ax=axs[4], fraction=0.046)

    axs[5].set_axis_off()

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(outname + '_deconv_comp.pdf', transparent = True)
    plt.show()


if __name__ == '__main__':
    main()