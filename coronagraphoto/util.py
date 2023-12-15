from __future__ import division

# import corner
# import cv2
# import emcee
import os
import sys
import time

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# UTIL
# =============================================================================


def gen_wavelength_grid(bandpass, resolution):
    """
    Generates wavelengths that sample a Synphot bandpass at a given resolution.

    This function calculates wavelengths within the specified bandpass range
    such that each wavelength interval corresponds to a constant spectral resolution.
    The spectral resolution is defined as R = λ/Δλ, where λ is the wavelength and Δλ
    is the wavelength interval. The function iteratively adds these intervals starting
    from the lower limit of the bandpass until it reaches or surpasses the upper limit.

    Args:
        bandpass (synphot.SpectralElement):
            A synphot bandpass object with the "waveset" attribute that
            indicates the wavelengths necessary to sample the bandpass.
        resolution (float):
            The desired constant spectral resolution (R).
    Returns:
        wavelengths (astropy.units.quantity.Quantity):
            An array of wavelengths sampled across the bandpass at the
            specified resolution.
        delta_lambdas (astropy.units.quantity.Quantity):
            An array of wavelength intervals that correspond to the
            specified resolution.
    """
    first_wavelength = bandpass.waveset[0]
    last_wavelength = bandpass.waveset[-1]
    wavelengths = []
    delta_lambdas = []
    current_wavelength = first_wavelength
    while current_wavelength < last_wavelength:
        wavelengths.append(current_wavelength.value)
        delta_lambda = current_wavelength / resolution
        current_wavelength += delta_lambda
        delta_lambdas.append(delta_lambda.value)
    wavelengths = np.array(wavelengths) * first_wavelength.unit
    delta_lambdas = np.array(delta_lambdas) * first_wavelength.unit
    return wavelengths, delta_lambdas


def tdot(im, kernel):
    t0 = time.time()
    im_conv = np.tensordot(im, kernel)
    t1 = time.time()
    print("   %.3f s" % (t1 - t0))

    return im_conv


def movie_time(name, odir, fdir, wave_plot=0.5):  # micron
    print("Plotting images for time movie...")

    path = odir + name + "/RAW/sci_imgs.fits"
    imgs = pyfits.getdata(path, 0)
    star = pyfits.getdata(path[:-9] + "star.fits", 0)
    plan = pyfits.getdata(path[:-9] + "plan.fits", 0)
    disk = pyfits.getdata(path[:-9] + "disk.fits", 0)
    pixscale = pyfits.getheader(path, 0)["PIXSCALE"]  # lambda/D
    diam = pyfits.getheader(path, 0)["DIAM"]  # m
    time = pyfits.getdata(path, 1)  # yr
    wave = pyfits.getdata(path, 2) * 1e-6  # m
    path = fdir + name + "/RAW/"
    if not os.path.exists(path):
        os.makedirs(path)

    # Apply fliplr & invert_xaxis to invert x-axis labels but not image.
    ww = np.argmin(np.abs(wave - wave_plot * 1e-6))
    Ntime = len(time)
    vmin0 = np.log10(
        max([np.min(star[:, ww]), np.min(plan[:, ww]), np.min(disk[:, ww])])
    )
    vmax0 = np.log10(max([np.max(plan[:, ww]), np.max(disk[:, ww])]))
    vmin1 = np.log10(np.min(star[:, ww]))
    vmax1 = np.log10(np.max(star[:, ww]))
    vmin2 = np.log10(np.min(plan[:, ww]))
    vmax2 = np.log10(np.max(plan[:, ww]))
    vmin3 = np.log10(np.min(disk[:, ww]))
    vmax3 = np.log10(np.max(disk[:, ww]))
    for i in range(Ntime):
        sys.stdout.write("\r   Finished %.0f of %.0f times" % (i, Ntime))
        sys.stdout.flush()
        ext = (
            ((imgs.shape[2] - 1) // 2 * pixscale + 0.5 * pixscale)
            * wave[ww]
            / diam
            * rad2mas
        )
        f, ax = plt.subplots(1, 4, figsize=(4.8 * 4, 3.6 * 1))
        p0 = ax[0].imshow(
            np.fliplr(np.log10(imgs[i, ww])),
            origin="lower",
            extent=(-ext, ext, -ext, ext),
            vmin=vmin0,
            vmax=vmax0,
        )
        c0 = plt.colorbar(p0, ax=ax[0])
        c0.set_label("$\log_{10}$(ph/s/pix)", rotation=270, labelpad=20)
        ax[0].invert_xaxis()
        ax[0].set_xlabel("$\Delta$RA [mas]")
        ax[0].set_ylabel("$\Delta$DEC [mas]")
        ax[0].set_title("Scene")
        p1 = ax[1].imshow(
            np.fliplr(np.log10(star[i, ww])),
            origin="lower",
            extent=(-ext, ext, -ext, ext),
            vmin=vmin1,
            vmax=vmax1,
        )
        c1 = plt.colorbar(p1, ax=ax[1])
        c1.set_label("$\log_{10}$(ph/s/pix)", rotation=270, labelpad=20)
        ax[1].invert_xaxis()
        ax[1].set_xlabel("$\Delta$RA [mas]")
        ax[1].set_ylabel("$\Delta$DEC [mas]")
        ax[1].set_title("Star")
        p2 = ax[2].imshow(
            np.fliplr(np.log10(plan[i, ww])),
            origin="lower",
            extent=(-ext, ext, -ext, ext),
            vmin=vmin2,
            vmax=vmax2,
        )
        c2 = plt.colorbar(p2, ax=ax[2])
        c2.set_label("$\log_{10}$(ph/s/pix)", rotation=270, labelpad=20)
        ax[2].invert_xaxis()
        ax[2].set_xlabel("$\Delta$RA [mas]")
        ax[2].set_ylabel("$\Delta$DEC [mas]")
        ax[2].set_title("Planets")
        p3 = ax[3].imshow(
            np.fliplr(np.log10(disk[i, ww])),
            origin="lower",
            extent=(-ext, ext, -ext, ext),
            vmin=vmin3,
            vmax=vmax3,
        )
        c3 = plt.colorbar(p3, ax=ax[3])
        c3.set_label("$\log_{10}$(ph/s/pix)", rotation=270, labelpad=20)
        ax[3].invert_xaxis()
        ax[3].set_xlabel("$\Delta$RA [mas]")
        ax[3].set_ylabel("$\Delta$DEC [mas]")
        ax[3].set_title("Disk")
        plt.suptitle("t = %.3f yr" % time[i])
        plt.subplots_adjust(wspace=0.75)
        plt.savefig(path + "%03.0f" % i)
        plt.close()
    sys.stdout.write("\r   Finished %.0f of %.0f times" % (i + 1, Ntime))
    sys.stdout.flush()
    print("")


def movie_wave(name, odir, fdir, time_plot=0.0):  # yr
    print("Plotting images for wavelength movie...")

    path = odir + name + "/RAW/sci_imgs.fits"
    imgs = pyfits.getdata(path, 0)
    star = pyfits.getdata(path[:-9] + "star.fits", 0)
    plan = pyfits.getdata(path[:-9] + "plan.fits", 0)
    disk = pyfits.getdata(path[:-9] + "disk.fits", 0)
    pixscale = pyfits.getheader(path, 0)["PIXSCALE"]  # lambda/D
    diam = pyfits.getheader(path, 0)["DIAM"]  # m
    time = pyfits.getdata(path, 1)  # yr
    wave = pyfits.getdata(path, 2) * 1e-6  # m
    path = fdir + name + "/RAW/"
    if not os.path.exists(path):
        os.makedirs(path)

    imgs = pyfits.getdata(path, 0)
    star = pyfits.getdata(path[:-9] + "star.fits", 0)
    plan = pyfits.getdata(path[:-9] + "plan.fits", 0)
    disk = pyfits.getdata(path[:-9] + "disk.fits", 0)
    pixscale = pyfits.getheader(path, 0)["PIXSCALE"]  # lambda/D
    diam = pyfits.getheader(path, 0)["DIAM"]  # m
    time = pyfits.getdata(path, 1)  # yr
    wave = pyfits.getdata(path, 2) * 1e-6  # m
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # Apply fliplr & invert_xaxis to invert x-axis labels but not image.
    ww = np.argmin(np.abs(time - time_plot))
    Nwave = len(wave)
    vmin0 = np.log10(
        max([np.min(star[ww, :]), np.min(plan[ww, :]), np.min(disk[ww, :])])
    )
    vmax0 = np.log10(max([np.max(plan[ww, :]), np.max(disk[ww, :])]))
    vmin1 = np.log10(np.min(star[ww, :]))
    vmax1 = np.log10(np.max(star[ww, :]))
    vmin2 = np.log10(np.min(plan[ww, :]))
    vmax2 = np.log10(np.max(plan[ww, :]))
    vmin3 = np.log10(np.min(disk[ww, :]))
    vmax3 = np.log10(np.max(disk[ww, :]))
    for i in range(Nwave):
        sys.stdout.write("\r   Finished %.0f of %.0f wavelengths" % (i, Nwave))
        sys.stdout.flush()
        ext = (
            ((imgs.shape[2] - 1) // 2 * pixscale + 0.5 * pixscale)
            * wave[i]
            / diam
            * rad2mas
        )
        f, ax = plt.subplots(1, 4, figsize=(4.8 * 4, 3.6 * 1))
        p0 = ax[0].imshow(
            np.fliplr(np.log10(imgs[ww, i])),
            origin="lower",
            extent=(-ext, ext, -ext, ext),
            vmin=vmin0,
            vmax=vmax0,
        )
        c0 = plt.colorbar(p0, ax=ax[0])
        c0.set_label("$\log_{10}$(ph/s/pix)", rotation=270, labelpad=20)
        ax[0].invert_xaxis()
        ax[0].set_xlabel("$\Delta$RA [mas]")
        ax[0].set_ylabel("$\Delta$DEC [mas]")
        ax[0].set_title("Scene")
        p1 = ax[1].imshow(
            np.fliplr(np.log10(star[ww, i])),
            origin="lower",
            extent=(-ext, ext, -ext, ext),
            vmin=vmin1,
            vmax=vmax1,
        )
        c1 = plt.colorbar(p1, ax=ax[1])
        c1.set_label("$\log_{10}$(ph/s/pix)", rotation=270, labelpad=20)
        ax[1].invert_xaxis()
        ax[1].set_xlabel("$\Delta$RA [mas]")
        ax[1].set_ylabel("$\Delta$DEC [mas]")
        ax[1].set_title("Star")
        p2 = ax[2].imshow(
            np.fliplr(np.log10(plan[ww, i])),
            origin="lower",
            extent=(-ext, ext, -ext, ext),
            vmin=vmin2,
            vmax=vmax2,
        )
        c2 = plt.colorbar(p2, ax=ax[2])
        c2.set_label("$\log_{10}$(ph/s/pix)", rotation=270, labelpad=20)
        ax[2].invert_xaxis()
        ax[2].set_xlabel("$\Delta$RA [mas]")
        ax[2].set_ylabel("$\Delta$DEC [mas]")
        ax[2].set_title("Planets")
        p3 = ax[3].imshow(
            np.fliplr(np.log10(disk[ww, i])),
            origin="lower",
            extent=(-ext, ext, -ext, ext),
            vmin=vmin3,
            vmax=vmax3,
        )
        c3 = plt.colorbar(p3, ax=ax[3])
        c3.set_label("$\log_{10}$(ph/s/pix)", rotation=270, labelpad=20)
        ax[3].invert_xaxis()
        ax[3].set_xlabel("$\Delta$RA [mas]")
        ax[3].set_ylabel("$\Delta$DEC [mas]")
        ax[3].set_title("Disk")
        plt.suptitle("λ = %.3f μm" % (wave[i] * 1e6))
        plt.subplots_adjust(wspace=0.75)
        plt.savefig(path + "%03.0f" % i)
        plt.close()
    sys.stdout.write("\r   Finished %.0f of %.0f wavelengths" % (i + 1, Nwave))
    sys.stdout.flush()
    print("")


def movie_make(name, fdir):
    print("Making movie from images...")

    path = fdir + name + "/RAW/"
    pngfiles = [path + f for f in os.listdir(path) if f.endswith(".png")]
    pngfiles = sorted(pngfiles)

    imgs = []
    for pngfile in pngfiles:
        img = cv2.imread(pngfile)
        height, width, layers = img.shape
        size = (width, height)
        imgs += [img]

    out = cv2.VideoWriter(
        path + "_movie.mp4", cv2.VideoWriter_fourcc(*"MP4V"), 24, size
    )
    for i in range(len(imgs)):
        out.write(imgs[i])
    out.release()

    pass


def proj_pc(xx, yy, inc, Omega):  # deg  # deg
    # Rotate.
    rho = np.sqrt(xx**2 + yy**2)
    phi = np.rad2deg(np.arctan2(yy, xx))
    phi -= Omega
    xx_rot = rho * np.cos(np.deg2rad(phi))
    yy_rot = rho * np.sin(np.deg2rad(phi))

    # Incline.
    xx_inc = xx_rot
    yy_inc = yy_rot / np.cos(np.deg2rad(inc))

    # Convert.
    rho = np.sqrt(xx_inc**2 + yy_inc**2)
    phi = np.rad2deg(np.arctan2(yy_inc, xx_inc))
    phi -= 90.0
    phi = ((phi + 180.0) % 360.0) - 180.0
    sgn = np.sign(phi)  # symmetry along Omega
    phi = np.abs(phi)  # symmetry along Omega

    return rho, phi, sgn


def proj_qc(xx, yy, inc, Omega):  # deg  # deg
    # Rotate.
    rho = np.sqrt(xx**2 + yy**2)
    phi = np.rad2deg(np.arctan2(yy, xx))
    phi -= Omega
    xx_rot = rho * np.cos(np.deg2rad(phi))
    yy_rot = rho * np.sin(np.deg2rad(phi))

    # Incline.
    xx_inc = xx_rot
    yy_inc = yy_rot / np.cos(np.deg2rad(inc))

    # Convert.
    rho = np.sqrt(xx_inc**2 + yy_inc**2)
    phi = np.rad2deg(np.arctan2(yy_inc, xx_inc))
    phi -= 90.0
    phi = ((phi + 180.0) % 360.0) - 180.0
    sgn = np.sign(phi)  # symmetry along Omega
    phi = np.abs(phi)  # symmetry along Omega

    szx = phi.shape[1]
    if szx % 2 == 0:
        phi[:, szx // 2 :] = np.fliplr(phi[:, : szx // 2])
    else:
        phi[:, (szx + 1) // 2 :] = np.fliplr(phi[:, : (szx - 1) // 2])

    return rho, phi, sgn


def proj_sa(xx, yy, inc, Omega):  # deg  # deg
    # Rotate.
    rho = np.sqrt(xx**2 + yy**2)
    phi = np.rad2deg(np.arctan2(yy, xx))
    phi -= Omega
    xx_rot = rho * np.cos(np.deg2rad(phi))
    yy_rot = rho * np.sin(np.deg2rad(phi))

    # Incline.
    xx_inc = xx_rot
    yy_inc = yy_rot / np.cos(np.deg2rad(inc))

    # Convert.
    rho = np.sqrt(xx_inc**2 + yy_inc**2)
    phi = np.rad2deg(
        np.arccos(yy_rot * np.tan(np.deg2rad(inc)) / np.sqrt(xx_inc**2 + yy_inc**2))
    )
    sgn = np.sign(phi)  # symmetry along Omega

    return rho, phi, sgn
