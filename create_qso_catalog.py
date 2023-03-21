"""
    This script will create a quasar catalog from a CoLoRe output. It is needed before converting density to transmission.
"""

import fitsio
import numpy as np
from pathlib import Path
import healpy as hp
from typing import *
from scipy.interpolate import interp1d
from scipy.integrate import quad
from lyacolore.utils import make_MOCKID

src_number = 1  # CoLoRe source to use.
z_min = 1.8  # Minimum redshift of quasars
nside = 16  # Nside of pixelization
footprint_pixels = np.asarray(
    [
        int(x.name.split(".")[0].split("-")[2])
        for x in Path(
            "/global/cfs/cdirs/desi/mocks/lya_forest/london/v9.0/v9.0.0"
        ).glob("**/transmission*.fits.gz")
    ]
)  # Valid pixels (set to None for all)
target_qso_sqd = 120  # Target QSOs per sqdg. If smaller than sample, there will be downsampling. (None for no downsampling)
seed = 0  # Random seed for downsampling
dndz_file = Path(
    "/global/cfs/cdirs/desi/users/cramirez/LyA_mocks/lya_mock_2LPT/Inputs/CoLoRe/Nz_qso_130618_2_colore1_hZs.txt"
)  # Dndz to read quasar density in sample. (None for no downsampling)

CoLoRe_dir = Path(
    "/global/cfs/cdirs/desi/users/cramirez/LyA_mocks/lya_mock_2LPT_runs/CoLoRe/CoLoRe_lognormal/CoLoRe_seed0_4096"
)
output_catalog = Path("/pscratch/sd/c/cramirez/deltas_from_colore/zcat.fits")


def main():
    if dndz_file is None or target_qso_sqd is None:
        downsampling = 1
    else:
        # Compute QSO density
        z, dndz = np.loadtxt(dndz_file, unpack=True)
        dndz_interp = interp1d(z, dndz)
        dens = quad(dndz_interp, z_min, max(z))[0]
        print("QSO density in input catalog: ", dens)

        # Compute downsampling
        downsampling = min((1, target_qso_sqd / dens))
        print("Using downsampling: ", downsampling)

    cat_ra = []
    cat_dec = []
    cat_z = []
    cat_mockid = []
    cat_pixel = []
    cat_filenum = []

    for colore_file in CoLoRe_dir.glob(f"out_srcs_s{src_number}*.fits"):
        print("Reading file: ", colore_file.name, len(cat_ra), end="\r")
        with fitsio.FITS(colore_file) as hdul:
            if downsampling != 1:
                np.random.seed(seed)
                mask = np.random.choice(
                    [True, False],
                    hdul[1].read_header()["NAXIS2"],
                    p=[downsampling, 1 - downsampling],
                )
            else:
                mask = np.ones(hdul[1].read_header()["NAXIS2"], dtype=bool)

            z = hdul[1]["Z_COSMO"].read()
            mask &= z > z_min

            z = z[mask] + hdul[1]["DZ_RSD"].read()[mask]
            ra = hdul[1]["RA"].read()[mask]
            dec = hdul[1]["DEC"].read()[mask]

            filenum = int(hdul._filename.split(".")[0].split("_")[-1])
            mockids = make_MOCKID(
                filenum,
                list(range(hdul[1].read_header()["NAXIS2"])),
            )[mask]

            pixels = hp.ang2pix(
                nside,
                ra,
                dec,
                lonlat=True,
                nest=True,
            )

            if footprint_pixels is not None:
                pixel_mask = np.in1d(pixels, footprint_pixels)
            else:
                pixel_mask = np.ones_like(pixels).astype(bool)

            cat_ra.append(ra[pixel_mask])
            cat_dec.append(dec[pixel_mask])
            cat_z.append(z[pixel_mask])
            cat_mockid.append(mockids[pixel_mask])
            cat_pixel.append(pixels[pixel_mask])
            cat_filenum.append(filenum * np.ones(pixel_mask.sum()))

    cat_ra = np.concatenate(cat_ra)
    cat_dec = np.concatenate(cat_dec)
    cat_z = np.concatenate(cat_z)
    cat_mockid = np.concatenate(cat_mockid)
    cat_pixel = np.concatenate(cat_pixel)
    cat_filenum = np.concatenate(cat_filenum).astype(int)

    with fitsio.FITS(output_catalog, "rw", clobber=True) as hdul:
        hdul.write(
            [cat_ra, cat_dec, cat_z, cat_mockid, cat_pixel, cat_filenum],
            names=["RA", "DEC", "Z", "TARGETID", "PIXELNUM", "FILENUM"],
            extname="ZCATALOG",
        )


if __name__ == "__main__":
    main()
