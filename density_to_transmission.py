import fitsio
import numpy as np
from pathlib import Path
import healpy as hp
from typing import *
from itertools import repeat
from lyacolore.utils import make_MOCKID
from multiprocessing import Pool

src_number = 1  # CoLoRe source to use
z_min = 1.8  # This is needed to match forests, can be find in config.ini from LyaCoLoRe
nside = 16
lya_rest = 1215.67  # Used to make RF cuts
lambda_min = 3600
lambda_max = 5500
lambda_rest_min = 1040
lambda_rest_max = 1200


CoLoRe_dir = Path(
    "/global/cfs/cdirs/desi/users/cramirez/LyA_mocks/lya_mock_2LPT_runs/CoLoRe/CoLoRe_lognormal/CoLoRe_seed0_4096"
)

quasar_catalog = Path("/pscratch/sd/c/cramirez/deltas_from_colore/zcat.fits")

output_dir = Path("/pscratch/sd/c/cramirez/deltas_from_colore/deltas")


def main():
    with fitsio.FITS(quasar_catalog) as hdul:
        valid_mockids = hdul[1]["TARGETID"].read()
        valid_filenum = hdul[1]["FILENUM"].read()

    for colore_file in sorted(CoLoRe_dir.glob(f"out_srcs_s{src_number}*.fits")):
        print("colore file :", colore_file)
        with fitsio.FITS(colore_file) as hdul:
            filenum = int(hdul._filename.split(".")[0].split("_")[-1])
            valid_mask = valid_filenum == filenum

            mockids = make_MOCKID(
                filenum,
                list(range(hdul[1].read_header()["NAXIS2"])),
            )

            mask = np.in1d(mockids, valid_mockids[valid_mask])

            mockids = mockids[mask]
            z = hdul[1]["Z_COSMO"].read()[mask] + hdul[1]["DZ_RSD"].read()[mask]
            ra = hdul[1]["RA"].read()[mask]
            dec = hdul[1]["DEC"].read()[mask]

            pixels = hp.ang2pix(
                nside,
                ra,
                dec,
                lonlat=True,
                nest=True,
            )

            unique_pixels = np.unique(pixels)

            if len(unique_pixels) > 0:
                lambda_grid = lya_rest * (1 + hdul[4]["Z"].read())
                wave_mask = lambda_grid < lambda_max
                wave_mask &= lambda_grid > lambda_min
                lambda_grid = lambda_grid[wave_mask]

                deltas = hdul[2].read()[mask]

                if wave_mask.sum() < len(wave_mask):
                    deltas = deltas.T[wave_mask].T  # This is just masking in wavelength

                # for pixel_to_use in unique_pixels:
                #     create_deltas_file(ra, dec, z, mockids, pixels, pixel_to_use, deltas, lambda_grid, output_dir)

                pool = Pool(processes=10)
                pool.starmap(
                    create_deltas_file,
                    zip(
                        repeat(ra),
                        repeat(dec),
                        repeat(z),
                        repeat(mockids),
                        repeat(pixels),
                        unique_pixels,
                        repeat(deltas),
                        repeat(lambda_grid),
                        repeat(output_dir),
                    ),
                )
                pool.close()


def create_deltas_file(
    ra: List[float],
    dec: List[float],
    z: List[float],
    mockids: List[int],
    pixels: List[int],
    pixel_to_use: int,
    deltas: List[float],
    lambda_grid: List[float],
    output_dir: Union[str, Path],
):
    print("Transforming pixel: ", pixel_to_use)
    output_dir = Path(output_dir)

    pixel_mask = pixels == pixel_to_use

    # Compute masking for RF wavelength
    weights = np.asarray(
        [lambda_rest_min < lambda_grid / (1 + zi) for zi in z[pixel_mask]]
    )
    weights &= np.asarray(
        [lambda_grid / (1 + zi) < lambda_rest_max for zi in z[pixel_mask]]
    )
    weights = weights.astype(float)
    weights_mask = (
        weights.sum(axis=1) > 10
    )  # only allow for forests with at least 10 pixels

    weights = weights[weights_mask]
    deltas = deltas[pixel_mask][weights_mask]
    cont = np.ones_like(deltas)

    z = z[pixel_mask][weights_mask]
    ra = ra[pixel_mask][weights_mask]
    dec = dec[pixel_mask][weights_mask]
    mockids = mockids[pixel_mask][weights_mask]
    meansnr = np.zeros_like(z)

    with fitsio.FITS(
        output_dir / f"delta-{pixel_to_use}.fits.gz", "rw", clobber=True
    ) as results:
        results.write(None)  # pseudo-primary

        ## Wavelength card
        hdr = fitsio.FITSHDR()
        hdr.add_record(
            {
                "name": "WAVE_SOLUTION",
                "value": "log",
            }
        )
        hdr.add_record(
            {
                "name": "DELTA_LOG_LAMBDA",
                "value": round(lambda_grid[1] - lambda_grid[0], 2),
            }
        )
        results.write(
            lambda_grid,
            extname="LAMBDA",
            header=hdr,
        )

        # Metadata card
        hdr = fitsio.FITSHDR()
        hdr.add_record(
            {
                "name": "BLINDING",
                "value": "none",
            }
        )
        results.write(
            [
                mockids,
                ra * np.pi / 180.0,
                dec * np.pi / 180,
                z,
                meansnr,
                mockids,
                mockids,
                mockids,
                mockids,
            ],
            names=[
                "LOS_ID",
                "RA",
                "DEC",
                "Z",
                "MEANSNR",
                "TARGETID",
                "NIGHT",
                "PETAL",
                "TILE",
            ],
            extname="METADATA",
        )

        # Deltas card
        results.write(deltas, extname="DELTA")

        # wEIGHTS
        results.write(weights, extname="WEIGHT")

        # CONT
        results.write(cont, extname="CONT")


if __name__ == "__main__":
    main()
