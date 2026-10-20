"""Build the photometric-sample template table (nside 256, NEST, equatorial).

Columns
-------
Imaging maps : copied from pixweight-1-dark.fits (DR9). All *DEPTH* columns are
               extinction-corrected with the pixweight SFD EBV,
               depth *= 10**(-0.4 * R_band * EBV) -- the spec convention
               (produce_imweights), now also applied to PSFDEPTH_W1.
SFD, CSFD    : E(B-V) [mag] from the CSFD release (Chiang 2023, Zenodo
               10.5281/zenodo.8207175: sfd_ebv.fits and csfd_ebv.fits, both
               HEALPix nside 2048 RING Galactic, i.e. identical pixelisation).
LSS          : the CIB/LSS 100um intensity removed from SFD to make CSFD [MJy/sr].
CSFD_COSMO   : fraction of the pixel inside the CSFD 'cosmology' mask (bit 2).
HI           : HI4PI N_HI [cm^-2] (CDS J/A+A/594/A116, NHI_HPX.fits.gz, nside
               1024 RING Galactic); linear N_HI like the DESI hpmaps `HI`.
EBV_PIXWEIGHT: raw pixweight EBV (cross-checks only).

Galactic-map columns: mean over the 64 nside-2048 NEST sub-pixel centres of
each nside-256 pixel (nearest-pixel lookup in the source map).

Inputs (data/ext/): csfd_zenodo_8207175/{sfd_ebv,csfd_ebv,mask,lss_intensity}.fits,
NHI_HPX.fits.gz. Run on a compute node.
"""
import os

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from common import NPIX, NSIDE, load_config

NSUB = 2048
NCHILD = (NSUB // NSIDE) ** 2  # 64
EXT_COEFF = {"G": 3.214, "R": 2.165, "Z": 1.211, "W1": 0.184, "W2": 0.113}  # LSS.common_tools.ext_coeff
IMAGING_COLS = ["FRACAREA", "STARDENS", "GALDEPTH_G", "GALDEPTH_R", "GALDEPTH_Z", "PSFDEPTH_G", "PSFDEPTH_R",
                "PSFDEPTH_Z", "PSFDEPTH_W1", "PSFDEPTH_W2", "PSFSIZE_G", "PSFSIZE_R", "PSFSIZE_Z"]


def read_hpx(path, col=None):
    with fits.open(path) as h:
        hdr = h[1].header
        d = h[1].data
        arr = np.asarray(d[col or d.columns.names[0]]).ravel()
    nest = str(hdr.get("ORDERING", "RING")).upper().startswith("NEST")
    return arr, nest


def main():
    cfg = load_config()
    ext = cfg["paths"]["ext_dir"]
    zdir = os.path.join(ext, "csfd_zenodo_8207175")

    src = {}
    for name, fn in (("SFD", "sfd_ebv.fits"), ("CSFD", "csfd_ebv.fits"), ("LSS", "lss_intensity.fits"), ("MASK", "mask.fits")):
        src[name] = read_hpx(os.path.join(zdir, fn))
    src["HI"] = read_hpx(os.path.join(ext, "NHI_HPX.fits.gz"), "NHI")

    acc = {k: np.zeros(NPIX) for k in ("SFD", "CSFD", "LSS", "CSFD_COSMO", "HI")}
    step = NPIX // 16
    for p0 in range(0, NPIX, step):
        parents = np.arange(p0, min(p0 + step, NPIX))
        child = (parents[:, None] * NCHILD + np.arange(NCHILD)[None, :]).ravel()
        theta, phi = hp.pix2ang(NSUB, child, nest=True)
        gal = SkyCoord(ra=np.degrees(phi) * u.deg, dec=(90 - np.degrees(theta)) * u.deg, frame="icrs").galactic
        l, b = gal.l.deg, gal.b.deg
        for name in ("SFD", "CSFD", "LSS", "MASK", "HI"):
            arr, nest = src[name]
            v = arr[hp.ang2pix(hp.npix2nside(len(arr)), l, b, nest=nest, lonlat=True)].astype(float)
            if name == "MASK":
                acc["CSFD_COSMO"][parents] = ((v.astype(np.int64) >> 2) & 1).reshape(-1, NCHILD).mean(1)
            else:
                if name == "LSS":  # ~1e30 sentinels outside the LSS-correction footprint (CSFD = SFD there)
                    v = np.where(np.abs(v) < 10, v, 0.0)
                acc[name][parents] = v.reshape(-1, NCHILD).mean(1)
        print(f"pixels {parents[-1] + 1}/{NPIX}", flush=True)

    out = Table(acc)
    pw = Table.read(cfg["paths"]["pixweight"])
    assert len(pw) == NPIX
    ebv = np.asarray(pw["EBV"], float)
    for c in IMAGING_COLS:
        v = np.asarray(pw[c], float)
        if "DEPTH" in c:
            v = v * 10 ** (-0.4 * EXT_COEFF[c.split("_")[-1]] * ebv)
        out[c] = v
    out["EBV_PIXWEIGHT"] = ebv
    out.meta.update({"HPXNSIDE": NSIDE, "ORDERING": "NESTED", "COORDSYS": "C",
                     "DEPTHCOR": "all *DEPTH* x 10**(-0.4 R EBV_PIXWEIGHT)",
                     "DUSTSRC": "CSFD release zenodo 8207175 (sfd_ebv, csfd_ebv, lss_intensity, mask bit2); HI4PI NHI",
                     "AVERAGE": "mean of 64 nside-2048 NEST subpixels"})
    path = cfg["paths"]["photo_templates"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.write(path, overwrite=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
