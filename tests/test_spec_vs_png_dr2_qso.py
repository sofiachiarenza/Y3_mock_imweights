"""Spec test: fresh planck_pr4 QSO WEIGHT_IMLIN (this repo, full compressed
randoms) vs PNG_DR2's WEIGHT_IMLIN_QSO for the same mocks (hdu 1). Same map
list / z-bins / regions; PNG_DR2 used a 20% randoms subsample, so agreement is
expected within that noise, not exactly.

PNG_DR2 ROW_ID indexes the table after Z in [0.8, 2.1) and dropping rows
outside the DR9 N/S footprint (lognormal_catalog_io.read_lognormal_data).

    python tests/test_spec_vs_png_dr2_qso.py --sidecar-root outputs/pilot --sims 260 261
"""
import argparse
import os
import sys

import h5py
import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_config, mock_path, photsys_of_pix, radec_to_pix, sidecar_path  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sidecar-root", default=None)
    p.add_argument("--sims", type=int, nargs="+", default=[260, 261])
    args = p.parse_args()
    cfg = load_config()
    if args.sidecar_root:
        cfg["paths"]["output_root"] = args.sidecar_root
    png = cfg["spec"]["png_dr2_qso_weights"]
    for sim in args.sims:
        with fits.open(mock_path(cfg, "spec", "planck_pr4", "QSO", sim), memmap=True) as h:
            d = h[1].data
            ra, dec, z = np.asarray(d["RA"], float), np.asarray(d["DEC"], float), np.asarray(d["Z"], float)
        zsel = np.flatnonzero((z >= 0.8) & (z < 2.1))
        idx = zsel[photsys_of_pix(radec_to_pix(ra[zsel], dec[zsel])) != ""]
        with h5py.File(os.path.join(png, f"file{sim:04d}_hdu1.h5"), "r") as f:
            row, w_png = f["LSS/ROW_ID"][:], f["LSS/WEIGHT_IMLIN_QSO"][:]
        assert len(row) == len(idx), (len(row), len(idx))
        full_png = np.ones(len(ra))
        full_png[idx[row]] = w_png
        with h5py.File(sidecar_path(cfg, "spec", "planck_pr4", "QSO", sim, 1), "r") as f:
            w_new = f["WEIGHT_IMLIN"][:]
        sel = idx[row]
        dw = w_new[sel] - full_png[sel]
        print(f"sim {sim}: N={len(sel)}  std(w_png)={full_png[sel].std():.4f} std(w_new)={w_new[sel].std():.4f}  "
              f"corr={np.corrcoef(w_new[sel], full_png[sel])[0, 1]:.4f}  rms(dw)={dw.std():.2e}  max|dw|={np.abs(dw).max():.2e}")


if __name__ == "__main__":
    main()
