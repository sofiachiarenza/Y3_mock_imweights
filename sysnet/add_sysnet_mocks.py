"""SysNet weights for one ELG mock -> sidecar (port of Y3_AMR/act_scripts/add_sysnet_ACT.py).

Same weight as Y3_AMR and LSS mkCat_main.py --add_sysnet:
  w_pix = 1 / mean(predictions), normalised to mean 1 over pixels, clipped to [0.5, 2],
  per region (N, S); objects get the weight of their pixel, 1 outside the region.
Pixel ordering is RING: prep4sysnet reorders the maps nest->ring and pixelises in RING
(LSS/imaging/sysnet_tools.py), so nn-weights.fits 'hpix' is RING, like the official
mkCat_main.py, which also uses hp.ang2pix(256, th, phi). Do NOT add nest=True here.
Changes from Y3_AMR: the mock is read-only, so the column WEIGHT_SN_MOCK goes to the sidecar
    outputs/spec/<lensmap>/ELG/mock_simNNNN_hdu1.h5
(the mocks' own WEIGHT_SN column is a copy of WEIGHT_SYS, not this).

    python sysnet/add_sysnet_mocks.py --lensmap act_dr6 --sim 1
"""
import argparse
import os
import sys

import fitsio
import healpy as hp
import numpy as np
from astropy.io import fits
from astropy.table import Table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import append_sidecar, existing_columns, load_config, mock_path, sidecar_path, write_sidecar  # noqa: E402
from prepare_sysnet_mocks import TRACER, WORKDIR, ZMAX, ZMIN, add_photsys, mock_dir  # noqa: E402

from LSS.imaging import densvar  # noqa: E402

COLUMN = "WEIGHT_SN_MOCK"


def sysnet_weights(data, mdir):
    dth, dphi = densvar.radec2thphi(data["RA"], data["DEC"])
    dpix = hp.ang2pix(256, dth, dphi)  # RING, see docstring
    w = np.ones(len(data))
    for reg in ("N", "S"):
        sn = fitsio.read(os.path.join(mdir, f"{ZMIN}_{ZMAX}_{reg}", "nn-weights.fits"))
        pix_weight = 1.0 / np.mean(sn["weight"], axis=1)
        pix_weight = np.clip(pix_weight / pix_weight.mean(), 0.5, 2.0)
        hpmap = np.ones(12 * 256 * 256)
        hpmap[sn["hpix"]] = pix_weight
        sel = data["PHOTSYS"] == reg
        w[sel] = hpmap[dpix[sel]]
    return w


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lensmap", required=True, choices=["act_dr6", "planck_pr4"])
    p.add_argument("--sim", type=int, required=True)
    p.add_argument("--workdir", default=WORKDIR)
    args = p.parse_args()
    cfg = load_config()
    hdu = cfg["hdu"]
    with fits.open(mock_path(cfg, "spec", args.lensmap, TRACER, args.sim), memmap=True) as h:
        data = add_photsys(Table({c: np.asarray(h[hdu].data[c], float) for c in ("RA", "DEC")}))
    w = sysnet_weights(data, mock_dir(args.workdir, args.lensmap, args.sim))
    out = sidecar_path(cfg, "spec", args.lensmap, TRACER, args.sim, hdu)
    if os.path.exists(out):
        if COLUMN in existing_columns(out):
            sys.exit(f"{out} already has {COLUMN}")
        append_sidecar(out, len(data), {COLUMN: w}, {})
    else:
        write_sidecar(out, len(data), {COLUMN: w}, {},
                      {"sample": "spec", "tracer": TRACER, "lensmap": args.lensmap, "sim": args.sim, "hdu": hdu,
                       "method": "SysNet snapshot ensemble (Y3_AMR setup), per-mock training"})
    fitted = w != 1
    print(f"{TRACER} {args.lensmap} sim {args.sim}: N={len(w)}, fitted={fitted.sum()}, "
          f"{COLUMN} mean={w[fitted].mean():.4f} min={w.min():.3f} max={w.max():.3f} -> {out}")


if __name__ == "__main__":
    main()
