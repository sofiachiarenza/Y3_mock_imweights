"""Photo test: new driver + compressed randoms vs the ORIGINAL
Y3_AMR_nersc/regress_photo_mocks.py logic on the same inputs.

The old module is imported unchanged. Its NERSC-only imports are satisfied by
stubs: LSS.common_tools -> {ext_coeff} (same values), alaeboss.linear_regression
/ alaeboss.produce_imweights -> the vendored, identical copies.

Same inputs for both: one photo mock (HDU 1), a uniform subset of the raw
randoms (~N_SUB rows, spread across the file), old pixweight-with-CSFD
templates with the old GALDEPTH-only extinction correction, old LRG/BGS map
list, South region (south | des). Expectation: weights agree to ~1e-8.
"""
import argparse
import logging
import os
import sys
import types

import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from common import imaging_footprints, load_config, mock_path, radec_to_pix  # noqa: E402

import linear_regression as _lr  # noqa: E402  (vendored, path set by common)
import produce_imweights as _pi  # noqa: E402

# ---- stubs so the old module imports unchanged ----
lss = types.ModuleType("LSS")
lss_ct = types.ModuleType("LSS.common_tools")
lss_ct.ext_coeff = dict(_pi.ext_coeff)
lss.common_tools = lss_ct
ala = types.ModuleType("alaeboss")
ala.linear_regression, ala.produce_imweights = _lr, _pi
sys.modules.update({"LSS": lss, "LSS.common_tools": lss_ct, "alaeboss": ala,
                    "alaeboss.linear_regression": _lr, "alaeboss.produce_imweights": _pi})
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "Y3_AMR_nersc"))
import regress_photo_mocks as old  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("new_regress_photo", os.path.join(ROOT, "regress_photo_mocks.py"))
_new = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_new)
fit_one = _new.fit_one  # new driver (same module name as the old one, so load by path)
assert old.__file__ != _new.__file__

OLD_PIXWEIGHT = os.path.join(os.path.dirname(ROOT), "Y3_AMR_nersc", "photo_outputs", "pixweight-1-dark-with-csfd.fits")


def randoms_subset(path, n_sub, nblocks=200):
    with h5py.File(path, "r") as f:
        ds = f["__astropy_table__"]
        n = ds.shape[0]
        blk = n_sub // nblocks
        r = np.concatenate([ds[o:o + blk] for o in np.linspace(0, n - blk, nblocks).astype(int)])
    return Table({"RA": np.asarray(r["RA"], float), "DEC": np.asarray(r["DEC"], float)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracer", default="LRG1")
    p.add_argument("--lensmap", default="act_dr6")
    p.add_argument("--sim", type=int, default=1)
    p.add_argument("--n-sub", type=int, default=40_000_000)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("old")
    cfg = load_config()

    with fits.open(mock_path(cfg, "photo", args.lensmap, args.tracer, args.sim), memmap=True) as h:
        data = Table({"RA": np.asarray(h[1].data["RA"], float), "DEC": np.asarray(h[1].data["DEC"], float)})
    rand = randoms_subset(os.path.join(cfg["paths"]["photo_mock_root"], f"mock_randoms_{args.tracer}.h5"), args.n_sub)

    # ---- old logic (South only; old code keeps PHOTSYS=='S' = south | des) ----
    data = old.ensure_cols_for_imlin(data)
    rand = old.ensure_cols_for_imlin(rand)
    data_s, rand_s = data[data["PHOTSYS"] == "S"], rand[rand["PHOTSYS"] == "S"]
    maps_bundle = old.load_pixweight_maps(OLD_PIXWEIGHT, logger)
    bundle = old.build_randoms_bundle(rand_s, args.tracer, maps_bundle, logger, "full")
    w_old = np.asarray(old.run_sample_regression(data_s, bundle, maps_bundle, "full", logger))
    maps = bundle["fit_maps"]

    # ---- new driver, compressed randoms from the SAME subset ----
    templates = maps_bundle["maps"]  # identical template arrays (old correction convention)
    rpix = radec_to_pix(rand_s["RA"], rand_s["DEC"])
    upix, counts = np.unique(rpix, return_counts=True)
    w_new, best = fit_one(radec_to_pix(data_s["RA"], data_s["DEC"]), upix, counts.astype(float),
                          templates, maps, cfg["nbins"], cfg["tail"])

    d = w_new - w_old
    foot = imaging_footprints()
    print(f"tracer={args.tracer} sim={args.sim} maps={maps}")
    print(f"N data (S)={len(data_s)}, N randoms subset (S)={len(rand_s)} in {len(upix)} pixels")
    print(f"old weights: mean={w_old.mean():.6f} std={w_old.std():.6f}")
    print(f"max|w_new - w_old| = {np.abs(d).max():.3e}   rms = {d.std():.3e}")
    print("new coeffs:", {k: round(v, 6) for k, v in best.items()})
    ok = np.abs(d).max() < 1e-6
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
