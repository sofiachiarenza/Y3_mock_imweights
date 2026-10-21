"""Spec test: produce_imweights on raw randoms vs on the compressed
(per-pixel, per-z-bin) randoms built from the SAME raw subset.
Expectation: agreement at the minimizer-tolerance level, not bit-exact. Random
weights are floats here (photo: integer counts -> exactly 0), so summation order
changes the binned sums in the last bits and MIGRAD (EDM goal 2e-4) stops at a
slightly different point: coefficients ~1e-6, weights a few 1e-6 (vs a weight
scatter of ~2e-2). Threshold 1e-5. Measured 2026-10-21 on LRG1 act sim 1: 3.9e-6.

    python tests/test_spec_compression.py --tracer LRG1 --lensmap act_dr6 --sim 1
"""
import argparse
import os
import sys

import h5py
import numpy as np
from astropy.table import Table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import NPIX, load_config, mock_path, photsys_of_pix, radec_to_pix, randoms_h5_path, spec_randoms_table  # noqa: E402
from regress_spec_mocks import build_templates, read_mock, regress_catalog  # noqa: E402


def raw_subset(path, n_sub, nblocks=200):
    with h5py.File(path, "r") as f:
        ds = f["__astropy_table__"]
        n = ds.shape[0]
        blk = n_sub // nblocks
        r = np.concatenate([ds[o:o + blk] for o in np.linspace(0, n - blk, nblocks).astype(int)])
    t = Table({c: np.asarray(r[c], float) for c in ["RA", "DEC", "Z", "WEIGHT", "WEIGHT_FKP", "WEIGHT_SYS", "WEIGHT_ZFAIL"]})
    t["PHOTSYS"] = photsys_of_pix(radec_to_pix(t["RA"], t["DEC"]))
    return t[t["PHOTSYS"] != ""]


def compress(raw, zbins):
    pix = radec_to_pix(raw["RA"], raw["DEC"])
    w = raw["WEIGHT"] * raw["WEIGHT_FKP"] / raw["WEIGHT_SYS"] / raw["WEIGHT_ZFAIL"]
    wsum, nsum = np.zeros(len(zbins) * NPIX), np.zeros(len(zbins) * NPIX, dtype=np.int64)
    for k, (lo, hi) in enumerate(zbins):
        sel = (raw["Z"] > lo) & (raw["Z"] < hi)
        wsum += np.bincount(pix[sel] + k * NPIX, weights=w[sel], minlength=wsum.size)
        nsum += np.bincount(pix[sel] + k * NPIX, minlength=nsum.size)
    keep = np.flatnonzero(nsum > 0)
    return Table({"PIX": keep % NPIX, "ZBIN": keep // NPIX, "W": wsum[keep], "NRAN": nsum[keep]})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracer", default="LRG1")
    p.add_argument("--lensmap", default="act_dr6")
    p.add_argument("--sim", type=int, default=1)
    p.add_argument("--n-sub", type=int, default=20_000_000)
    args = p.parse_args()
    cfg = load_config()
    zbins = cfg["spec"]["tracers"][args.tracer]["zbins"]

    data = read_mock(mock_path(cfg, "spec", args.lensmap, args.tracer, args.sim), cfg["hdu"])
    templates = build_templates(cfg, args.tracer)
    raw = raw_subset(randoms_h5_path(cfg, "spec", args.tracer), args.n_sub)
    comp = spec_randoms_table(compress(raw, zbins), zbins)

    w_raw, c_raw = regress_catalog(cfg, args.tracer, data, templates, raw, ["WEIGHT_IMLIN"])
    w_cmp, c_cmp = regress_catalog(cfg, args.tracer, data, templates, comp, ["WEIGHT_IMLIN"])
    d = w_cmp["WEIGHT_IMLIN"] - w_raw["WEIGHT_IMLIN"]
    fitted = w_raw["WEIGHT_IMLIN"] != 1
    print(f"{args.tracer} {args.lensmap} sim {args.sim}: N data={len(data)}, fitted={fitted.sum()}, "
          f"raw randoms={len(raw)}, compressed rows={len(comp)}")
    print(f"raw weights: mean={w_raw['WEIGHT_IMLIN'][fitted].mean():.6f} std={w_raw['WEIGHT_IMLIN'][fitted].std():.6f}")
    print(f"max|w_cmp - w_raw| = {np.abs(d).max():.3e}")
    for key in c_raw["WEIGHT_IMLIN"]:
        a, b = c_raw["WEIGHT_IMLIN"][key], c_cmp["WEIGHT_IMLIN"][key]
        print(f"  {key}: max coeff diff {max(abs(a[k] - b[k]) for k in a):.2e}")
    ok = np.abs(d).max() < 1e-5
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
