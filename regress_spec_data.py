"""Linear imaging-systematics regression on the REAL spectroscopic samples (CLAUDE.md task 2).

Same fit as regress_spec_mocks.py: the DESI default map lists, z-bins and regions, the
vendored produce_imweights and every variant in config.spec.variants (WEIGHT_IMLIN,
_CSFD, _CSFDHI, _NOHI).
Only the inputs change. The real clustering catalogs come from the upstream
DESI_Y3_x_CMB NonSavableSpecSample (Gerrit's scratch copies of DA2 loa-v1 v2, both caps),
and the randoms are the first n_randoms files per cap, compressed per (pixel, z-bin)
with the produce_imweights weight WEIGHT*WEIGHT_FKP/WEIGHT_SYS/WEIGHT_ZFAIL.
Output is a sidecar and the catalogs are never modified:
    outputs/spec_data/<T>/data.h5
The sidecar holds the weight columns in catalog row order, plus WEIGHT_IMLIN_DESI, the
catalog's official weight copied for reference (WEIGHT_IMLIN, or WEIGHT_IMLIN_CLUS for BGS). extra/ holds TARGETID, RA, DEC, Z and
W_ANALYSIS = WEIGHT*WEIGHT_FKP/WEIGHT_SYS, the analysis weight without the imaging part.

The log compares the refit default WEIGHT_IMLIN with the official one. They should agree
closely but not exactly: 4 vs 18 randoms, and PHOTSYS from the footprint.

Validation: --mock LENSMAP SIM pushes a mock + its randoms cache through the same path and
compares the result with the existing sidecar. They should agree to MIGRAD tolerance (< 1e-5),
since the Narval and NERSC jax/iminuit builds differ.

    python regress_spec_data.py --tracer LRG1
    python regress_spec_data.py --tracer LRG1 --mock act_dr6 1
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
from astropy.table import Table

from common import (RandomsCompressor, compare_sidecars, load_config, load_randoms_cache, mock_path, sidecar_path,
                    spec_random_weight, spec_randoms_table, write_sidecar)
from regress_spec_mocks import build_templates, read_mock, regress_catalog

log = logging.getLogger("regress_spec_data")
DATA_COLS = ["TARGETID", "RA", "DEC", "Z", "WEIGHT", "WEIGHT_FKP", "WEIGHT_SYS", "WEIGHT_ZFAIL"]
OFFICIAL_IMLIN = ("WEIGHT_IMLIN", "WEIGHT_IMLIN_CLUS")  # BGS_BRIGHT-21.35 clustering catalogs carry the _CLUS name
RAND_COLS = ["RA", "DEC", "Z", "WEIGHT", "WEIGHT_FKP", "WEIGHT_SYS", "WEIGHT_ZFAIL"]


def spec_sample(cfg, tracer):
    sys.path.insert(0, cfg["data"]["upstream_repo"])
    from DESI_Y3_x_CMB.measurement_pipeline.samples.spec_sample import NonSavableSpecSample
    return NonSavableSpecSample(tracer, galactic_caps=["NGC", "SGC"])


def data_randoms_cache(cfg, sample, tracer, n_randoms, rebuild=False):
    """Per-(pixel, z-bin) summed randoms weight, read one randoms file (both caps) at a time."""
    path = os.path.join(cfg["paths"]["cache_root"], f"spec_data_{tracer}_nran{n_randoms}.fits")
    if os.path.exists(path) and not rebuild:
        return Table.read(path)
    t0 = time.time()
    comp = RandomsCompressor(cfg["spec"]["tracers"][tracer]["zbins"])
    nrows = 0
    for i in range(n_randoms):
        # same reader (and z window) as sample.load_randoms, one file at a time to bound memory
        r, _ = sample._load_catalog(i_randoms=i, columns=RAND_COLS)
        comp.add(np.asarray(r["RA"], float), np.asarray(r["DEC"], float), np.asarray(spec_random_weight(r), float),
                 np.asarray(r["Z"], float))
        nrows += len(r)
        log.info("%s: randoms file %d: %d rows", tracer, i, len(r))
    t = comp.table(SAMPLE="spec_data", TRACER=tracer, NRANFILE=n_randoms, NROWS=nrows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t.write(path, overwrite=True)
    log.info("%s: randoms cache %s (%d rows -> %d, %.0fs)", tracer, path, nrows, len(t), time.time() - t0)
    return t


def compare_official(w_refit, w_desi, log):
    fit = w_refit != 1
    a, b = w_refit[fit], w_desi[fit]
    log("refit WEIGHT_IMLIN vs official over %d objects: corr = %.5f, rms diff = %.2e, rms(official-1) = %.2e"
        % (fit.sum(), np.corrcoef(a, b)[0, 1], np.std(a - b), np.std(b - 1)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracer", required=True, choices=["BGS", "LRG1", "LRG2", "LRG3", "QSO", "QSO1", "QSO2"])
    p.add_argument("--variants", nargs="+", default=None)
    p.add_argument("--n-randoms", type=int, default=None, help="per cap; default: config data.n_randoms")
    p.add_argument("--mock", nargs=2, metavar=("LENSMAP", "SIM"), help="validation run on one mock")
    p.add_argument("--rebuild-cache", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config()
    variants = args.variants or list(cfg["spec"]["variants"])
    zbins = cfg["spec"]["tracers"][args.tracer]["zbins"]
    templates = build_templates(cfg, args.tracer)
    t0 = time.time()

    if args.mock:
        lensmap, sim = args.mock[0], int(args.mock[1])
        hdu = cfg["hdu"]
        data = read_mock(mock_path(cfg, "spec", lensmap, args.tracer, sim), hdu)
        cache = load_randoms_cache(cfg, "spec", args.tracer)
        extra = None
        out = os.path.join(cfg["paths"]["output_root"], "validation", "spec", lensmap, args.tracer,
                           f"mock_sim{sim:04d}_hdu{hdu}.h5")
        meta = {"sample": "spec", "tracer": args.tracer, "lensmap": lensmap, "sim": sim, "hdu": hdu,
                "validation": "regress_spec_data.py --mock"}
    else:
        n_ran = args.n_randoms or cfg["data"]["n_randoms"]
        sample = spec_sample(cfg, args.tracer)
        import fitsio
        present = fitsio.FITS(sample.get_catalog_filename("NGC"))[1].get_colnames()
        official = next(c for c in OFFICIAL_IMLIN if c in present)
        sample.load_data(columns=DATA_COLS + [official])
        cat = sample.get_cat_attrs(DATA_COLS + [official], conv2array=False)
        data = Table({c: np.asarray(cat[c], float) for c in DATA_COLS if c != "TARGETID"})
        data["WEIGHT_IMLIN_OFFICIAL"] = np.asarray(cat[official], float)
        log.info("%s: official imaging weight column %s", args.tracer, official)
        log.info("%s: %d objects in z %s", args.tracer, len(data), sample.redshift_range)
        extra = {"TARGETID": np.asarray(cat["TARGETID"], np.int64), "RA": data["RA"], "DEC": data["DEC"],
                 "Z": data["Z"], "W_ANALYSIS": data["WEIGHT"] * data["WEIGHT_FKP"] / data["WEIGHT_SYS"]}
        cache = data_randoms_cache(cfg, sample, args.tracer, n_ran, args.rebuild_cache)
        out = os.path.join(cfg["paths"]["output_root"], "spec_data", args.tracer, "data.h5")
        meta = {"sample": "spec_data", "tracer": args.tracer, "n_randoms": n_ran}

    randoms = spec_randoms_table(cache, zbins)
    log.info("%s: %d compressed randoms rows", args.tracer, len(randoms))
    weights, coeffs = regress_catalog(cfg, args.tracer, data, templates, randoms, variants)
    if not args.mock:
        weights["WEIGHT_IMLIN_DESI"] = np.asarray(data["WEIGHT_IMLIN_OFFICIAL"], float)
        if "WEIGHT_IMLIN" in weights:
            compare_official(weights["WEIGHT_IMLIN"], weights["WEIGHT_IMLIN_DESI"], log.info)
    write_sidecar(out, len(data), weights, coeffs, meta, extra=extra)
    log.info("%s: N=%d, %.1fs -> %s", args.tracer, len(data), time.time() - t0, out)

    if args.mock:
        ref = sidecar_path(cfg, "spec", lensmap, args.tracer, sim, cfg["hdu"])
        worst = compare_sidecars(out, ref, log=log.info)
        log.info("VALIDATION %s vs %s: max|dw| = %.3e -> %s", out, ref, worst, "PASS" if worst < 1e-5 else "CHECK")


if __name__ == "__main__":
    main()
