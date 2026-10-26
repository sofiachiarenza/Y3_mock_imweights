"""Linear imaging-systematics regression on the REAL photometric samples (CLAUDE.md task 1).

Same fit as regress_photo_mocks.py: the template file, map lists, SFD-corrected
*DEPTH* maps, north / south / des regions and every variant in
config.photo.variants, including the data_only CSFD+SFD+HI diagnostic.
Only the inputs change. The real catalog comes from the upstream DESI_Y3_x_CMB
PhotoSample (Gerrit's saved h5), and the randoms are its masked randoms-1-<i>.fits,
compressed to one row per pixel. The compression is exact here too, because the
templates are constant per pixel.
Output is a sidecar and the catalogs are never modified:
    outputs/photo_data/<T>/data.h5      (weight columns in catalog row order; extra/ TARGETID, RA, DEC,
                                         W_ANALYSIS = Z_WEIGHT, the non-imaging analysis weight)

Validation (CLAUDE.md): --mock LENSMAP SIM pushes a mock + its randoms cache through
the same path, writes outputs/validation/photo/..., and compares it with the existing
mock sidecar; it should agree to max|dw| = 0.

    python regress_photo_data.py --tracer LRG1
    python regress_photo_data.py --tracer LRG1 --mock act_dr6 1
"""
import argparse
import logging
import os
import sys
import time

import jax
import numpy as np
from astropy.io import fits
from astropy.table import Table

from common import (RandomsCompressor, compare_sidecars, load_config, load_randoms_cache, mock_path, sidecar_path,
                    write_sidecar)
from regress_photo_mocks import load_templates, regress_catalog

jax.config.update("jax_enable_x64", True)
log = logging.getLogger("regress_photo_data")


def photo_sample(cfg, tracer):
    sys.path.insert(0, cfg["data"]["upstream_repo"])
    from DESI_Y3_x_CMB.measurement_pipeline.sample_factory import get_photo_sample_obj
    return get_photo_sample_obj(tracer, "all")


def data_randoms_cache(cfg, sample, tracer, n_randoms, rebuild=False):
    """Per-pixel randoms counts of the sample's masked randoms, cached in outputs/randoms_cache/."""
    path = os.path.join(cfg["paths"]["cache_root"], f"photo_data_{tracer}_nran{n_randoms}.fits")
    if os.path.exists(path) and not rebuild:
        return Table.read(path)
    t0 = time.time()
    sample.load_randoms(columns=["RA", "DEC"], n_randoms=n_randoms, distribute_randoms_across_processes=False)
    rr = sample.randoms_get_cat_attrs(["RA", "DEC"])
    comp = RandomsCompressor()
    comp.add(rr[:, 0], rr[:, 1])
    t = comp.table(SAMPLE="photo_data", TRACER=tracer, NRANFILE=n_randoms, NROWS=len(rr))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t.write(path, overwrite=True)
    log.info("%s: randoms cache %s (%d rows, %d pixels, %.0fs)", tracer, path, len(rr), len(t), time.time() - t0)
    return t


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracer", required=True, choices=["BGS", "LRG1", "LRG2", "LRG3"])
    p.add_argument("--variants", nargs="+", default=None)
    p.add_argument("--n-randoms", type=int, default=None, help="default: config data.n_randoms")
    p.add_argument("--data-weight", choices=["none", "Z_WEIGHT"], default="none",
                   help="data weights in the fit (mocks use none)")
    p.add_argument("--mock", nargs=2, metavar=("LENSMAP", "SIM"), help="validation run on one mock")
    p.add_argument("--rebuild-cache", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config()
    variants = args.variants or list(cfg["photo"]["variants"])
    regions = cfg["photo"]["regions"]
    templates = load_templates(cfg["paths"]["photo_templates"])
    t0 = time.time()

    if args.mock:
        lensmap, sim = args.mock[0], int(args.mock[1])
        hdu = cfg["hdu"]
        cache = load_randoms_cache(cfg, "photo", args.tracer)
        with fits.open(mock_path(cfg, "photo", lensmap, args.tracer, sim), memmap=True) as h:
            ra, dec = np.asarray(h[hdu].data["RA"], float), np.asarray(h[hdu].data["DEC"], float)
        data_w, extra, is_data = None, None, False
        out = os.path.join(cfg["paths"]["output_root"], "validation", "photo", lensmap, args.tracer,
                           f"mock_sim{sim:04d}_hdu{hdu}.h5")
        meta = {"sample": "photo", "tracer": args.tracer, "lensmap": lensmap, "sim": sim, "hdu": hdu,
                "validation": "regress_photo_data.py --mock"}
    else:
        n_ran = args.n_randoms or cfg["data"]["n_randoms"]
        sample = photo_sample(cfg, args.tracer)
        sample.load_data()
        cat = sample.get_cat_attrs(["TARGETID", "RA", "DEC", "Z_WEIGHT"], conv2array=False)
        ra, dec = np.asarray(cat["RA"], float), np.asarray(cat["DEC"], float)
        zw = np.asarray(cat["Z_WEIGHT"], float)
        log.info("%s: %d objects, Z_WEIGHT min/mean/max %.3f/%.3f/%.3f", args.tracer, len(ra), zw.min(), zw.mean(), zw.max())
        data_w = zw if args.data_weight == "Z_WEIGHT" else None
        extra, is_data = {"TARGETID": np.asarray(cat["TARGETID"], np.int64), "RA": ra, "DEC": dec, "W_ANALYSIS": zw}, True
        cache = data_randoms_cache(cfg, sample, args.tracer, n_ran, args.rebuild_cache)
        suffix = "" if args.data_weight == "none" else "_zweight"
        out = os.path.join(cfg["paths"]["output_root"], "photo_data", args.tracer, f"data{suffix}.h5")
        meta = {"sample": "photo_data", "tracer": args.tracer, "n_randoms": n_ran, "data_weight": args.data_weight}

    rand_pix, rand_w = np.asarray(cache["PIX"]), np.asarray(cache["W"], float)
    log.info("%s: %d randoms pixels (%d randoms)", args.tracer, len(cache), int(cache["NRAN"].sum()))
    weights, coeffs = regress_catalog(cfg, args.tracer, ra, dec, templates, rand_pix, rand_w, variants, regions,
                                      data=is_data, data_w=data_w)
    write_sidecar(out, len(ra), weights, coeffs, {**meta, "templates": cfg["paths"]["photo_templates"]}, extra=extra)
    log.info("%s: N=%d, %.1fs -> %s", args.tracer, len(ra), time.time() - t0, out)

    if args.mock:
        ref = sidecar_path(cfg, "photo", lensmap, args.tracer, sim, cfg["hdu"])
        worst = compare_sidecars(out, ref, log=log.info)
        log.info("VALIDATION %s vs %s: max|dw| = %.3e -> %s", out, ref, worst, "PASS" if worst == 0 else "CHECK")


if __name__ == "__main__":
    main()
