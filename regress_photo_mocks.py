"""Linear imaging-systematics regression on the photometric lognormal mocks.

One process = one (tracer, lensmap), looping over sims. Templates and the
compressed randoms are loaded once; every variant in config.photo.variants is
fit for every region (north / south / des) of every mock, and all variants are
written to one sidecar per mock:
    outputs/photo/<lensmap>/<T>/mock_simNNNN_hdu1.h5
Weights are full-HDU-length, 1.0 for objects outside the three regions.

Fit per (region, variant): alaeboss LinearRegressor, unit data weights,
random weights = randoms per pixel, cut_outliers(tail) -> prepare(nbins) ->
regress_minuit -> export_weights (same sequence as
Y3_AMR_nersc/regress_photo_mocks.py:run_sample_regression).
"""
import argparse
import logging
import os
import time

import jax
import numpy as np
from astropy.io import fits
from astropy.table import Table

from common import (imaging_footprints, load_config, load_randoms_cache, mock_path, radec_to_pix,
                    sidecar_path, sim_ids, write_sidecar)
from linear_regression import LinearRegressor

jax.config.update("jax_enable_x64", True)
log = logging.getLogger("regress_photo")


def variant_maps(cfg, tracer, variant, data=False):
    """Template list for (tracer, variant); None if the variant doesn't apply
    (wrong tracer, or a data_only variant on a mock)."""
    spec = cfg["photo"]["variants"][variant]
    if "tracers" in spec and tracer not in spec["tracers"]:
        return None
    if spec.get("data_only") and not data:
        return None
    if "maps" in spec:
        return list(spec["maps"])
    family = "BGS" if tracer.startswith("BGS") else "LRG"
    return list(cfg["photo"]["imaging_maps"][family]) + list(spec["dust"])


def load_templates(path):
    t = Table.read(path)
    return {c: np.asarray(t[c], float) for c in t.colnames}


def fit_one(data_pix, rand_pix, rand_w, templates, maps, nbins, tail, data_w=None):
    """-> (weights for data_pix, coefficient dict incl. chi2) or (None, None)."""
    reg = LinearRegressor.from_stacked_templates(
        data_weights=np.ones(len(data_pix)) if data_w is None else data_w,
        random_weights=np.asarray(rand_w, float),
        template_values_data=np.vstack([templates[m][data_pix] for m in maps]),
        template_values_randoms=np.vstack([templates[m][rand_pix] for m in maps]),
        template_names=maps,
        loglevel="WARNING",
    )
    reg.cut_outliers(tail=tail)
    reg.prepare(nbins=nbins)
    best = reg.regress_minuit()
    if best is None:
        return None, None
    best["chi2_null"] = float(reg.initial_chi2)
    best["chi2_fit"] = float(reg.chi2(reg.coefficients))
    best["n_dof"] = float(nbins * len(maps) - len(maps) - 1)
    # export_weights() covers only rows with finite templates; others keep 1.
    w = np.ones(len(data_pix))
    w[np.asarray(reg.good_values_data)] = np.asarray(reg.export_weights())
    best["n_bad_template"] = float(len(data_pix) - np.asarray(reg.good_values_data).sum())
    return w, best


def region_of_pix(pix, regions):
    foot = imaging_footprints()
    lab = np.full(len(pix), "", dtype="<U8")
    for r in regions:
        lab[foot[r][pix]] = r
    return lab


def regress_catalog(cfg, tracer, ra, dec, templates, rand_pix, rand_w, variants, regions, data=False, data_w=None):
    """Fit every (variant, region) for one catalog. Returns ({col: weights}, {variant: {region: coeffs}}).
    data=True enables data_only variants; data_w are optional per-object data weights (default 1)."""
    pix = radec_to_pix(ra, dec)
    dw = np.ones(len(pix)) if data_w is None else np.asarray(data_w, float)
    lab = region_of_pix(pix, regions)
    rlab = region_of_pix(rand_pix, regions)
    weights, coeffs = {}, {}
    for v in variants:
        maps = variant_maps(cfg, tracer, v, data=data)
        if maps is None:
            continue
        w = np.ones(len(pix))
        coeffs[v] = {}
        for r in regions:
            sd, sr = lab == r, rlab == r
            if sd.sum() == 0 or sr.sum() == 0:
                continue
            wr, best = fit_one(pix[sd], rand_pix[sr], rand_w[sr], templates, maps, cfg["nbins"], cfg["tail"],
                               data_w=dw[sd])
            if wr is None:
                log.error("%s %s %s: regression failed", tracer, v, r)
                w[sd] = np.nan
                continue
            w[sd] = wr
            coeffs[v][r] = best
        weights[v] = w
    return weights, coeffs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracer", required=True, choices=["BGS", "LRG1", "LRG2", "LRG3"])
    p.add_argument("--lensmap", required=True, choices=["act_dr6", "planck_pr4"])
    p.add_argument("--sim-start", type=int)
    p.add_argument("--sim-stop", type=int, help="inclusive")
    p.add_argument("--hdu", type=int, default=None)
    p.add_argument("--variants", nargs="+", default=None)
    p.add_argument("--output-root", default=None, help="override (tests)")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config()
    if args.output_root:
        cfg["paths"]["output_root"] = args.output_root
    hdu = args.hdu or cfg["hdu"]
    variants = args.variants or list(cfg["photo"]["variants"])
    regions = cfg["photo"]["regions"]

    templates = load_templates(cfg["paths"]["photo_templates"])
    cache = load_randoms_cache(cfg, "photo", args.tracer)
    rand_pix, rand_w = np.asarray(cache["PIX"]), np.asarray(cache["W"], float)
    log.info("%s: %d randoms pixels (%d randoms)", args.tracer, len(cache), int(cache["NRAN"].sum()))

    for sim in sim_ids(cfg, args.lensmap, args.sim_start, args.sim_stop):
        out = sidecar_path(cfg, "photo", args.lensmap, args.tracer, sim, hdu)
        if args.skip_existing and os.path.exists(out):
            continue
        t0 = time.time()
        with fits.open(mock_path(cfg, "photo", args.lensmap, args.tracer, sim), memmap=True) as h:
            d = h[hdu].data
            ra, dec = np.asarray(d["RA"], float), np.asarray(d["DEC"], float)
        weights, coeffs = regress_catalog(cfg, args.tracer, ra, dec, templates, rand_pix, rand_w, variants, regions)
        write_sidecar(out, len(ra), weights, coeffs,
                      {"sample": "photo", "tracer": args.tracer, "lensmap": args.lensmap, "sim": sim, "hdu": hdu,
                       "templates": cfg["paths"]["photo_templates"]})
        log.info("%s %s sim %d: N=%d, %.1fs -> %s", args.tracer, args.lensmap, sim, len(ra), time.time() - t0, out)


if __name__ == "__main__":
    main()
