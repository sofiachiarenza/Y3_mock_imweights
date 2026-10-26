"""Linear imaging-systematics regression on the spectroscopic lognormal mocks.

DESI default map lists / z-bins / regions (config.yaml: spec.tracers), fit with
the vendored alaeboss produce_imweights() (imaging_weights/), using the
compressed per-pixel per-z-bin randoms (common.spec_randoms_table).

Variants (config.yaml: spec.variants), fit in the same pass:
  WEIGHT_IMLIN       DESI default
  WEIGHT_IMLIN_CSFD  EBV_DIFF_XX -> EBV_DIFF_XX + (SFD - CSFD): SFD reference swapped for CSFD

One process = one (tracer, lensmap), looping over sims; sidecar per mock:
    outputs/spec/<lensmap>/<T>/mock_simNNNN_hdu1.h5
Weights are full-HDU-length, 1.0 outside the z-bins / imaging footprint.
"""
import argparse
import logging
import os
import shutil
import tempfile
import time

import numpy as np
from astropy.io import fits
from astropy.table import Table

from common import (append_sidecar, existing_columns, load_config, load_randoms_cache, mock_path, photsys_of_pix,
                    radec_to_pix, sidecar_path, sim_ids, spec_randoms_table, spec_region_masks, write_sidecar)
from produce_imweights import get_debv, produce_imweights

log = logging.getLogger("regress_spec")
QSO_FAMILY = ("QSO", "QSO1", "QSO2")


def build_templates(cfg, tracer):
    """{'N': Table, 'S': Table} with the extra EBV_DIFF_*_CSFD columns."""
    hpdir = cfg["paths"]["spec_hpmaps_dir"]
    prefix = cfg["spec"]["tracers"][tracer]["templates"]
    debv = get_debv(cfg["paths"]["debv_map"])
    dust = Table.read(cfg["paths"]["photo_templates"])
    shift = np.asarray(dust["SFD"], float) - np.asarray(dust["CSFD"], float)
    out = {}
    for reg in ("N", "S"):
        t = Table.read(os.path.join(hpdir, f"{prefix}_mapprops_healpix_nested_nside256_{reg}.fits"))
        t["CSFD"] = np.asarray(dust["CSFD"], float)
        for ec in ("GR", "RZ"):
            t[f"EBV_DIFF_{ec}_CSFD"] = np.asarray(debv[f"EBV_DIFF_{ec}"], float) + shift
        out[reg] = t
    return out


def variant_maps(cfg, tracer, variant):
    """Tracer's DESI map list with the variant's replace -> drop -> add applied;
    None if the variant is restricted to other tracers."""
    v = cfg["spec"]["variants"][variant]
    if "tracers" in v and tracer not in v["tracers"]:
        return None
    rep = v.get("replace", {})
    maps = [rep.get(m, m) for m in cfg["spec"]["tracers"][tracer]["maps"]]
    maps = [m for m in maps if m not in v.get("drop", [])]
    return maps + [m for m in v.get("add", []) if m not in maps]


def read_coeffs(directory):
    coeffs = {}
    for fn in sorted(os.listdir(directory)):
        if fn.endswith("_linfitparam.txt"):
            key = fn[: -len("_linfitparam.txt")].split("_", 1)[1]  # drop tracer prefix -> region_zmin_zmax
            with open(os.path.join(directory, fn)) as f:
                coeffs[key] = {line.split()[0]: float(line.split()[1]) for line in f if line.strip()}
    return coeffs


def regress_catalog(cfg, tracer, data, templates, randoms, variants):
    """data: Table with RA, DEC, Z, WEIGHT* (full HDU). Returns ({col: w}, {variant: coeffs})."""
    tcfg = cfg["spec"]["tracers"][tracer]
    zbins = [tuple(z) for z in tcfg["zbins"]]
    data = data.copy(copy_data=False)
    data["PHOTSYS"] = photsys_of_pix(radec_to_pix(data["RA"], data["DEC"]))
    inside = np.flatnonzero(data["PHOTSYS"] != "")
    sub = data[inside]
    kwargs = spec_region_masks(sub) if tracer in QSO_FAMILY else {}
    paths = cfg["paths"]
    weights, coeffs = {}, {}
    for v in variants:
        maps = variant_maps(cfg, tracer, v)
        if maps is None:
            continue
        tmp = tempfile.mkdtemp(prefix="y3imw_")
        try:
            w = produce_imweights(
                data_catalogs=sub, randoms_catalogs=randoms, tracer_type=tracer, redshift_range=zbins,
                templates_maps_path_S=templates["S"], templates_maps_path_N=templates["N"],
                fit_maps=maps,
                debv_map_path=paths["debv_map"],
                sky_res_map_path_north=paths["sky_resid_north"],
                sky_res_map_path_south=paths["sky_resid_south"],
                output_directory=tmp, save_summary_plots=False, nbins=cfg["nbins"], tail=cfg["tail"],
                logger=log, loglevel="WARNING", **kwargs)
            coeffs[v] = read_coeffs(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        full = np.ones(len(data))
        full[inside] = w
        weights[v] = full
    return weights, coeffs


def read_mock(path, hdu):
    cols = ["RA", "DEC", "Z", "WEIGHT", "WEIGHT_FKP", "WEIGHT_SYS", "WEIGHT_ZFAIL"]
    with fits.open(path, memmap=True) as h:
        d = h[hdu].data
        return Table({c: np.asarray(d[c], float) for c in cols})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracer", required=True, choices=["BGS", "LRG1", "LRG2", "LRG3", "QSO", "QSO1", "QSO2"])
    p.add_argument("--lensmap", required=True, choices=["act_dr6", "planck_pr4"])
    p.add_argument("--sim-start", type=int)
    p.add_argument("--sim-stop", type=int, help="inclusive")
    p.add_argument("--hdu", type=int, default=None)
    p.add_argument("--variants", nargs="+", default=None)
    p.add_argument("--output-root", default=None, help="override (tests)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip sims whose sidecar already has every requested variant; if the sidecar exists "
                        "but lacks some, fit only those and append them")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config()
    if args.output_root:
        cfg["paths"]["output_root"] = args.output_root
    hdu = args.hdu or cfg["hdu"]
    variants = args.variants or list(cfg["spec"]["variants"])
    zbins = cfg["spec"]["tracers"][args.tracer]["zbins"]

    templates = build_templates(cfg, args.tracer)
    randoms = spec_randoms_table(load_randoms_cache(cfg, "spec", args.tracer), zbins)
    log.info("%s: %d compressed randoms rows", args.tracer, len(randoms))

    for sim in sim_ids(cfg, args.lensmap, args.sim_start, args.sim_stop):
        out = sidecar_path(cfg, "spec", args.lensmap, args.tracer, sim, hdu)
        todo, append = variants, False
        if args.skip_existing and os.path.exists(out):
            todo = [v for v in variants if v not in existing_columns(out) and variant_maps(cfg, args.tracer, v) is not None]
            if not todo:
                continue
            append = True
        t0 = time.time()
        data = read_mock(mock_path(cfg, "spec", args.lensmap, args.tracer, sim), hdu)
        weights, coeffs = regress_catalog(cfg, args.tracer, data, templates, randoms, todo)
        if append:
            append_sidecar(out, len(data), weights, coeffs)
        else:
            write_sidecar(out, len(data), weights, coeffs,
                          {"sample": "spec", "tracer": args.tracer, "lensmap": args.lensmap, "sim": sim, "hdu": hdu})
        log.info("%s %s sim %d: N=%d, %.1fs -> %s", args.tracer, args.lensmap, sim, len(data), time.time() - t0, out)


if __name__ == "__main__":
    main()
