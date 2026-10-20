"""Shared helpers: config, footprints, compressed randoms cache, sidecar I/O.

Compressed randoms
------------------
LinearRegressor (imaging_weights/linear_regression.py) takes its outlier
percentiles (cut_outliers) and histogram bin edges (prepare) from the DATA
templates only; randoms only enter through weighted per-bin sums. Every
template is constant inside an nside-256 pixel, so replacing the randoms by one
row per pixel (per z-bin for spec) carrying the summed random weight is exact
(up to floating-point summation order). This removes the 10^8-10^9-row randoms
from every fit and also removes the downsampling the older scripts needed.
"""
import os
import sys

import h5py
import healpy as hp
import numpy as np
import yaml
from astropy.table import Table
from regressis import footprint as rfp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "imaging_weights"))

NSIDE = 256
NPIX = hp.nside2npix(NSIDE)
CHUNK = 20_000_000


def load_config(path=os.path.join(HERE, "config.yaml")):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("ext_dir", "photo_templates", "output_root", "figures"):
        p = cfg["paths"][key]
        cfg["paths"][key] = p if os.path.isabs(p) else os.path.join(HERE, p)
    # randoms cache stays put even when output_root is overridden (pilot/tests)
    cfg["paths"]["cache_root"] = os.path.join(cfg["paths"]["output_root"], "randoms_cache")
    return cfg


def sim_ids(cfg, lensmap, start=None, stop=None):
    lo, hi = cfg["sims"][lensmap]
    lo = lo if start is None else start
    hi = hi if stop is None else stop
    return list(range(lo, hi + 1))


# --------------------------------------------------------------------------
# Footprints (regressis DR9Footprint at nside 256, NEST -- same lookup as
# PNG_DR2/scripts/lognormal_catalog_io.py and Y3_AMR_nersc/regress_photo_mocks.py)
# --------------------------------------------------------------------------
_FOOT = None


def imaging_footprints():
    """-> dict north/south/des of boolean nside-256 NEST pixel masks.
    DR9Footprint's `south` excludes DES; DES is its own mask."""
    global _FOOT
    if _FOOT is None:
        foot = rfp.DR9Footprint(NSIDE, mask_lmc=False, clear_south=True, mask_around_des=False, cut_desi=False)
        n, s, d = foot.get_imaging_surveys()
        _FOOT = {"north": np.asarray(n), "south": np.asarray(s), "des": np.asarray(d)}
    return _FOOT


def radec_to_pix(ra, dec):
    return hp.ang2pix(NSIDE, np.radians(90.0 - np.asarray(dec)), np.radians(np.asarray(ra)), nest=True)


def pix_to_radec(pix):
    theta, phi = hp.pix2ang(NSIDE, pix, nest=True)
    return np.degrees(phi), 90.0 - np.degrees(theta)


def photsys_of_pix(pix):
    """'N' / 'S' (south incl. DES, DESI convention) / '' outside both."""
    f = imaging_footprints()
    out = np.full(len(pix), "", dtype="<U1")
    out[f["north"][pix]] = "N"
    out[(f["south"] | f["des"])[pix]] = "S"
    return out


def spec_region_masks(table):
    """QSO-style N / SnotDES / DES row-mask callables for produce_imweights
    (mirrors PNG_DR2 fit_lognormal_imaging_weights.qso_region_kwargs)."""
    des = imaging_footprints()["des"]

    def _des(t):
        return des[radec_to_pix(t["RA"], t["DEC"])]

    return dict(
        photometric_regions=["N", "SnotDES", "DES"],
        region_row_masks={
            "N": lambda t: np.asarray(t["PHOTSYS"] == "N"),
            "SnotDES": lambda t: np.asarray(t["PHOTSYS"] == "S") & ~_des(t),
            "DES": lambda t: np.asarray(t["PHOTSYS"] == "S") & _des(t),
        },
        region_template_key={"N": "N", "SnotDES": "S", "DES": "S"},
    )


# --------------------------------------------------------------------------
# Compressed randoms cache
# --------------------------------------------------------------------------
def randoms_h5_path(cfg, sample, tracer):
    root = cfg["paths"]["spec_mock_root" if sample == "spec" else "photo_mock_root"]
    return os.path.join(root, f"mock_randoms_{tracer}.h5")


def cache_path(cfg, sample, tracer):
    return os.path.join(cfg["paths"]["cache_root"], f"{sample}_{tracer}.fits")


def build_randoms_cache(cfg, sample, tracer, max_rows=None, log=print):
    """Chunked read of mock_randoms_<T>.h5 -> per-pixel (per z-bin for spec)
    summed random weight. Spec weight = WEIGHT*WEIGHT_FKP/WEIGHT_SYS/WEIGHT_ZFAIL
    (the produce_imweights convention); photo weight = 1 per random.
    z-bin membership uses the same strict (Z > zmin) & (Z < zmax) as
    produce_imweights. Returns a Table with PIX, ZBIN, W, NRAN."""
    zbins = cfg["spec"]["tracers"][tracer]["zbins"] if sample == "spec" else [None]
    nz = len(zbins)
    wsum = np.zeros(nz * NPIX)
    nsum = np.zeros(nz * NPIX, dtype=np.int64)
    path = randoms_h5_path(cfg, sample, tracer)
    with h5py.File(path, "r") as f:
        ds = f["__astropy_table__"]
        n = ds.shape[0] if max_rows is None else min(max_rows, ds.shape[0])
        for start in range(0, n, CHUNK):
            r = ds[start:min(start + CHUNK, n)]
            pix = radec_to_pix(r["RA"], r["DEC"])
            if sample == "spec":
                w = r["WEIGHT"] * r["WEIGHT_FKP"] / r["WEIGHT_SYS"] / r["WEIGHT_ZFAIL"]
                for k, (lo, hi) in enumerate(zbins):
                    sel = (r["Z"] > lo) & (r["Z"] < hi)
                    wsum += np.bincount(pix[sel] + k * NPIX, weights=w[sel], minlength=nz * NPIX)
                    nsum += np.bincount(pix[sel] + k * NPIX, minlength=nz * NPIX)
            else:
                wsum += np.bincount(pix, minlength=NPIX).astype(float)
                nsum += np.bincount(pix, minlength=NPIX)
            log(f"[{sample} {tracer}] randoms rows {min(start + CHUNK, n):,}/{n:,}")
    keep = np.flatnonzero(nsum > 0)
    t = Table({"PIX": keep % NPIX, "ZBIN": keep // NPIX, "W": wsum[keep], "NRAN": nsum[keep]})
    t.meta.update({"SAMPLE": sample, "TRACER": tracer, "SOURCE": path, "NROWS": int(n), "NSIDE": NSIDE, "ORDERING": "NEST"})
    return t


def load_randoms_cache(cfg, sample, tracer):
    return Table.read(cache_path(cfg, sample, tracer))


def spec_randoms_table(cache, zbins):
    """Expand the cache into the clustering-randoms Table produce_imweights
    expects: one row per (pixel, z-bin) at the pixel centre, Z = bin centre,
    WEIGHT = summed weight, other weight columns 1."""
    ra, dec = pix_to_radec(np.asarray(cache["PIX"]))
    zc = np.array([0.5 * (lo + hi) for lo, hi in zbins])[np.asarray(cache["ZBIN"])]
    one = np.ones(len(cache))
    t = Table({"RA": ra, "DEC": dec, "Z": zc, "WEIGHT": np.asarray(cache["W"], float),
               "WEIGHT_FKP": one, "WEIGHT_SYS": one, "WEIGHT_ZFAIL": one})
    t["PHOTSYS"] = photsys_of_pix(np.asarray(cache["PIX"]))
    return t[t["PHOTSYS"] != ""]


# --------------------------------------------------------------------------
# Mock catalogs + sidecars
# --------------------------------------------------------------------------
def mock_path(cfg, sample, lensmap, tracer, sim):
    root = cfg["paths"]["spec_mock_root" if sample == "spec" else "photo_mock_root"]
    return os.path.join(root, lensmap, tracer, f"mock_sim{sim:04d}.fits")


def sidecar_path(cfg, sample, lensmap, tracer, sim, hdu):
    return os.path.join(cfg["paths"]["output_root"], sample, lensmap, tracer, f"mock_sim{sim:04d}_hdu{hdu}.h5")


def write_sidecar(path, nrows, weights, coeffs, meta):
    """weights: {column: full-HDU-length array}; coeffs: {variant: {key: {par: value}}}."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.attrs["NROWS"] = nrows
        for k, v in meta.items():
            f.attrs[k] = v
        for col, w in weights.items():
            assert len(w) == nrows, (col, len(w), nrows)
            f.create_dataset(col, data=np.asarray(w, "f8"), compression="gzip")
        g = f.create_group("coeffs")
        for variant, fits_ in coeffs.items():
            gv = g.create_group(variant)
            for key, pars in fits_.items():
                gk = gv.create_group(key)
                for par, val in pars.items():
                    gk.attrs[par] = float(val)
    os.replace(tmp, path)
