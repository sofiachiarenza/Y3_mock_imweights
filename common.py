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


class RandomsCompressor:
    """Accumulate randoms chunk by chunk into per-pixel (per z-bin) summed weights.
    z-bin membership uses the same strict (Z > zmin) & (Z < zmax) as produce_imweights."""

    def __init__(self, zbins=None):
        self.zbins = zbins or [None]
        nz = len(self.zbins)
        self.wsum = np.zeros(nz * NPIX)
        self.nsum = np.zeros(nz * NPIX, dtype=np.int64)

    def add(self, ra, dec, w=None, z=None):
        pix = radec_to_pix(ra, dec)
        size = len(self.wsum)
        if self.zbins[0] is None:
            self.wsum += np.bincount(pix, weights=w, minlength=size) if w is not None else np.bincount(pix, minlength=size)
            self.nsum += np.bincount(pix, minlength=size)
            return
        for k, (lo, hi) in enumerate(self.zbins):
            sel = (z > lo) & (z < hi)
            self.wsum += np.bincount(pix[sel] + k * NPIX, weights=None if w is None else w[sel], minlength=size)
            self.nsum += np.bincount(pix[sel] + k * NPIX, minlength=size)

    def table(self, **meta):
        keep = np.flatnonzero(self.nsum > 0)
        t = Table({"PIX": keep % NPIX, "ZBIN": keep // NPIX, "W": self.wsum[keep].astype(float), "NRAN": self.nsum[keep]})
        t.meta.update({"NSIDE": NSIDE, "ORDERING": "NEST", **meta})
        return t


def spec_random_weight(r):
    """produce_imweights randoms weight convention."""
    return r["WEIGHT"] * r["WEIGHT_FKP"] / r["WEIGHT_SYS"] / r["WEIGHT_ZFAIL"]


def build_randoms_cache(cfg, sample, tracer, max_rows=None, log=print):
    """Chunked read of mock_randoms_<T>.h5 -> per-pixel (per z-bin for spec)
    summed random weight. Spec weight = WEIGHT*WEIGHT_FKP/WEIGHT_SYS/WEIGHT_ZFAIL
    (the produce_imweights convention); photo weight = 1 per random.
    Returns a Table with PIX, ZBIN, W, NRAN."""
    comp = RandomsCompressor(cfg["spec"]["tracers"][tracer]["zbins"] if sample == "spec" else None)
    path = randoms_h5_path(cfg, sample, tracer)
    with h5py.File(path, "r") as f:
        ds = f["__astropy_table__"]
        n = ds.shape[0] if max_rows is None else min(max_rows, ds.shape[0])
        for start in range(0, n, CHUNK):
            r = ds[start:min(start + CHUNK, n)]
            if sample == "spec":
                comp.add(r["RA"], r["DEC"], spec_random_weight(r), r["Z"])
            else:
                comp.add(r["RA"], r["DEC"])
            log(f"[{sample} {tracer}] randoms rows {min(start + CHUNK, n):,}/{n:,}")
    return comp.table(SAMPLE=sample, TRACER=tracer, SOURCE=path, NROWS=int(n))


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


def compare_sidecars(new, ref, log=print):
    """max |w_new - w_ref| per weight column present in both; returns the overall max."""
    worst = 0.0
    with h5py.File(new, "r") as a, h5py.File(ref, "r") as b:
        for col in sorted(k for k in a.keys() if k.startswith("WEIGHT")):
            if col not in b:
                log(f"  {col:28s} not in reference")
                continue
            d = float(np.max(np.abs(a[col][:] - b[col][:])))
            worst = max(worst, d)
            log(f"  {col:28s} max|dw| = {d:.3e}")
    return worst


def read_data_sidecar(path):
    """-> ({weight column: array}, {extra column: array}) of a real-data sidecar."""
    with h5py.File(path, "r") as f:
        w = {k: f[k][:] for k in f.keys() if k.startswith("WEIGHT")}
        ex = {k: f["extra"][k][:] for k in f["extra"].keys()} if "extra" in f else {}
    return w, ex


def existing_columns(path):
    with h5py.File(path, "r") as f:
        return {k for k in f.keys() if k.startswith("WEIGHT")}


def append_sidecar(path, nrows, weights, coeffs):
    """Add new weight columns (+ coeffs) to an existing sidecar; written to a copy then swapped in."""
    import shutil
    tmp = path + ".tmp"
    shutil.copyfile(path, tmp)
    with h5py.File(tmp, "a") as f:
        assert int(f.attrs["NROWS"]) == nrows, (path, f.attrs["NROWS"], nrows)
        for col, w in weights.items():
            assert len(w) == nrows and col not in f, col
            f.create_dataset(col, data=np.asarray(w, "f8"), compression="gzip")
        for variant, fits_ in coeffs.items():
            gv = f["coeffs"].create_group(variant)
            for key, pars in fits_.items():
                gk = gv.create_group(key)
                for par, val in pars.items():
                    gk.attrs[par] = float(val)
    os.replace(tmp, path)


def write_sidecar(path, nrows, weights, coeffs, meta, extra=None):
    """weights: {column: full-HDU-length array}; coeffs: {variant: {key: {par: value}}};
    extra: optional {name: array} written with its own dtype under extra/ (real catalogs:
    TARGETID, RA, DEC and the non-imaging analysis weight, so later steps need no catalog reads)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.attrs["NROWS"] = nrows
        for k, v in meta.items():
            f.attrs[k] = v
        for col, w in weights.items():
            assert len(w) == nrows, (col, len(w), nrows)
            f.create_dataset(col, data=np.asarray(w, "f8"), compression="gzip")
        for col, v in (extra or {}).items():
            assert len(v) == nrows, (col, len(v), nrows)
            f.create_dataset(f"extra/{col}", data=np.asarray(v), compression="gzip")
        g = f.create_group("coeffs")
        for variant, fits_ in coeffs.items():
            gv = g.create_group(variant)
            for key, pars in fits_.items():
                gk = gv.create_group(key)
                for par, val in pars.items():
                    gk.attrs[par] = float(val)
    os.replace(tmp, path)
