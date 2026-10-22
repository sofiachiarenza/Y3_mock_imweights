"""Test 2: is CSFD - SFD the CIB? Check against an INDEPENDENT CIB map.

The CSFD 'LSS' component (= SFD - CSFD) is reconstructed from galaxy cross-
correlations (Chiang 2023), so its correlation with kappa / galaxies is partly
by construction. Lenz, Dore & Lagache (2019) CIB maps are built differently:
Planck HFI minus an HI4PI-based dust model, on low-N_HI sky. If D = CSFD - SFD
correlates with the Lenz CIB, the CIB attribution doesn't rely on galaxies.

Reports (figures/cib_check.txt, figures/cib_check.png), per CIB frequency/mask
found under data/ext/lenz19_cib/<freq>/<mask>/:
 1. r_ell(X, CIB) for X = D, SFD, CSFD, LSS, HI (HI is a control: the Lenz CIB
    is HI-cleaned, so r(HI, CIB) should be ~0).
 2. r_ell(kappa, CIB) for Planck PR4 and ACT DR6 (reference).
 3. Euclid-style test: measured r_ell(D, kappa) vs the prediction through the
    CIB, r(D, CIB) * r(CIB, kappa), on the same footprint.
Footprints: CIB mask, and CIB mask & photo footprint & CSFD cosmology mask.
CIB maps are averaged over the 64 nside-2048 sub-pixels of each nside-256
pixel; only pixels fully inside the CIB mask are kept.
"""
import glob
import os

import astropy.units as u
import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from common import NPIX, NSIDE, imaging_footprints, load_config
from dust_map_check import BINS, binned, kappa_map, lensing_mask, pcl

NSUB = 2048
NCHILD = (NSUB // NSIDE) ** 2


def read_hpx(path):
    with fits.open(path) as h:
        arr = np.asarray(h[1].data.field(0), float).ravel()
        nest = str(h[1].header.get("ORDERING", "RING")).upper().startswith("NEST")
        assert str(h[1].header.get("COORDSYS", "G")).upper().startswith("G")
    return arr, nest


def to_eq256(cib, mask, nest):
    """Galactic nside-2048 map + boolean mask -> equatorial nside-256 NEST
    (mean over sub-pixels, kept only where all 64 sub-pixels are unmasked)."""
    val, frac = np.zeros(NPIX), np.zeros(NPIX)
    step = NPIX // 16
    for p0 in range(0, NPIX, step):
        parents = np.arange(p0, min(p0 + step, NPIX))
        child = (parents[:, None] * NCHILD + np.arange(NCHILD)[None, :]).ravel()
        th, ph = hp.pix2ang(NSUB, child, nest=True)
        gal = SkyCoord(ra=np.degrees(ph) * u.deg, dec=(90 - np.degrees(th)) * u.deg, frame="icrs").galactic
        idx = hp.ang2pix(hp.npix2nside(len(cib)), gal.l.deg, gal.b.deg, nest=nest, lonlat=True)
        val[parents] = cib[idx].reshape(-1, NCHILD).mean(1)
        frac[parents] = mask[idx].reshape(-1, NCHILD).mean(1)
    return val, frac == 1.0


def rho(a, b, m):
    return binned(pcl(a, b, m)) / np.sqrt(binned(pcl(a, a, m)) * binned(pcl(b, b, m)))


def main():
    cfg = load_config()
    figdir = cfg["paths"]["figures"]
    out = []

    def say(s=""):
        print(s, flush=True)
        out.append(s)

    t = Table.read(cfg["paths"]["photo_templates"])
    m = {c: np.asarray(t[c], float) for c in ("SFD", "CSFD", "LSS", "HI", "CSFD_COSMO", "FRACAREA")}
    m["D"] = m["CSFD"] - m["SFD"]
    foot = imaging_footprints()
    photo = (m["FRACAREA"] > 0) & (foot["north"] | foot["south"] | foot["des"])
    cosmo = m["CSFD_COSMO"] > 0.5

    kp = cfg["paths"]["kappa"]
    kappa = {}
    for lens in ("planck_pr4", "act_dr6"):
        kappa[lens] = (kappa_map(kp[lens], kp["galactic"][lens]), lensing_mask(kp[lens + "_mask"], kp["galactic"][lens]))

    ell_c = 0.5 * (BINS[:-1] + BINS[1:])
    hdr = f"{'':>22s}" + "".join(f"{f'{lo}-{hi}':>9s}" for lo, hi in zip(BINS[:-1], BINS[1:]))
    root = os.path.join(cfg["paths"]["ext_dir"], "lenz19_cib")
    cache_dir = os.path.join(os.path.dirname(cfg["paths"]["photo_templates"]), "cib_nside256")
    os.makedirs(cache_dir, exist_ok=True)
    maps = sorted(glob.glob(os.path.join(root, "*", "*", "*", "cib_fullmission.hpx.fits")))
    fig, axes = plt.subplots(len(maps), 3, figsize=(15, 3.8 * len(maps)), squeeze=False)

    for row, path in zip(axes, maps):
        d = os.path.dirname(path)
        freq, maskname = d.split(os.sep)[-3], d.split(os.sep)[-1]
        label = f"{freq} GHz, {maskname}"
        cache = os.path.join(cache_dir, f"cib_{freq}_{maskname}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            cib, cmask = z["cib"], z["mask"]
        else:
            arr, nest = read_hpx(path)
            mk, _ = read_hpx(os.path.join(d, "mask_bool.hpx.fits"))
            cib, cmask = to_eq256(arr, mk > 0, nest)
            np.savez(cache, cib=cib, mask=cmask)

        regions = {"CIB mask": cmask, "CIB & photo & CSFD-cosmo": cmask & photo & cosmo}
        say(f"\n==== {label}")
        for rname, reg in regions.items():
            say(f"-- {rname}: npix={reg.sum()} fsky={reg.mean():.4f}")
            say(hdr)
            for x in ("D", "LSS", "SFD", "CSFD", "HI"):
                say(f"{'r(' + x + ', CIB)':>22s}" + "".join(f"{v:9.3f}" for v in rho(m[x], cib, reg)))
        reg = regions["CIB & photo & CSFD-cosmo"]
        ax = row[0]
        for x in ("D", "LSS", "SFD", "CSFD", "HI"):
            ax.plot(ell_c, rho(m[x], cib, reg), marker="o", ms=3, label=x)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"r(X, CIB {label})\nCIB & photo & CSFD-cosmo")
        ax.set_xscale("log")
        ax.legend(fontsize=7)
        for ax, lens in zip(row[1:], ("planck_pr4", "act_dr6")):
            k, lm = kappa[lens]
            rr = reg & lm
            r_dk = rho(m["D"], k, rr)
            r_dc, r_ck = rho(m["D"], cib, rr), rho(cib, k, rr)
            say(f"-- {lens} x CIB {label}: npix={rr.sum()} fsky={rr.mean():.4f}")
            say(hdr)
            say(f"{'r(CIB, kappa)':>22s}" + "".join(f"{v:9.3f}" for v in r_ck))
            say(f"{'r(D, kappa) measured':>22s}" + "".join(f"{v:9.3f}" for v in r_dk))
            say(f"{'r(D,CIB) r(CIB,kappa)':>22s}" + "".join(f"{v:9.3f}" for v in r_dc * r_ck))
            sig = 1 / np.sqrt((2 * ell_c + 1) * np.diff(BINS) * rr.mean())
            ax.fill_between(ell_c, -sig, sig, color="0.85", label=r"$\pm1/\sqrt{N_{\rm modes}}$")
            ax.plot(ell_c, r_dk, "o", ms=4, label=r"measured $r(D,\kappa)$")
            ax.plot(ell_c, r_dc * r_ck, "-", label=r"$r(D,{\rm CIB})\,r({\rm CIB},\kappa)$")
            ax.plot(ell_c, r_ck, "--", lw=1, label=r"$r({\rm CIB},\kappa)$")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xscale("log")
            ax.set_title(f"{lens}, CIB {label}")
            ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\ell$")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "cib_check.png"), dpi=120)
    with open(os.path.join(figdir, "cib_check.txt"), "w") as f:
        f.write("D = CSFD - SFD. CIB = Lenz, Dore & Lagache (2019).\n" + "\n".join(out) + "\n")
    print("done")


if __name__ == "__main__":
    main()
