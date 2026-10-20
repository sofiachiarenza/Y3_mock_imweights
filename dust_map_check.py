"""Map-level dust tests (run after build_dust_templates.py, on a compute node).

Footprints: photo north / south / des (DR9 FRACAREA>0), des_photo-only
(DES & DEC <= -19.6), and the DESI spec footprint (spec hpmaps defined).

Reports (figures/dust_map_check.txt + figures/dust_*.png):
 1. Consistency: dustmaps SFD vs pixweight EBV; HI4PI vs DESI hpmap HI.
 2. Amplitude of D = CSFD - SFD (mean, rms; rms relative to SFD's).
 3. Pearson correlation of D, SFD, CSFD with every other photo template.
 4. Pseudo-C_ell of SFD, CSFD, D on the photo footprint.
 5. CIB-kappa risk: binned correlation coefficient r_ell between D (and SFD,
    CSFD, and the release's LSS/CIB map) and the Planck PR4 / ACT DR6 kappa
    maps on the common footprint.
"""
import os

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from common import NSIDE, imaging_footprints, load_config, pix_to_radec

LMAX = 3 * NSIDE - 1
OTHER = ["STARDENS", "HI", "GALDEPTH_G", "GALDEPTH_R", "GALDEPTH_Z", "PSFDEPTH_W1", "PSFSIZE_G", "PSFSIZE_R", "PSFSIZE_Z"]
BINS = np.arange(2, LMAX + 1, 30)


def ring(m):
    return hp.reorder(m, n2r=True)


def pcl(a, b, mask):
    """fsky-corrected pseudo-C_ell of mean-subtracted maps (NEST in)."""
    am, bm = [np.where(mask, x - x[mask].mean(), 0.0) for x in (a, b)]
    return hp.anafast(ring(am), ring(bm), lmax=LMAX) / mask.mean()


def binned(cl):
    ell = np.arange(len(cl))
    return np.array([cl[(ell >= lo) & (ell < hi)].mean() for lo, hi in zip(BINS[:-1], BINS[1:])])


def kappa_map(alm_path, galactic):
    alm = np.nan_to_num(hp.read_alm(alm_path).astype(complex))  # a few NaN coeffs (monopole/dipole) in both files
    lmax_in = hp.Alm.getlmax(len(alm))
    if lmax_in > LMAX:
        alm = hp.resize_alm(alm, lmax_in, lmax_in, LMAX, LMAX)
    if galactic:  # Planck PR4 is in Galactic coordinates; templates are equatorial
        hp.Rotator(coord=["G", "C"]).rotate_alm(alm, lmax=LMAX, inplace=True)
    return hp.reorder(hp.alm2map(alm, NSIDE, lmax=LMAX), r2n=True)


def lensing_mask(path, galactic):
    m = hp.read_map(path, dtype=np.float64)
    if hp.get_nside(m) != NSIDE:
        m = hp.ud_grade(m, NSIDE)
    if galactic:  # rotate by pixel lookup (mask is ~binary)
        th, ph = hp.pix2ang(NSIDE, np.arange(len(m)))
        thg, phg = hp.Rotator(coord=["C", "G"])(th, ph)
        m = m[hp.ang2pix(NSIDE, thg, phg)]
    return hp.reorder(m, r2n=True) > 0.5


def main():
    cfg = load_config()
    figdir = cfg["paths"]["figures"]
    os.makedirs(figdir, exist_ok=True)
    out = []

    def say(s=""):
        print(s, flush=True)
        out.append(s)

    t = Table.read(cfg["paths"]["photo_templates"])
    m = {c: np.asarray(t[c], float) for c in t.colnames}
    m["D"] = m["CSFD"] - m["SFD"]
    # lss_intensity carries ~1e30 sentinels outside the CSFD LSS-correction footprint
    # (where CSFD = SFD); set them to 0 so the LSS row is meaningful.
    m["LSS"] = np.where(np.abs(m["LSS"]) < 10, m["LSS"], 0.0)

    foot = imaging_footprints()
    good = m["FRACAREA"] > 0
    _, dec = pix_to_radec(np.arange(len(good)))
    hpdir = cfg["paths"]["spec_hpmaps_dir"]
    specN = Table.read(os.path.join(hpdir, "LRG_mapprops_healpix_nested_nside256_N.fits"))
    specS = Table.read(os.path.join(hpdir, "LRG_mapprops_healpix_nested_nside256_S.fits"))
    spec_hi = np.where(np.asarray(specN["HI"]) > 0, np.asarray(specN["HI"]), np.asarray(specS["HI"]))
    spec = spec_hi > 0
    regions = {
        "north": good & foot["north"],
        "south": good & foot["south"],
        "des": good & foot["des"],
        "des_photo-only": good & foot["des"] & (dec <= -19.6),
        "photo_all": good & (foot["north"] | foot["south"] | foot["des"]),
        "spec_footprint": good & spec,
        "photo_all_csfd_cosmo": good & (foot["north"] | foot["south"] | foot["des"]) & (m["CSFD_COSMO"] > 0.5),
    }

    say("== 1. consistency")
    for name, r in regions.items():
        e, s = m["EBV_PIXWEIGHT"][r], m["SFD"][r]
        line = f"{name:15s} npix={r.sum():7d}  SFD/EBV_pixweight median={np.median(s / e):.4f}  corr={np.corrcoef(s, e)[0, 1]:.5f}"
        rs = r & spec
        if rs.sum() > 100:
            line += f"  | HI4PI vs spec HI: median ratio={np.median(m['HI'][rs] / spec_hi[rs]):.4f} corr={np.corrcoef(m['HI'][rs], spec_hi[rs])[0, 1]:.5f}"
        say(line)

    say("\n== 2. D = CSFD - SFD amplitude")
    for name, r in regions.items():
        d, s = m["D"][r], m["SFD"][r]
        say(f"{name:15s} <SFD>={s.mean():.4f} rms(SFD)={s.std():.4f}  <D>={d.mean():+.4f} rms(D)={d.std():.4f}"
            f"  rms(D)/rms(SFD)={d.std() / s.std():.3f}  corr(CSFD,SFD)={np.corrcoef(m['CSFD'][r], s)[0, 1]:.4f}")

    say("\n== 3. Pearson correlation with other templates")
    for name, r in regions.items():
        say(f"-- {name}")
        say(" " * 12 + "".join(f"{c:>12s}" for c in OTHER))
        for dm in ("D", "SFD", "CSFD"):
            say(f"{dm:>12s}" + "".join(f"{np.corrcoef(m[dm][r], m[c][r])[0, 1]:12.3f}" for c in OTHER))

    say("\n== 4. pseudo-C_ell on photo_all")
    r = regions["photo_all"]
    ell_c = 0.5 * (BINS[:-1] + BINS[1:])
    fig, ax = plt.subplots(figsize=(6, 4))
    for dm in ("SFD", "CSFD", "D"):
        ax.loglog(ell_c, binned(pcl(m[dm], m[dm], r)), label=dm)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell$ [mag$^2$]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "dust_cl.png"), dpi=130)
    say("saved dust_cl.png")

    say("\n== 5. correlation coefficient with CMB lensing kappa")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, lens in zip(axes, ("planck_pr4", "act_dr6")):
        kp = cfg["paths"]["kappa"]
        gal = kp["galactic"][lens]
        k = kappa_map(kp[lens], gal)
        lm = lensing_mask(kp[lens + "_mask"], gal)
        rr = r & lm
        ckk = binned(pcl(k, k, rr))
        say(f"-- {lens}: common npix={rr.sum()} (fsky={rr.mean():.3f})")
        say(f"{'ell bin':>12s}" + "".join(f"{f'{lo}-{hi}':>10s}" for lo, hi in zip(BINS[:-1], BINS[1:])))
        for dm in ("D", "SFD", "CSFD", "LSS"):
            rho = binned(pcl(m[dm], k, rr)) / np.sqrt(binned(pcl(m[dm], m[dm], rr)) * ckk)
            say(f"{'r(' + dm + ',k)':>12s}" + "".join(f"{x:10.3f}" for x in rho))
            ax.plot(ell_c, rho, label=dm)
        sig = 1 / np.sqrt((2 * ell_c + 1) * np.diff(BINS) * rr.mean())
        ax.fill_between(ell_c, -sig, sig, color="0.85", label=r"$\pm1/\sqrt{N_{\rm modes}}$")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(lens)
        ax.set_xlabel(r"$\ell$")
    axes[0].set_ylabel(r"$r_\ell$(dust, $\kappa$)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "dust_kappa_corr.png"), dpi=130)

    plt.figure(figsize=(10, 6))
    dmap = np.where(r, m["D"], hp.UNSEEN)
    hp.mollview(ring(dmap), fig=plt.gcf().number, title="CSFD - SFD [mag] on photo footprint", min=-0.02, max=0.02, cmap="RdBu_r")
    plt.savefig(os.path.join(figdir, "dust_diff_map.png"), dpi=130)

    with open(os.path.join(figdir, "dust_map_check.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("done")


if __name__ == "__main__":
    main()
