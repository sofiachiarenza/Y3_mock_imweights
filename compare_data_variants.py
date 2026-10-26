"""Compare imaging-weight variants on the REAL catalogs: Cgg and Cg-kappa (CLAUDE.md tasks 1-2).

Per tracer, reading the data sidecar (regress_{photo,spec}_data.py) and its randoms cache:
  * delta_g at nside 256 from the counts weighted by W_ANALYSIS * w_variant (W_ANALYSIS
    is the analysis weight without the imaging part), over the summed randoms weight, on
    the mask "randoms >= 20% of the median" (as in compare_variants.py)
  * shot-noise-subtracted pseudo-C_ell^gg / fsky (compare_variants.cl_of)
  * pseudo-C_ell^g-kappa / fsky for ACT DR6 and Planck PR4 on mask & lensing mask
  * per bin: the ratios C(variant)/C(baseline WEIGHT_IMLIN), the baseline/unweighted ratio,
    and the Cg-kappa shift in units of the Knox error on Cg-kappa,
    sigma^2 = (C^gg_tot C^kk_tot + C^gk^2) / ((2l+1) dl fsky)
The variants share the footprint and the mask, so mode coupling cancels to first order
in the ratios. These are diagnostics, not the analysis bandpowers.

    python compare_data_variants.py --sample photo --tracers BGS LRG1 LRG2 LRG3
    python compare_data_variants.py --sample spec --tracers BGS LRG1 LRG2 LRG3 QSO QSO1 QSO2
"""
import argparse
import os

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from common import NPIX, load_config, radec_to_pix, read_data_sidecar
from compare_variants import EDGES, LMAX, binned, cl_of
from dust_map_check import kappa_map, lensing_mask

LENSMAPS = ("act_dr6", "planck_pr4")


def data_paths(cfg, sample, tracer):
    root = cfg["paths"]["output_root"]
    n = cfg["data"]["n_randoms"]
    return (os.path.join(root, f"{sample}_data", tracer, "data.h5"),
            os.path.join(cfg["paths"]["cache_root"], f"{sample}_data_{tracer}_nran{n}.fits"))


def randoms_map(cache_path):
    c = Table.read(cache_path)
    return np.bincount(np.asarray(c["PIX"]), weights=np.asarray(c["W"], float), minlength=NPIX)


def delta_map(pix, w, R, mask):
    n = np.bincount(pix, weights=w, minlength=NPIX)
    alpha = n[mask].sum() / R[mask].sum()
    return np.where(mask, n / np.where(mask, alpha * R, 1.0) - 1.0, 0.0)


def cross(a, b, mask):
    """fsky-corrected pseudo-C_ell of two NEST maps already zero outside mask."""
    return binned(hp.anafast(hp.reorder(a, n2r=True), hp.reorder(b, n2r=True), lmax=LMAX) / mask.mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", required=True, choices=["photo", "spec"])
    p.add_argument("--tracers", nargs="+", required=True)
    args = p.parse_args()
    cfg = load_config()
    figdir = os.path.join(cfg["paths"]["figures"], "data_variants")
    os.makedirs(figdir, exist_ok=True)
    kp = cfg["paths"]["kappa"]
    kappa = {L: (kappa_map(kp[L], kp["galactic"][L]), lensing_mask(kp[L + "_mask"], kp["galactic"][L])) for L in LENSMAPS}
    ell_c = 0.5 * (EDGES[:-1] + EDGES[1:] - 1)
    nmodes = (EDGES[1:] - EDGES[:-1]) * (2 * ell_c + 1)

    for tracer in args.tracers:
        side, cache = data_paths(cfg, args.sample, tracer)
        if not os.path.exists(side):
            print(f"{tracer}: no sidecar {side}")
            continue
        ws, ex = read_data_sidecar(side)
        pix = radec_to_pix(ex["RA"], ex["DEC"])
        wa = ex["W_ANALYSIS"]
        R = randoms_map(cache)
        mask = R >= 0.2 * np.median(R[R > 0])
        cols = {"UNWEIGHTED": np.ones(len(pix)), **ws}
        base = "WEIGHT_IMLIN"
        lines = [f"# {args.sample} data {tracer}: N={len(pix)}, fsky(mask)={mask.mean():.4f}, W_ANALYSIS x variant",
                 "# ell bins: " + " ".join(f"{lo}-{hi - 1}" for lo, hi in zip(EDGES[:-1], EDGES[1:]))]
        cgg, cgk, sig = {}, {L: {} for L in LENSMAPS}, {}
        deltas = {}
        for k, w in cols.items():
            cgg[k] = cl_of(pix, wa * w, R, mask)
            deltas[k] = delta_map(pix, wa * w, R, mask)
        for L, (kmap, lm) in kappa.items():
            mm = mask & lm
            km = np.where(mm, kmap - kmap[mm].mean(), 0.0)
            ckk = cross(km, km, mm)
            for k in cols:
                cgk[L][k] = cross(np.where(mm, deltas[k], 0.0), km, mm)
            dg = np.where(mm, deltas[base], 0.0)
            cgg_tot = cross(dg, dg, mm)  # includes shot noise: the Knox variance wants the total
            sig[L] = np.sqrt((cgg_tot * ckk + cgk[L][base] ** 2) / (nmodes * mm.mean()))
            lines.append(f"# {L}: fsky(mask & lensing mask)={mm.mean():.4f}")

        def row(label, vals, fmt="7.4f"):
            lines.append(f"{label:44s} " + " ".join(f"{v:{fmt}}" for v in vals))

        lines.append("## C_gg ratios")
        row("baseline/unweighted", cgg[base] / cgg["UNWEIGHTED"])
        for k in ws:
            if k != base:
                row(f"{k}/baseline", cgg[k] / cgg[base])
        for L in LENSMAPS:
            lines.append(f"## C_gk {L}: baseline C_gk x1e7 and ratios; [shift/sigma_Knox]")
            row("C_gk baseline x1e7", cgk[L][base] * 1e7, "7.3f")
            row("baseline/unweighted", cgk[L][base] / cgk[L]["UNWEIGHTED"])
            row("  (base-unw)/sigma", (cgk[L][base] - cgk[L]["UNWEIGHTED"]) / sig[L], "7.2f")
            for k in ws:
                if k != base:
                    row(f"{k}/baseline", cgk[L][k] / cgk[L][base])
                    row(f"  ({k}-base)/sigma", (cgk[L][k] - cgk[L][base]) / sig[L], "7.2f")
        for k in ws:
            if k != base:
                sel = ws[base] != 1
                lines.append(f"rms(w_{k} - w_baseline) = {np.std(ws[k][sel] - ws[base][sel]):.2e}")

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for k in cols:
            if k == base:
                continue
            axes[0].plot(ell_c, cgg[k] / cgg[base], marker="o", ms=3, label=f"{k}/baseline")
            for ax, L in zip(axes[1:], LENSMAPS):
                ax.plot(ell_c, (cgk[L][k] - cgk[L][base]) / sig[L], marker="o", ms=3, label=k)
        axes[0].axhline(1, color="k", lw=0.5)
        axes[0].set_ylabel(r"$C_\ell^{gg}$ ratio")
        for ax, L in zip(axes[1:], LENSMAPS):
            ax.axhline(0, color="k", lw=0.5)
            ax.axhspan(-1, 1, color="0.9")
            ax.set_ylabel(r"$(C_\ell^{g\kappa}-C_\ell^{g\kappa,\rm base})/\sigma_{\rm Knox}$")
            ax.set_title(L)
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlabel(r"$\ell$")
            ax.legend(fontsize=7)
        fig.suptitle(f"{args.sample} data {tracer}")
        fig.tight_layout()
        stem = os.path.join(figdir, f"{args.sample}_{tracer}")
        fig.savefig(stem + ".png", dpi=130)
        np.savez(stem + ".npz", ell_edges=EDGES, **{f"cgg_{k}": v for k, v in cgg.items()},
                 **{f"cgk_{L}_{k}": v for L in LENSMAPS for k, v in cgk[L].items()},
                 **{f"sigma_{L}": v for L, v in sig.items()})
        with open(stem + ".txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
