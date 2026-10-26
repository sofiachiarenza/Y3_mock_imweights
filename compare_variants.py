"""Compare imaging-weight variants on the mocks (pilot or full set).

Per tracer, over the requested sims, from the sidecars:
  * rms(w_variant - w_baseline) over weighted objects
  * mean shot-noise-subtracted pseudo-C_ell^gg (nside 256, fsky-corrected, fixed mask = pixels with
    randoms >= 20% of the median) for the unweighted catalog and every variant
  * binned ratios <C_ell(variant)>/<C_ell(baseline)> and
    <C_ell(baseline)>/<C_ell(unweighted)> (the mode-removal effect), with the
    scatter of the per-mock ratio / sqrt(N_sims) as the error on the mean ratio.
Spec objects carry WEIGHT*FKP/SYS/ZFAIL and the zbins cut, matching the randoms cache; photo
objects and randoms are unweighted. The variants then differ only in the imaging weight.

    python compare_variants.py --sample photo --lensmap act_dr6 --tracers LRG1 --sim-start 1 --sim-stop 5
"""
import argparse
import os

import h5py
import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from common import (NPIX, NSIDE, load_config, load_randoms_cache, mock_path, radec_to_pix, sidecar_path, sim_ids,
                    spec_random_weight)

LMAX = 3 * NSIDE - 1
EDGES = np.array([2, 6, 10, 15, 20, 30, 45, 60, 90, 130, 200, 300, 450, 600, LMAX + 1])


def binned(cl):
    ell = np.arange(len(cl))
    return np.array([cl[(ell >= lo) & (ell < hi)].mean() for lo, hi in zip(EDGES[:-1], EDGES[1:])])


def cl_of(pix, w, R, mask):
    """Shot-noise-subtracted pseudo-C_ell / fsky of delta = n_w / (alpha R) - 1.
    Shot noise of a weighted count map: N = Omega_pix * <sum_i w_i^2 / nbar_p^2>_mask."""
    n = np.bincount(pix, weights=w, minlength=NPIX)
    n2 = np.bincount(pix, weights=w * w, minlength=NPIX)
    alpha = n[mask].sum() / R[mask].sum()
    nbar = np.where(mask, alpha * R, 1.0)
    delta = np.where(mask, n / nbar - 1.0, 0.0)
    shot = 4 * np.pi / NPIX * np.mean(n2[mask] / nbar[mask] ** 2)
    return binned(hp.anafast(hp.reorder(delta, n2r=True), lmax=LMAX) / mask.mean() - shot)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", required=True, choices=["spec", "photo"])
    p.add_argument("--lensmap", required=True)
    p.add_argument("--tracers", nargs="+", required=True)
    p.add_argument("--sim-start", type=int)
    p.add_argument("--sim-stop", type=int)
    p.add_argument("--hdu", type=int)
    p.add_argument("--output-root", default=None, help="sidecar root override")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    cfg = load_config()
    if args.output_root:
        cfg["paths"]["output_root"] = args.output_root
    hdu = args.hdu or cfg["hdu"]
    figdir = os.path.join(cfg["paths"]["figures"], "variants")
    os.makedirs(figdir, exist_ok=True)
    ell_c = 0.5 * (EDGES[:-1] + EDGES[1:] - 1)

    for tracer in args.tracers:
        cache = load_randoms_cache(cfg, args.sample, tracer)
        R = np.bincount(np.asarray(cache["PIX"]), weights=np.asarray(cache["W"], float), minlength=NPIX)
        mask = R >= 0.2 * np.median(R[R > 0])
        cls, dw = {}, {}
        sims = [s for s in sim_ids(cfg, args.lensmap, args.sim_start, args.sim_stop)
                if os.path.exists(sidecar_path(cfg, args.sample, args.lensmap, tracer, s, hdu))]
        for sim in sims:
            with fits.open(mock_path(cfg, args.sample, args.lensmap, tracer, sim), memmap=True) as h:
                d = h[hdu].data
                pix = radec_to_pix(d["RA"], d["DEC"])
                if args.sample == "spec":
                    # the spec randoms cache is weighted WEIGHT*FKP/SYS/ZFAIL and covers only the zbins,
                    # so the objects get the same weight and z range (unit weights gave a spurious
                    # large-scale power from the completeness/FKP pattern: 100-300x at l < 10 for QSO)
                    zb = cfg["spec"]["tracers"][tracer]["zbins"]
                    wobj = np.asarray(spec_random_weight(d), float)
                    sel = (d["Z"] >= zb[0][0]) & (d["Z"] < zb[-1][1])
                else:
                    wobj, sel = np.ones(len(pix)), np.ones(len(pix), bool)
            with h5py.File(sidecar_path(cfg, args.sample, args.lensmap, tracer, sim, hdu), "r") as f:
                cols = [k for k in f.keys() if k.startswith("WEIGHT")]
                ws = {k: f[k][:] for k in cols}
            pix, wobj = pix[sel], wobj[sel]
            ws = {k: w[sel] for k, w in ws.items()}
            cls.setdefault("UNWEIGHTED", []).append(cl_of(pix, wobj, R, mask))
            for k, w in ws.items():
                cls.setdefault(k, []).append(cl_of(pix, wobj * w, R, mask))
                if "WEIGHT_IMLIN" in ws and k != "WEIGHT_IMLIN":
                    sel = ws["WEIGHT_IMLIN"] != 1
                    dw.setdefault(k, []).append(np.std(w[sel] - ws["WEIGHT_IMLIN"][sel]))
        if not sims:
            print(f"{tracer}: no sidecars found")
            continue
        cls = {k: np.array(v) for k, v in cls.items()}
        n = len(sims)
        bkey = "WEIGHT_IMLIN" if "WEIGHT_IMLIN" in cls else next(k for k in cls if k != "UNWEIGHTED")  # ELG: WEIGHT_SN_MOCK
        base = cls[bkey]
        lines = [f"# {args.sample} {args.lensmap} {tracer}: {n} sims, fsky(mask)={mask.mean():.4f}",
                 "# ell bins: " + " ".join(f"{lo}-{hi - 1}" for lo, hi in zip(EDGES[:-1], EDGES[1:]))]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for k, c in cls.items():
            num, den = (base, cls["UNWEIGHTED"]) if k == "UNWEIGHTED" else (c, base)
            label = "baseline/unweighted" if k == "UNWEIGHTED" else f"{k}/baseline"
            if k == bkey:
                continue
            r = num / den
            mean, err = r.mean(0), r.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(r.shape[1])
            lines.append(f"{label:40s} " + " ".join(f"{m:7.4f}±{e:.4f}" for m, e in zip(mean, err)))
            ax.errorbar(ell_c, mean, err, marker="o", ms=3, capsize=2, label=label)
        for k, v in dw.items():
            lines.append(f"rms(w_{k} - w_baseline) per mock: mean={np.mean(v):.2e}")
        ax.axhline(1, color="k", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$\langle C_\ell^{gg}\rangle$ ratio")
        ax.set_title(f"{args.sample} {args.lensmap} {tracer} ({n} sims)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        stem = os.path.join(figdir, f"{args.sample}_{args.lensmap}_{tracer}{args.tag}")
        fig.savefig(stem + ".png", dpi=130)
        np.savez(stem + ".npz", ell_edges=EDGES, sims=sims, **{f"cl_{k}": v for k, v in cls.items()})
        with open(stem + ".txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
