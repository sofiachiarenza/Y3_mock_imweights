"""Euclid-style dust/CIB test on the REAL catalogs (CLAUDE.md task 3).

Per tracer, reading the data sidecar and randoms cache (regress_{photo,spec}_data.py):
  r_ell(delta_g, X) for X = SFD, CSFD, D = CSFD - SFD, EBV_DIFF_GR, EBV_DIFF_RZ and each
  Lenz+19 CIB map, plus the prediction through the CIB, r(delta_g, CIB) r(CIB, X).
delta_g is built two ways:
  * raw   = W_ANALYSIS only, with no imaging weight. This is the actual dust correlation of the sample.
  * imlin = W_ANALYSIS x WEIGHT_IMLIN, i.e. what remains after the baseline regression.
Footprint: the data mask (randoms >= 20% of the median) & the CIB mask. The EBV_DIFF rows
also require EBV_DIFF to be defined (the ~-98 sentinels mark pixels with no DESI stars).
The CIB auto in r(., CIB) is the odd x even half-ring cross, which is noise-free, as in the
Euclid CMBX test (Daley, KP2-3, 22 Oct 2026, slide 13). The noise-biased full-mission
prediction is kept as '... via CIB fullmission' for comparison.
Reuses cib_check (CIB loading, rho, rho_cib) and dust_map_check (BINS).

    python euclid_data_check.py --sample photo --tracers BGS LRG1 LRG2 LRG3
    python euclid_data_check.py --sample spec --tracers BGS LRG1 LRG2 LRG3 QSO QSO1 QSO2
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from cib_check import cib_auto, cib_maps, load_cib, rho, rho_cib
from common import load_config, radec_to_pix, read_data_sidecar
from compare_data_variants import data_paths, delta_map, randoms_map
from dust_map_check import BINS
from produce_imweights import get_debv


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", required=True, choices=["photo", "spec"])
    p.add_argument("--tracers", nargs="+", required=True)
    args = p.parse_args()
    cfg = load_config()
    figdir = cfg["paths"]["figures"]

    t = Table.read(cfg["paths"]["photo_templates"])
    X = {c: np.asarray(t[c], float) for c in ("SFD", "CSFD")}
    X["D"] = X["CSFD"] - X["SFD"]
    debv = get_debv(cfg["paths"]["debv_map"])
    for c in ("EBV_DIFF_GR", "EBV_DIFF_RZ"):
        X[c] = np.asarray(debv[c], float)
    debv_ok = (X["EBV_DIFF_GR"] > -1) & (X["EBV_DIFF_RZ"] > -1)
    paths, cache_dir = cib_maps(cfg)
    cibs = [load_cib(pth, cache_dir) for pth in paths]
    ell_c = 0.5 * (BINS[:-1] + BINS[1:])
    hdr = f"{'':>30s}" + "".join(f"{f'{lo}-{hi}':>9s}" for lo, hi in zip(BINS[:-1], BINS[1:]))

    for tracer in args.tracers:
        side, cache = data_paths(cfg, args.sample, tracer)
        if not os.path.exists(side):
            print(f"{tracer}: no sidecar {side}")
            continue
        ws, ex = read_data_sidecar(side)
        pix = radec_to_pix(ex["RA"], ex["DEC"])
        R = randoms_map(cache)
        mask = R >= 0.2 * np.median(R[R > 0])
        dg = {"raw": delta_map(pix, ex["W_ANALYSIS"], R, mask),
              "imlin": delta_map(pix, ex["W_ANALYSIS"] * ws["WEIGHT_IMLIN"], R, mask)}
        out = [f"# {args.sample} data {tracer}: N={len(pix)}, fsky(mask)={mask.mean():.4f}. D = CSFD - SFD."]
        fig, axes = plt.subplots(len(cibs), 2, figsize=(12, 3.8 * len(cibs)), squeeze=False)
        for (label, cmaps, cmask), row in zip(cibs, axes):
            cib = cmaps["full"]
            reg = mask & cmask
            out.append(f"\n==== CIB {label}: npix={reg.sum()} fsky={reg.mean():.4f}")
            out.append(hdr)
            for ax, (dname, d) in zip(row, dg.items()):
                r_gc = rho_cib(d, cmaps, reg)
                out.append(f"{f'r(g_{dname}, CIB)':>30s}" + "".join(f"{v:9.3f}" for v in r_gc))
                for x, xm in X.items():
                    rx = reg & debv_ok if x.startswith("EBV_DIFF") else reg
                    if rx.sum() < 1000:
                        continue
                    auto = cib_auto(cmaps, rx)
                    meas = rho(d, xm, rx)
                    pred = rho_cib(d, cmaps, rx, auto) * rho_cib(xm, cmaps, rx, auto)
                    pred_fm = rho(d, cib, rx) * rho(cib, xm, rx)
                    out.append(f"{f'r(g_{dname}, {x}) measured':>30s}" + "".join(f"{v:9.3f}" for v in meas))
                    out.append(f"{f'r(g_{dname}, {x}) via CIB':>30s}" + "".join(f"{v:9.3f}" for v in pred))
                    out.append(f"{f'r(g_{dname}, {x}) via CIB fullmission':>30s}" + "".join(f"{v:9.3f}" for v in pred_fm))
                    if x in ("D", "SFD", "EBV_DIFF_GR"):
                        line, = ax.plot(ell_c, meas, "o", ms=3, label=f"{x} measured")
                        ax.plot(ell_c, pred, "-", color=line.get_color(), label=f"{x} via CIB")
                        ax.plot(ell_c, pred_fm, ":", lw=1, color=line.get_color(), label=f"{x} via CIB, full-mission auto")
                sig = 1 / np.sqrt((2 * ell_c + 1) * np.diff(BINS) * reg.mean())
                ax.fill_between(ell_c, -sig, sig, color="0.9")
                ax.axhline(0, color="k", lw=0.5)
                ax.set_xscale("log")
                ax.set_title(f"{tracer} g_{dname}, CIB {label}")
                ax.legend(fontsize=6)
        for ax in axes[-1]:
            ax.set_xlabel(r"$\ell$")
        fig.tight_layout()
        stem = os.path.join(figdir, f"euclid_data_{args.sample}_{tracer}")
        fig.savefig(stem + ".png", dpi=110)
        with open(stem + ".txt", "w") as f:
            f.write("\n".join(out) + "\n")
        print("\n".join(out))


if __name__ == "__main__":
    main()
