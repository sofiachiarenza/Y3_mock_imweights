"""SysNet input tables for the ELG lognormal mocks (CLAUDE.md task 5).

A port of Y3_AMR/act_scripts/prepare_sysnet_ACT.py (and the Planck copy) to the new
2026-07-14 spec mocks. The prep itself is unchanged:
  * maps STARDENS, PSFSIZE_G/R/Z, GALDEPTH_G/R/Z, EBV_DIFF_GR/RZ, HI from the DA2
    ELG_LOPnotqso_mapprops_healpix_nested_nside256_{N,S}.fits, with the DEPTH maps
    extinction-corrected and EBV_DIFF from LSS.common_tools.get_debv()
  * z 1.1-1.6 (the new mocks cover only this range), regions N and S, PHOTSYS from the
    regressis DR9 footprint (S, then DES -> S, then N), the mock randoms read once, and
    prep4sysnet(wtmd="none", nran_exp=None, nest=True); it still multiplies by WEIGHT_ZFAIL
    when the column exists, as before
Changes from Y3_AMR:
  * flat mock paths (config spec_mock_root/<lensmap>/ELG/mock_simNNNN.fits)
  * the mocks are read-only: PHOTSYS is added in memory, and nothing is written back
  * prep files go to <workdir>/<lensmap>/ELG/mock_simNNNN/prep_1.1_1.6_{N,S}.fits

    python sysnet/prepare_sysnet_mocks.py --lensmap act_dr6 --sims 1 2 3
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Column, Table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import imaging_footprints, load_config, mock_path, radec_to_pix  # noqa: E402

import LSS.common_tools as common  # noqa: E402
from LSS.imaging import sysnet_tools  # noqa: E402

TRACER, SAMPLE_NAME = "ELG", "ELG_LOPnotqso"
ZMIN, ZMAX = 1.1, 1.6
FIT_MAPS = ["STARDENS", "PSFSIZE_G", "PSFSIZE_R", "PSFSIZE_Z", "GALDEPTH_G", "GALDEPTH_R", "GALDEPTH_Z",
            "EBV_DIFF_GR", "EBV_DIFF_RZ", "HI"]
WORKDIR = "/pscratch/sd/s/schiaren/y3imw_sysnet"


def add_photsys(t):
    """Y3_AMR ensure_photsys order: south -> S, des -> S, north -> N."""
    foot = imaging_footprints()
    pix = radec_to_pix(t["RA"], t["DEC"])
    ps = np.full(len(t), "", dtype="U1")
    ps[foot["south"][pix]] = "S"
    ps[foot["des"][pix]] = "S"
    ps[foot["north"][pix]] = "N"
    t.add_column(Column(ps, name="PHOTSYS"))
    return t


def mock_dir(workdir, lensmap, sim):
    return os.path.join(workdir, lensmap, TRACER, f"mock_sim{sim:04d}")


def load_mock_randoms(cfg):
    path = os.path.join(cfg["paths"]["spec_mock_root"], f"mock_randoms_{TRACER}.h5")
    with h5py.File(path, "r") as f:
        ds = f["__astropy_table__"]
        t = Table({c: np.asarray(ds.fields(c)[:], float) for c in ("RA", "DEC")})
    return add_photsys(t)


def sys_tables(cfg):
    debv = common.get_debv(cfg["paths"]["debv_map"])
    out = {}
    for reg in ("N", "S"):
        sys_tab = Table.read(os.path.join(cfg["paths"]["spec_hpmaps_dir"],
                                          f"{SAMPLE_NAME}_mapprops_healpix_nested_nside256_{reg}.fits"))
        for col in list(sys_tab.dtype.names):
            if "DEPTH" in col:
                bnd = col.split("_")[-1]
                sys_tab[col] *= 10 ** (-0.4 * common.ext_coeff[bnd] * sys_tab["EBV"])
        for ec in ("GR", "RZ"):
            sys_tab[f"EBV_DIFF_{ec}"] = debv[f"EBV_DIFF_{ec}"]
        out[reg] = sys_tab
    return out


def prepare(cfg, lensmap, sim, randoms, systab, workdir, hdu):
    path = mock_path(cfg, "spec", lensmap, TRACER, sim)
    if not os.path.exists(path):
        print(f"mock {sim:04d}: missing {path} (skipping)", flush=True)
        return
    with fits.open(path, memmap=True) as h:
        data = add_photsys(Table(h[hdu].data))
    outdir = mock_dir(workdir, lensmap, sim)
    os.makedirs(outdir, exist_ok=True)
    for reg in ("N", "S"):
        prep = sysnet_tools.prep4sysnet(
            data[data["PHOTSYS"] == reg], randoms[randoms["PHOTSYS"] == reg], systab[reg],
            zcolumn="Z", zmin=ZMIN, zmax=ZMAX, nran_exp=None, nside=256, nest=True,
            use_obiwan=False, columns=FIT_MAPS, wtmd="none")
        common.write_LSS_scratchcp(prep, os.path.join(outdir, f"prep_{ZMIN}_{ZMAX}_{reg}.fits"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lensmap", required=True, choices=["act_dr6", "planck_pr4"])
    p.add_argument("--sims", type=int, nargs="+", required=True)
    p.add_argument("--workdir", default=WORKDIR)
    args = p.parse_args()
    cfg = load_config()
    t0 = time.time()
    randoms = load_mock_randoms(cfg)
    systab = sys_tables(cfg)
    print(f"randoms loaded: {len(randoms):,} ({time.time() - t0:.0f}s)", flush=True)
    for sim in args.sims:
        t1 = time.time()
        prepare(cfg, args.lensmap, sim, randoms, systab, args.workdir, cfg["hdu"])
        print(f"mock {sim:04d}: prep done ({time.time() - t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
