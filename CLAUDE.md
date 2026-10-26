# Y3_mock_imweights: working notes

`README.md` covers what this repo does, the map choices and every result so far. This file is the hand-off: the state as of 2026-10-26 and what is left, most of it on NERSC.

## Status (2026-10-26)

Done, all on Narval:
- Linear imaging regression on 100 mocks (HDU 1) for photo (BGS, LRG1-3) and spec (BGS, LRG1-3, QSO, QSO1/2), both lensmaps (act_dr6 sims 1-100, planck_pr4 sims 260-359).
- All dust variants, including the spec `CSFDHI`/`NOHI` added 2026-10-22, with the 100-mock comparisons in `figures/variants/`.
- Map-level dust tests (`dust_map_check.py`) and test 2 at 857 GHz (`cib_check.py`).
- `outputs/` (weight sidecars + randoms caches, ~46 GB) and `data/` (templates, CSFD, HI4PI, Lenz CIB) are git-ignored. They went to NERSC by Globus on 2026-10-26: Narval `/lustre06/project/6033532/schiaren/Y3_mock_imweights/` → `/global/cfs/cdirs/desicollab/users/schiarenza/Y3_mock_imweights/` (task 47ca1100-d147-11f1-b253-0effda64b40f). `env/` was not transferred.

The key open question: CSFD − SFD (the CIB removed by CSFD) correlates with CMB κ at r ≈ −0.5. Mocks can't show whether fitting dust maps removes κ-correlated signal, because mock galaxies don't correlate with the real sky. Only the real catalogs can answer it (tasks 1-3 below).

## NERSC to-do

### 0. Setup
- `git pull`, then `bash create_env.sh`.
- Point `config.yaml` at the NERSC paths. Every path in it is a Narval path now: spec/photo mock and randoms roots (gfarren's directories), DA2 `hpmaps`, `pixweight-1-dark.fits`, the Planck PR4 and ACT DR6 κ alms and masks.
- `data/photo_templates_nside256_nest.fits` and `data/ext/` came with the transfer, so `build_dust_templates.py` does not need rerunning. Check the file is there.
- Quick check that the env works: `python dust_map_check.py` on a compute node should reproduce `figures/dust_map_check.txt`.

### 1. Dust test on the real photo catalogs (the decisive test)
Regress the real photo samples (BGS, LRG1-3) with the same variants as the mocks, then compare the weights, Cgg and Cgκ (ACT and Planck):
- `WEIGHT_IMLIN` CSFD+HI (baseline)
- `_SFDHI` SFD+HI
- `_CSFDONLY` CSFD
- CSFD+SFD+HI, optional, to see the CIB direction act on real data. It is degenerate on mocks and was dropped there; on data it is a diagnostic only.

A Cgκ shift between variants is what would show the CIB/κ effect matters.

**Missing: a data-regression driver.** `Y3_AMR_nersc/regress_photo_samples.py` uses the old setup: the old CSFD from pixweight, only GALDEPTH extinction-corrected, and a CSFD-only map list. Write a driver in this repo (e.g. `regress_photo_data.py`) that reuses `common.py` and `regress_photo_mocks.py`:
- the same template file, map lists, `*DEPTH*` SFD extinction correction, N/S/DES regions and variants loop
- the real catalog + randoms in place of the mock + compressed-randoms cache. The compression is exact for data too, since templates are constant per pixel.
- sidecar output, never modifying the catalogs

Validate it first by running it on one mock and comparing with the existing sidecar in `outputs/photo/`.

### 2. The same test on the real spec catalogs
Fit the four spec variants on the real catalogs: `WEIGHT_IMLIN` (DESI default), `_CSFD` (EBV_DIFF referenced to CSFD), `_CSFDHI`, `_NOHI`. Compare Cgκ. This says whether the DESI default's EBV_DIFF, which contains −(CIB part of SFD), has the same issue. It needs a spec data driver too, the counterpart of `regress_spec_mocks.py`.

### 3. Euclid-style test on data
For each tracer, compute r_ℓ(δ_g, X) for X = SFD, CSFD, CSFD − SFD, EBV_DIFF_GR/RZ and the Lenz CIB. Overlay the prediction through the CIB, r(δ_g, CIB)·r(CIB, X). The pseudo-Cℓ / r_ℓ helpers in `dust_map_check.py` and `cib_check.py` (`pcl`, `binned`, `BINS`) can be reused.

### 4. Finish test 2 (CIB frequencies)
Download the Lenz+19 545 and 353 GHz maps (`2.5e+20_gp20`) and, optionally, the 857 GHz `4.0e+20_gp40` map, from https://doi.org/10.7910/DVN/8A1SR3, in a browser. On Narval the scripted downloads returned 6 KB XML error pages: `data/ext/lenz19_cib/archives/{353,545}_2.5e20_gp20.7z` and `857_4.0e20_gp40.7z` are not maps, so delete them. Unpack into `data/ext/lenz19_cib/<freq>/<mask>/` and rerun `cib_check.py`.
- Optional: use the 857 GHz half-mission maps (already downloaded) to estimate how much the CIB map's noise dilutes the predicted r(D, CIB)·r(CIB, κ). That tells whether the ~2× excess at ℓ ≥ 90 is real.

### 5. ELG SysNet on the new mocks
100 mocks, HDU 1: ACT sims 1-100, Planck sims 260-359. Scripts in `Y3_AMR_nersc/act_scripts/` (`prepare_sysnet_ACT.py`, `run_sysnet.sh`, `add_sysnet_ACT.py`). Before running:
- **Bug:** `add_sysnet_ACT.py:122` computes `dpix = hp.ang2pix(256, dth, dphi)` in RING ordering, while line 94 and the prep step use `nest=True`. The SysNet weights would land on the wrong pixels. Add `nest=True`.
- **Bug:** `run_sysnet.sh:36` uses `${tracer}`, which the script never defines.
- The scripts still point at the old `2026-01-27-sofia` mocks. Repoint them at the new spec mocks (`LogNormal_Mocks/{act_dr6,planck_pr4}/ELG/`).

### 6. Cℓ measurement (Gerrit)
Gerrit measures the analysis Cℓ from the sidecars in `outputs/` on the correct grid. The `figures/variants/` pseudo-Cℓ are a diagnostic only. Point him to the NERSC copy; the sidecar format is in `README.md` → Outputs.

## Conventions
- Run on compute nodes only (Narval: `srun`/`sbatch`, account `rrg-wperciva`).
- Never modify the mock or data catalogs; write sidecars.
- Depth maps stay extinction-corrected with SFD in every variant. That matches how the targets were selected (fluxes dereddened with SFD), so it's a consistency requirement, not a choice to vary.
- New results go into `README.md`; update this file's status and to-do as tasks are done.
