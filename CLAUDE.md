# Y3_mock_imweights: working notes

`README.md` covers what this repo does, the map choices and every result so far. This file is the hand-off: the state as of 2026-10-26 and what is left. The repo now runs on NERSC (Perlmutter).

## NERSC setup (done 2026-10-26)
- Clone: `/global/cfs/cdirs/desicollab/users/schiarenza/Y3_mock_imweights`. `config.yaml` points at the NERSC paths.
  - mocks: Gerrit's `/pscratch/sd/g/gfarren/desi_y3_x_cmb_data/mock_catalogs/{spec/2026-07-14,photo/2026-09-29}`
  - maps: DA2 hpmaps; the EBV_DIFF and sky-residual maps from Rongpu's dirs, now their own `paths` keys
  - κ: Gerrit's PR4 scratch copy and the ACT DR6 public baseline
- **Env:** `source slurm/_env.sh` (conda `c3y3_regr` + upstream `DESI_Y3_x_CMB` on PYTHONPATH). The copied `env/` is a dead Narval venv, so don't use it and don't run `create_env.sh`. SysNet uses `source slurm/_env_sysnet.sh` (cosmodesi main + `amry3_env` + `LSS/py`).
- **Accounts:** CPU hours are short, so every job is charged to `desi_g` on GPU nodes.
  - whole-node jobs: `--exclusive`, several processes per node
  - small jobs: `-q shared --gpus=1` (a quarter node)

## Status (2026-10-26)

Done, all on Narval:
- Linear imaging regression on 100 mocks (HDU 1) for photo (BGS, LRG1-3) and spec (BGS, LRG1-3, QSO, QSO1/2), both lensmaps (act_dr6 sims 1-100, planck_pr4 sims 260-359).
- All dust variants, including the spec `CSFDHI`/`NOHI` added 2026-10-22, with the 100-mock comparisons in `figures/variants/`.
- Map-level dust tests (`dust_map_check.py`) and test 2 at 857 GHz (`cib_check.py`).
- `outputs/` (weight sidecars + randoms caches, ~46 GB) and `data/` (templates, CSFD, HI4PI, Lenz CIB) are git-ignored. They went to NERSC by Globus on 2026-10-26: Narval `/lustre06/project/6033532/schiaren/Y3_mock_imweights/` → `/global/cfs/cdirs/desicollab/users/schiarenza/Y3_mock_imweights/` (task 47ca1100-d147-11f1-b253-0effda64b40f). `env/` was not transferred.

**Done on NERSC (2026-10-26):** tasks 1–4 on the real catalogs; results are in README → Real-data tests.
- The photo baseline is safe. CSFD+SFD+HI wipes out 46–73% of Cgκ.
- On spec, EBV_DIFF costs ~1% of Cgκ.
- Spec BGS switched to the DESI 3-map default (STARDENS, GALDEPTH_R, HI). The old 5-map list is now the variant `WEIGHT_IMLIN_ALLEBV`.
- Pending:
  - (done) BGS act_dr6 100-mock rerun with the 3-map default → `figures/variants/spec_act_dr6_BGS.txt`
  - ELG SysNet array (60318577) → ELG mode removal (60318589)

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

**Driver written: `regress_photo_data.py` (2026-10-26).** It reuses `regress_photo_mocks.regress_catalog`, the upstream `get_photo_sample_obj` and 4 randoms files. The CSFD+SFD+HI diagnostic is the `data_only` variant `WEIGHT_IMLIN_CSFDSFDHI`. `--mock LENSMAP SIM` is the validation. Background: `Y3_AMR_nersc/regress_photo_samples.py` uses the old setup: the old CSFD from pixweight, only GALDEPTH extinction-corrected, and a CSFD-only map list. Write a driver in this repo (e.g. `regress_photo_data.py`) that reuses `common.py` and `regress_photo_mocks.py`:
- the same template file, map lists, `*DEPTH*` SFD extinction correction, N/S/DES regions and variants loop
- the real catalog + randoms in place of the mock + compressed-randoms cache. The compression is exact for data too, since templates are constant per pixel.
- sidecar output, never modifying the catalogs

Validate it first by running it on one mock and comparing with the existing sidecar in `outputs/photo/`.

### 2. The same test on the real spec catalogs
Fit the four spec variants on the real catalogs: `WEIGHT_IMLIN` (DESI default), `_CSFD` (EBV_DIFF referenced to CSFD), `_CSFDHI`, `_NOHI`. Compare Cgκ. This says whether the DESI default's EBV_DIFF, which contains −(CIB part of SFD), has the same issue. **Driver written: `regress_spec_data.py`** (upstream `NonSavableSpecSample`, 4 randoms per cap). It also stores the catalog's official weight as `WEIGHT_IMLIN_DESI` and logs its agreement with the refit default. Cgg and Cgκ variant ratios come from `compare_data_variants.py`.

### 3. Euclid-style test on data
For each tracer, compute r_ℓ(δ_g, X) for X = SFD, CSFD, CSFD − SFD, EBV_DIFF_GR/RZ and the Lenz CIB. Overlay the prediction through the CIB, r(δ_g, CIB)·r(CIB, X). **Written: `euclid_data_check.py`.** It uses δ_g with and without WEIGHT_IMLIN, on the data mask ∩ each CIB mask.

### 4. Finish test 2 (CIB frequencies)
**Maps in place (2026-10-26).** The Dataverse API works from NERSC: `curl -L https://dataverse.harvard.edu/api/access/datafile/<id>`. 545 (id 3478465) and 353 (3478493) `2.5e+20_gp20`, and 857 `4.0e+20_gp40` (3478474), are unpacked into `data/ext/lenz19_cib/<freq>/<mask>/`. The 6 KB error pages are deleted. `cib_check.py` runs inside `slurm/data_regress.sbatch`.
- Optional: use the 857 GHz half-mission maps (already downloaded) to estimate how much the CIB map's noise dilutes the predicted r(D, CIB)·r(CIB, κ). That tells whether the ~2× excess at ℓ ≥ 90 is real.

### 5. ELG SysNet on the new mocks
100 mocks, HDU 1: ACT sims 1-100, Planck sims 260-359. Scripts in `Y3_AMR_nersc/act_scripts/` (`prepare_sysnet_ACT.py`, `run_sysnet.sh`, `add_sysnet_ACT.py`). Before running:
- **Not a bug (corrected 2026-10-26):** the RING `hp.ang2pix(256, dth, dphi)` in `add_sysnet_ACT.py:122` is right.
  - `prep4sysnet` reorders the maps nest→ring and pixelises in RING (`LSS/py/LSS/imaging/sysnet_tools.py:128-145`), so `nn-weights.fits` `hpix` is RING.
  - Official `mkCat_main.py:1314` does the same.
  - `nest=True` would misplace the weights.
- Real issues, fixed in the port: `${tracer}` and `$srun_flags` were undefined, `srun -t 60` exceeded the job limit, the hard-coded 4-HDU in-place rewrite crashed, and the mocks are read-only.
- **Ported to `sysnet/`:** `prepare_sysnet_mocks.py`, `run_sysnet.sh`, `add_sysnet_mocks.py` and `sysnet_app_snapshot_mpi.py`.
  - Same maps, hyperparameters, snapshot ensemble (T_0=5) and mock-randoms fracgood as Y3_AMR.
  - The new mocks cover only z 1.1–1.6, so there is a single z bin.
  - Prep and training go to `/pscratch/sd/s/schiaren/y3imw_sysnet/`. The weights go to the sidecar column `WEIGHT_SN_MOCK`. The mocks' own `WEIGHT_SN` is a copy of `WEIGHT_SYS`; ignore it.
- Run: `slurm/sysnet_elg.sbatch`, a 20-task array of 10 sims per GPU node. Set `LANES`, `NTASKS` and `GPU` from the pilot timing. Then `slurm/elg_compare.sbatch` gives the mode-removal ratio.

### 6. Cℓ measurement (Gerrit)
Gerrit measures the analysis Cℓ from the sidecars in `outputs/` on the correct grid. The `figures/variants/` pseudo-Cℓ are a diagnostic only. Point him to the NERSC copy; the sidecar format is in `README.md` → Outputs.

## Conventions
- Run on compute nodes only. NERSC: `desi_g` GPU nodes. Sofia submits the jobs.
- **Run order:**
  1. `sbatch slurm/pilot_gpu.sbatch` (debug, 30 min): photo and spec `--mock` validation, plus ELG sim 1 trained on CPU and on GPU.
  2. `sbatch slurm/data_regress.sbatch`: tests 1–4.
  3. `sbatch slurm/sysnet_elg.sbatch`.
  4. `slurm/elg_compare.sbatch`.
- Never modify the mock or data catalogs; write sidecars.
- Depth maps stay extinction-corrected with SFD in every variant. That matches how the targets were selected (fluxes dereddened with SFD), so it's a consistency requirement, not a choice to vary.
- New results go into `README.md`; update this file's status and to-do as tasks are done.
