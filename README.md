# Y3_mock_imweights

Linear imaging-systematics regression (alaeboss "linear" method) on the DESI Y3 × CMB lensing lognormal mocks, for the photometric samples (BGS, LRG1-3) and the spectroscopic samples (BGS, LRG1-3, QSO, QSO1/2). The mocks contain no injected systematics. The weights fitted on them measure the mode removal the regression causes, i.e. how much it suppresses real clustering, so that it can be corrected in the data measurement.

ELG is not handled here; it uses SysNet, run at NERSC.

## Inputs (paths in `config.yaml`)

| | Narval path |
|---|---|
| spec mocks + randoms | `LogNormal_Mocks/{act_dr6,planck_pr4}/<T>/mock_simNNNN.fits`, `mock_randoms_<T>.h5` |
| photo mocks + randoms | `2026-09-29/{act_dr6,planck_pr4}/<T>/...` |
| spec templates (DESI DA2 hpmaps) | `Y3_AMR/hpmaps/` |
| photo imaging maps | `pixweight-1-dark.fits` (DR9) |
| dust / HI (downloaded, `data/ext/`) | CSFD release (Chiang 2023, Zenodo 10.5281/zenodo.8207175: `sfd_ebv`, `csfd_ebv`, `lss_intensity`, `mask`), HI4PI `NHI_HPX.fits.gz` (CDS J/A+A/594/A116) |

Sims: hdu 1, act_dr6 1–100, planck_pr4 260–359.

## Map choices

**Spec:** the DESI default lists from the Y3 `lin_regr_*.py` scripts. They include EBV_DIFF_GR/RZ and HI, which are defined only on the DESI spec footprint.

**Photo:** DESI has no default for these samples. The list mirrors spec as closely as possible:
- Imaging maps: GALDEPTH_G/R/Z, PSFDEPTH_W1 (LRG only), PSFSIZE_G/R/Z, STARDENS.
- Dust baseline: **CSFD + HI4PI**. EBV_DIFF can't be used over the photo footprint (no DESI stars outside the spec footprint), and the spec HI map is 0 outside the spec footprint, which is why HI is rebuilt from HI4PI.
- Every `*DEPTH*` map is extinction-corrected with SFD, the same convention as spec.
- Regions north, south and des are fitted separately. The 2026-09-29 mocks cover the full DR9 footprint.
- Variants fitted in the same pass:

  | column | dust maps |
  |---|---|
  | `WEIGHT_IMLIN` | CSFD + HI (baseline) |
  | `_SFDHI` | SFD + HI |
  | `_CSFDONLY` | CSFD (the previous photo choice) |
  | `_SPECLIKE` | BGS only: the spec BGS map list with CSFD |

## Dust findings (`dust_map_check.py`, `figures/dust_map_check.txt`)

- **Inputs check out.** The SFD from the CSFD release matches the pixweight EBV (correlation 0.9997). The HI4PI rebuild matches the spec `HI` (correlation 0.9999, ratio 1.000).
- **CSFD − SFD is small.** rms 0.0006 mag, about 2% of the SFD rms. Its correlation with any imaging map is below 0.03.
- **CSFD − SFD correlates with CMB lensing.** It is mostly the CIB removed from SFD, and it correlates with κ at r ≈ −0.5 (Planck PR4) and −0.6 (ACT DR6) for ℓ < 100, falling to about −0.2 at ℓ ~ 300.
  - A fit that spans this direction (SFD and CSFD together) can remove κ-correlated signal from the data.
  - The mocks can't detect this. It has to be tested on data, by comparing Cgκ across the dust variants.

## Photo pilot (5 act_dr6 mocks/tracer, `figures/variants/*_pilot.txt`)

- **SFD+HI vs CSFD+HI are identical on mocks.** Weight rms difference ≤ 8e-4. ⟨Cgg⟩ changes by < 0.1% for ℓ ≥ 10 and ≤ 0.7% at ℓ = 2–9. The SFD/CSFD choice is irrelevant for the mock mode-removal correction.
- **CSFD+SFD+HI is degenerate.** The two maps are 0.9998 correlated, and the fit returns coefficients of about ±10 that effectively fit about 9×(SFD − CSFD), i.e. the CIB. The weights then differ from the baseline by 3–6% rms and *add* 1–4% power at all ℓ. This variant was dropped and should never be used.
- **Mode removal by the baseline, ⟨C(baseline)⟩/⟨C(unweighted)⟩:** 0.75 (BGS) to 0.93 (LRG3) at ℓ = 2–5, 0.96–0.99 at ℓ = 10–19, and < 1% for ℓ ≥ 30.
- **Spec-like BGS** (4 maps: STARDENS, GALDEPTH_R, HI, CSFD) removes about 15% less power at ℓ = 2–5 than the 9-map default (which adds GALDEPTH_G/Z and PSFSIZE_G/R/Z).
- **Timing, all variants:** BGS about 11 min/mock, LRG 3–6 min/mock, 15 GB peak. Production runs as 10-sim array chunks (`slurm/regress_photo.sbatch`).

## Method notes

- **Compressed randoms.** `LinearRegressor` takes its outlier cuts and bin edges from the data only, and every template is constant within a pixel. So the randoms are reduced, with no approximation, to one row per nside-256 pixel (per z-bin for spec) carrying the summed weight: about 750M rows become about 360k.
- **Validated against the old code.** `tests/test_photo_vs_old_code.py` runs the original `Y3_AMR_nersc/regress_photo_mocks.py` logic on raw randoms; the weights are identical (max difference 0).

## Running

```bash
bash create_env.sh                                        # ./env
# one-off downloads into data/ext/ (see build_dust_templates.py docstring)
python build_dust_templates.py                            # data/photo_templates_nside256_nest.fits
python dust_map_check.py
python build_randoms_cache.py --sample photo --tracer LRG1
python regress_photo_mocks.py --tracer LRG1 --lensmap act_dr6 --sim-start 1 --sim-stop 100
python compare_variants.py --sample photo --lensmap act_dr6 --tracers LRG1
sbatch slurm/photo_pilot.sbatch 5                         # pilot: 5 mocks/tracer, all variants
sbatch slurm/regress_photo.sbatch                         # production: 4 tracers x 2 lensmaps x 100 sims
```

Run everything on compute nodes.

## Outputs

Sidecars are written to `outputs/<sample>/<lensmap>/<T>/mock_simNNNN_hdu1.h5`:
- one dataset per weight column, with the full HDU length and in HDU row order (1.0 where no fit applies)
- `coeffs/<variant>/<region or region_z>/`: the best-fit coefficients, plus `chi2_null`, `chi2_fit` and `n_dof`.

The mock catalogs themselves are never modified.
