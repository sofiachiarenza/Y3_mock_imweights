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

**Spec:** the DESI default lists. LRG and QSO include EBV_DIFF_GR/RZ and HI, which are defined only on the DESI spec footprint.
- **BGS is STARDENS, GALDEPTH_R, HI** (LSS `globals.py` default; changed 2026-10-26).
  - The Y3 `lin_regr_*.py` scripts added EBV_DIFF_GR/RZ to BGS. That is the non-default LSS `allebv` list.
  - On data it does not reproduce the official `WEIGHT_IMLIN_CLUS`: corr 0.48, versus 0.9985 for the 3-map list.
  - It is kept as the BGS-only variant `WEIGHT_IMLIN_ALLEBV`.
  - The old 5-map mock sidecars are in `outputs/spec/act_dr6/BGS_allebv5map_20261022/`.

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

## Full 100-mock results (`figures/variants/<sample>_<lensmap>_<T>.txt`)

These are shot-noise-subtracted pseudo-Cℓ at nside 256 over the full mock footprint. They are a diagnostic for comparing variants, not the analysis bandpowers.

- **Dust choice is negligible.** Photo SFD+HI vs CSFD+HI, and spec CSFD-referenced EBV_DIFF vs the default, change ⟨Cgg⟩ by ≤ 0.05% at ℓ ≥ 10 and ≤ 0.4% at ℓ < 10, consistent with 1 within the errors. The rms weight difference is 5–11e-4.
- **Photo mode removal** (baseline/unweighted, ℓ = 2–5 | 6–9 | 10–14): BGS 0.79–0.80 | 0.91 | 0.96; LRG1–3 0.80–0.87 | 0.90–0.93 | 0.96–0.98. It is below 1% for ℓ ≳ 30.
- **Spec mode removal** is much milder: LRG 0.97–1.00 at ℓ < 20; QSO/QSO1/QSO2 within about 0.5–1% of 1.
- **Spec BGS with the DESI 3-map default** (rerun 2026-10-26, act_dr6, 100 mocks): baseline/unweighted is 0.989 ± 0.004 (ℓ = 2–5), 0.970 ± 0.005 (6–9), 0.986 ± 0.002 (10–14) and 0.989 ± 0.002 (15–19), and ≤ 0.3% at ℓ ≥ 20. The old 5-map list (now `WEIGHT_IMLIN_ALLEBV`) removed 0.5–1.4% more at ℓ < 20 (ALLEBV/baseline 0.986–0.997).
- **CSFD-only vs the CSFD+HI baseline:** up to about 3% differences at ℓ < 10 (HI removes a little more power). **Spec-like BGS** removes about 13% less power at ℓ = 2–5 than the 9-map photo default.
- **Spec dust-block variants** (added 2026-10-22, appended to the existing sidecars):

  | column | dust maps |
  |---|---|
  | `WEIGHT_IMLIN_CSFDHI` | CSFD + HI in place of EBV_DIFF_GR/RZ, the photo-style dust block (like-for-like with photo) |
  | `WEIGHT_IMLIN_NOHI` | the DESI default without HI |

  - **CSFD+HI vs the default** (mean over 100 mocks; σ is the error on that mean): ⟨Cgg⟩ is 0.5–1.5% *higher* for LRG and BGS at ℓ < 30. The largest shift is act_dr6 LRG2 at ℓ = 10–14 (1.0153 ± 0.0030). For ℓ ≥ 30 the change is ≤ 0.5%. QSO and LRG3 change by ≤ 0.6%. The weights differ by 6–10e-3 rms, ten times more than in the SFD↔CSFD swap. On mocks, EBV_DIFF therefore removes slightly more power than a directly fitted CSFD.
  - **No HI vs the default:** ≤ 1% at ℓ < 30 (largest: act_dr6 BGS at ℓ = 6–9, 1.010 ± 0.003) and ≤ 0.13% at ℓ ≥ 30. The weights differ by 4–7e-3 rms.
  - Both shifts are small next to the mode-removal correction itself, but unlike the SFD↔CSFD swap they are not negligible at ℓ < 30.

## Test 2: CSFD − SFD vs an independent CIB (`cib_check.py`, `figures/cib_check.{txt,png}`)

This checks CSFD − SFD against the Lenz, Doré & Lagache (2019) CIB, which unlike the CSFD LSS map is not reconstructed from galaxies. It has only been run at 857 GHz on the N_HI < 2.5e20 mask, over the photo footprint ∩ CSFD cosmology mask (fsky 0.14).
- **r(CSFD − SFD, CIB)** is about −0.10 at ℓ < 32, −0.34 to −0.39 at ℓ = 32–92, and −0.50 to −0.58 at ℓ ≥ 92, so CSFD − SFD is CIB-dominated.
  - The HI control gives r ≈ 0.01–0.09, as expected, since the Lenz CIB is HI-cleaned.
- **r with the CIB at ℓ ≳ 90:** SFD 0.22–0.28, CSFD 0.15–0.19. CSFD removes about a third of SFD's CIB correlation.
- **Euclid-style test:** the measured r(CSFD − SFD, κ) is 1.5–2.5× the CIB-route prediction r(D, CIB)·r(CIB, κ) at ℓ ≥ 90, and far larger at ℓ < 60. This fits with the CSFD correction itself containing galaxy-traced structure. Noise in the CIB map dilutes the prediction, so the low-ℓ numbers shouldn't be over-interpreted.
- **All frequencies (2026-10-26):** 353, 545 and 857 GHz on the 2.5e20 mask, plus 857 GHz on the 4.0e20 mask.
  - r(D, CIB) at ℓ ≥ 92 is −0.38 to −0.44 (353 GHz), −0.43 to −0.47 (545 GHz), −0.49 to −0.57 (857 GHz, 2.5e20 mask), and −0.42 to −0.48 (857 GHz, 4.0e20 mask).
  - The measured r(D, κ) is the same for every map, while the CIB-route prediction stays 1.5–3× smaller at ℓ ≥ 92 for all four. This holds with the noise-free odd × even CIB auto; `CIB auto S/(S+N)` in `cib_check.txt` gives the signal fraction per ℓ.
  - So the excess does not depend on the frequency or the mask.

## Real-data tests (2026-10-26; `regress_{photo,spec}_data.py`, `compare_data_variants.py`, `euclid_data_check.py`)

Inputs and method:
- Real catalogs come from the upstream DESI_Y3_x_CMB loaders, with 4 randoms files (per cap for spec) and the same fit as on the mocks.
- Cℓ are nside-256 pseudo-Cℓ, all variants on the same mask. Ratios and shifts are relative to the baseline `WEIGHT_IMLIN`; σ is the per-bin Knox error on Cgκ.
- These are diagnostics, not the analysis bandpowers. Results are in `figures/data_variants/<sample>_<T>.txt` and `figures/euclid_data_<sample>_<T>.txt`.
- **Validation:**
  - On one mock, the data drivers reproduce the Narval sidecars: photo max|Δw| = 3e-5, spec 4e-4. The spec value comes from MIGRAD stopping differently in one fit (N, z 0.5–0.6); the other fits agree to ~1e-7.
  - Refit default vs the official catalog weight: LRG1–3 and QSO corr 0.994–0.996, BGS (3 maps) 0.9985.
  - QSO1 (0.96) and QSO2 (0.86) are lower, as expected: their official weight is the full-QSO fit on the QSO z bins.

**Test 1, photo dust choice: the baseline is safe; fitting SFD and CSFD together is not.**
- The baseline (CSFD + HI) agrees with SFD + HI and with CSFD only: Cgκ(ℓ ≥ 30) within 0.5% for BGS and LRG1–3, at most 0.07σ per bin, for both ACT and Planck. Cgg shifts by < 0.5% at ℓ ≥ 10.
- **The CSFD + SFD + HI diagnostic removes 46–73% of Cgκ at ℓ ≥ 30**, up to 6–8σ per bin (BGS 8.2σ, LRG1 7.4σ), and 5–20% of Cgg.
  - On data, this fit spans the CIB direction (≈ 9 × (CSFD − SFD)), which correlates with κ, exactly as feared. Never use it.

**Test 2, spec: EBV_DIFF removes ~1% of Cgκ, below the noise per bin.**
- EBV_DIFF referenced to CSFD (`_CSFD`) raises Cgκ(ℓ ≥ 30) by 0.6–1.3% (ACT) and 0.1–0.7% (Planck) for LRG1–3 and QSO.
- The photo-style dust block (`_CSFDHI`) raises QSO Cgκ by 0.8–1.9%.
- BGS + EBV_DIFF (`_ALLEBV`) lowers Cgκ by 3%.
- The sign is the same everywhere, which suggests the DESI EBV_DIFF (which contains −(SFD's CIB)) takes out ~1% of the κ correlation. Every shift is ≤ 0.2σ per bin.
- Dropping HI (`_NOHI`) changes Cgκ by ≤ 0.5%.

**Test 3, Euclid-style (C. Daley, Euclid CMBX KP2-3, 22 Oct 2026, slide 13): galaxy × extinction is the CIB in the template; CSFD removes it.**
- Test: if a template X = dust + α·CIB and the dust does not correlate with the tracer, then r(g, X) = r(g, CIB)·r(CIB, X) band by band, with nothing fitted. The CIB auto is the noise-free odd × even half-ring cross, as in Euclid. The full-mission auto is ≥ 99% signal at 857 GHz for ℓ < 768, so this changes the prediction by < 1% there.
- **SFD is the analog of Euclid's dust-emission E(B−V) template.** At 857 GHz, ℓ ≥ 92, raw δ_g:
  - All six LRG samples: measured/predicted 0.85–1.03 (inverse-variance projection), χ²/n 0.4–0.7.
  - The ratio rises to 1.5–1.9 at 545 GHz and 1.7–2.2 at 353 GHz, the same trend as Euclid's (δ₅: 1.26 → 1.63 → 1.89).
  - BGS: ≈ 2× on the 2.5e20 mask but 0.85–1.07 on the 857 GHz 4e20 mask, so not robust.
  - QSO: 0.5–0.8.
- **CSFD:** r(g, CSFD) ≈ 0 for every sample, although r(CSFD, CIB) ≈ 0.17. The CIB that CSFD leaves in does not trace the galaxies. κ × SFD is small (r ≈ 0.02–0.1) and at or below its CIB route; κ × CSFD ≈ 0.
- **D = CSFD − SFD** is the galaxy-traced CIB that CSFD removed (Chiang 2023 reconstructs it from galaxy cross-correlations). Its correct null is not the CIB route alone.
  - Two reference models: the CIB route r(g, c)·r(c, D) (g sees the CIB as a whole) and the projection r(g, c)/r(D, c) (g sees only the galaxy-traced CIB), larger by 1/r(D, c)² ≈ 4. They are reference points, not strict bounds.
  - Measured r(g, D) ≈ −0.2 to −0.28 sits at (measured − route)/(projection − route): BGS 1.4–1.6 (beyond the projection: D also contains low-z density structure that BGS traces directly), LRG1 0.9–1.0, LRG2 0.8, LRG3 0.5, QSO 0–0.2, falling with redshift as expected for a low-z galaxy reconstruction.
  - r(D, κ) is 1.5–3× its CIB route, between the two models.
- Summary notebook: `TESTS_SUMMARY.ipynb`, organised as A (for Euclid) and B (DESI weights are safe).

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
