"""Drive `LinearRegressor` (linear_regression.py) to fit imaging-systematics
weights for LRG/ELG clustering catalogs, following the DESI/eBOSS "à la
eBOSS" linear regression convention.

Adapted from github.com/domichbt/alaeboss's `produce_imweights()`
(src/alaeboss/produce_imweights.py, alaeboss v0.0.1), with three changes so
PNG_DR2 doesn't need `alaeboss`, `fitsio`, or the full `desihub/LSS` package
installed (see PNG_DR2/create_env_imaging_weights.sh for why):

1. Originally scoped to LRG/ELG clustering catalogs only (`photometric_regions`
   hardcoded to ["S", "N"] -- the upstream QSO-specific DES/SnotDES split was
   dropped, since `LSS.common_tools.select_regressis_DES` wasn't available
   here). Later generalized (optional `photometric_regions`/`region_row_masks`/
   `region_template_key` kwargs, all defaulting to the original N/S-by-PHOTSYS
   behavior byte-for-byte) so QSO's regression could reuse this function too,
   using `regions.select_regressis_des` (PNG_DR2's own `LSS`-free
   reimplementation, see scripts/regions.py) instead of `LSS.common_tools`.
2. The `ext_coeff` dict, `get_debv()`, and `mk_zcmbmap()`/`get_zcmbdipole()`
   are vendored below, copied from `desihub/LSS`'s `py/LSS/common_tools.py`
   (confirmed against the upstream source on GitHub), reimplemented with
   `astropy.table.Table.read` instead of `fitsio.read` so `fitsio` isn't
   needed either. `mk_zcmbmap` is fully analytic (a per-pixel projection
   onto the Planck-2018 CMB dipole direction) -- no map file involved.
3. All template/EBV/sky-residual paths are explicit function arguments
   (pointing at PNG_DR2/reference_data/hpmaps/ by default via config.py's
   IMAGING_TEMPLATES_DIR), not hardcoded absolute paths.
"""

import logging
from pathlib import Path
from time import time

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
from astropy.table import Table

from linear_regression import LinearRegressor

# The regression is fit in float64 (small overdensity residuals around 1.0
# need it); JAX defaults to float32 unless this is set. Must happen before
# any jax array is created -- alaeboss's own produce_imweights() and
# lin_regr_ACT.py both set this at import/start time for the same reason.
jax.config.update("jax_enable_x64", True)

ext_coeff = {"G": 3.214, "R": 2.165, "Z": 1.211, "W1": 0.184, "W2": 0.113}


def get_debv(mapname):
    """EBV_DIFF_GR/RZ = EBV_DESI - EBV_SFD, per color. Vendored from
    LSS.common_tools.get_debv, using astropy instead of fitsio."""
    ebvn = Table.read(mapname)
    debv = Table()
    for ec in ["gr", "rz"]:
        debv_a = np.asarray(ebvn["EBV_DESI_" + ec.upper()]) - np.asarray(ebvn["EBV_SFD_" + ec.upper()])
        debv_a = hp.reorder(debv_a, r2n=True)
        debv["EBV_DIFF_" + ec.upper()] = debv_a
    return debv


def get_zcmbdipole(ra, dec):
    """Redshift equivalent of the CMB dipole at (ra, dec) [deg]. Vendored
    from LSS.common_tools.get_zcmbdipole. Planck 2018 (arXiv:1807.06205,
    Table 3): v=369.82 km/s toward (l,b)=(264.021, 48.253) deg, i.e.
    (RA, Dec) = (167.942, -6.944) deg."""
    z_cmb = 369.82 / 2.9979e5
    ra_cmb, dec_cmb = 167.942, -6.944

    sin_ra_cmb, cos_ra_cmb = np.sin(np.pi * ra_cmb / 180), np.cos(np.pi * ra_cmb / 180)
    sin_dec_cmb, cos_dec_cmb = np.sin(np.pi * dec_cmb / 180), np.cos(np.pi * dec_cmb / 180)
    sin_ra, cos_ra = np.sin(np.pi * ra / 180), np.cos(np.pi * ra / 180)
    sin_dec, cos_dec = np.sin(np.pi * dec / 180), np.cos(np.pi * dec / 180)

    cos_angdis2cmb = cos_dec * cos_dec_cmb * (cos_ra * cos_ra_cmb + sin_ra * sin_ra_cmb) + sin_dec * sin_dec_cmb
    return z_cmb * cos_angdis2cmb


def mk_zcmbmap(nside=256, nest=True):
    """ZCMB template: get_zcmbdipole evaluated at every pixel center of an
    nside healpix grid. Vendored from LSS.common_tools.mk_zcmbmap."""
    pix = np.arange(12 * nside * nside)
    theta, phi = hp.pix2ang(nside, pix, nest=nest)
    ra = phi * 180 / np.pi
    dec = 90 - theta * 180 / np.pi
    return get_zcmbdipole(ra, dec)


def get_skyres(sky_res_map_path_north, sky_res_map_path_south):
    """Sky-background-residual templates (SKY_RES_G/R/Z), per photometric
    region. Adapted from alaeboss's own get_skyres(), using astropy instead
    of fitsio and explicit paths instead of hardcoded ones."""
    sky_g = {"N": np.zeros(256 * 256 * 12), "S": np.zeros(256 * 256 * 12)}
    sky_r = {"N": np.zeros(256 * 256 * 12), "S": np.zeros(256 * 256 * 12)}
    sky_z = {"N": np.zeros(256 * 256 * 12), "S": np.zeros(256 * 256 * 12)}
    for region, path in (("N", sky_res_map_path_north), ("S", sky_res_map_path_south)):
        f = Table.read(path)
        pix_nest = hp.ring2nest(256, np.asarray(f["HPXPIXEL"]))
        sky_g[region][pix_nest] = f["sky_median_g"]
        sky_r[region][pix_nest] = f["sky_median_r"]
        sky_z[region][pix_nest] = f["sky_median_z"]
    return sky_g, sky_r, sky_z


def _read_systematic_templates_stacked(ra, dec, sys_tab, use_maps, nside, nest):
    pixnums = hp.ang2pix(nside, np.radians(-dec + 90), np.radians(ra), nest=nest)
    systematics_values = sys_tab[use_maps][pixnums]
    return np.vstack([systematics_values[column_name] for column_name in systematics_values.colnames])


def make_fit_maps_dictionary(default, except_when, regions, redshift_range):
    """Build a {region: {(zmin, zmax): fit_maps}} dict. Ported unchanged
    from alaeboss.produce_imweights.make_fit_maps_dictionary."""
    fit_maps_dictionary = {region: dict.fromkeys(redshift_range, default) for region in regions}
    if except_when is not None:
        for region, zrange, alternative_maps in except_when:
            fit_maps_dictionary[region][zrange] = alternative_maps
    return fit_maps_dictionary


def produce_imweights(
    data_catalogs: np.ndarray,
    randoms_catalogs: np.ndarray,
    tracer_type: str,
    redshift_range: list,
    templates_maps_path_S,
    templates_maps_path_N,
    fit_maps,
    debv_map_path,
    sky_res_map_path_north,
    sky_res_map_path_south,
    output_directory=None,
    output_column_name="WEIGHT_IMLIN",
    save_summary_plots=True,
    nbins: int = 10,
    tail: float = 0.5,
    logger: logging.Logger | None = None,
    loglevel: str = "INFO",
    templates_maps_nside: int = 256,
    templates_maps_nested: bool = True,
    photometric_regions=None,
    region_row_masks=None,
    region_template_key=None,
):
    """Adapted version of alaeboss's `produce_imweights` -- see module
    docstring for what's different and why.

    `data_catalogs`/`randoms_catalogs` must carry RA, DEC, PHOTSYS, Z,
    WEIGHT, WEIGHT_FKP, WEIGHT_SYS, WEIGHT_ZFAIL (clustering-catalog
    convention: the regression uses WEIGHT*WEIGHT_FKP/WEIGHT_SYS/WEIGHT_ZFAIL
    as the pre-regression object weight, i.e. whatever WEIGHT_SYS the input
    already carries is divided back out before the new fit).

    `photometric_regions`/`region_row_masks`/`region_template_key` default to
    the original LRG/ELG behavior (`["S", "N"]`, each region selected by
    `PHOTSYS == region`, each region's own template file) -- pass all three
    to regress a different region scheme (e.g. QSO's 3-way DES split: region
    names `["N", "SnotDES", "DES"]`, with `SnotDES`/`DES` both reading the "S"
    template file but split by `region_row_masks` instead of `PHOTSYS`
    equality). `region_row_masks[region]` is a callable `table -> bool array`
    evaluated against `all_data`/`rands` (which always carry RA/DEC/PHOTSYS).

    Returns the fitted weights, same length/order as `data_catalogs`
    (1.0 where a weight wasn't computed, e.g. outside `redshift_range`).
    """
    time_start = time()
    logger = logger or logging.getLogger("dummy")
    logger.info("Doing linear regression for imaging systematics")

    photometric_regions = photometric_regions or ["S", "N"]
    region_template_key = region_template_key or {r: r for r in photometric_regions}
    if region_row_masks is None:
        # Compare the astropy Column directly (t["PHOTSYS"] == r), THEN wrap in
        # np.asarray -- not the other way around. Column.__eq__ correctly
        # decodes bytes-dtype string columns (e.g. after a FITS round-trip,
        # where character columns always come back as fixed-width bytes);
        # np.asarray(column) == r strips that handling and silently returns
        # all-False when comparing a bytes array against a Python str.
        region_row_masks = {r: (lambda t, r=r: np.asarray(t["PHOTSYS"] == r)) for r in photometric_regions}

    if isinstance(fit_maps, dict):
        pass
    elif isinstance(fit_maps, list) and isinstance(fit_maps[0], str):
        fit_maps = make_fit_maps_dictionary(
            default=fit_maps, except_when=None, regions=photometric_regions, redshift_range=redshift_range
        )
    else:
        raise TypeError("Argument `fit_maps` is neither a dictionary nor a list of strings.")

    all_fit_maps = sorted(
        {map_name for fitmap_r in fit_maps.values() for fitmap_rz in fitmap_r.values() for map_name in fitmap_rz}
    )
    logger.debug("All fit maps: %s", all_fit_maps)

    fit_maps_masks = {
        region: {
            zrange: np.array([(mapname in fit_maps_rz) for mapname in all_fit_maps], dtype=bool)
            for zrange, fit_maps_rz in fit_maps_r.items()
        }
        for region, fit_maps_r in fit_maps.items()
    }

    data_colnames = ["RA", "DEC", "PHOTSYS", "Z", "WEIGHT", "WEIGHT_FKP", "WEIGHT_SYS", "WEIGHT_ZFAIL"]
    random_colnames = ["RA", "DEC", "PHOTSYS", "Z", "WEIGHT", "WEIGHT_FKP", "WEIGHT_SYS", "WEIGHT_ZFAIL"]

    debv = get_debv(debv_map_path)
    sky_g, sky_r, sky_z = get_skyres(sky_res_map_path_north, sky_res_map_path_south)
    if output_directory is not None:
        output_directory = Path(output_directory)

    all_data = Table(data_catalogs[data_colnames], copy=True)
    rands = Table(randoms_catalogs[random_colnames], copy=True)

    weights_imlin = np.ones(len(all_data), dtype=float)

    for region in photometric_regions:
        logger.info("Loading healpix templates for region %s", region)
        templates_path = templates_maps_path_S if region_template_key[region] == "S" else templates_maps_path_N
        sys_tab = templates_path.copy() if isinstance(templates_path, (np.ndarray, Table)) else Table.read(templates_path)

        cols = list(sys_tab.dtype.names)
        for col in cols:
            if "DEPTH" in col:
                bnd = col.split("_")[-1]
                sys_tab[col] *= 10 ** (-0.4 * ext_coeff[bnd] * sys_tab["EBV"])
        for ec in ["GR", "RZ"]:
            sys_tab["EBV_DIFF_" + ec] = debv["EBV_DIFF_" + ec]
        if "ZCMB" in all_fit_maps:
            sys_tab["ZCMB"] = mk_zcmbmap(nside=templates_maps_nside, nest=templates_maps_nested)
        if "SKY_RES_G" in all_fit_maps:
            sys_tab["SKY_RES_G"] = sky_g[region_template_key[region]]
        if "SKY_RES_R" in all_fit_maps:
            sys_tab["SKY_RES_R"] = sky_r[region_template_key[region]]
        if "SKY_RES_Z" in all_fit_maps:
            sys_tab["SKY_RES_Z"] = sky_z[region_template_key[region]]

        logger.info("Masking randoms/data according to photometric region %s", region)
        region_mask_randoms = np.asarray(region_row_masks[region](rands))
        region_mask_data = np.asarray(region_row_masks[region](all_data))
        region_randoms = rands[region_mask_randoms]

        randoms_templates_values = _read_systematic_templates_stacked(
            ra=region_randoms["RA"], dec=region_randoms["DEC"], sys_tab=sys_tab,
            use_maps=all_fit_maps, nside=templates_maps_nside, nest=templates_maps_nested,
        )

        for z_range in redshift_range:
            logger.info("Region %s, redshift bin %s", region, z_range)
            local_fit_maps = sorted(fit_maps[region][z_range])
            local_fit_maps_mask = fit_maps_masks[region][z_range]
            t1 = time()

            selection_data = region_mask_data & (all_data["Z"] > z_range[0]) & (all_data["Z"] < z_range[1])
            selected_data = all_data[selection_data]

            selection_randoms = (region_randoms["Z"] > z_range[0]) & (region_randoms["Z"] < z_range[1])
            selected_randoms = region_randoms[selection_randoms]
            selected_randoms_templates_values = randoms_templates_values[local_fit_maps_mask, :][:, selection_randoms]

            data_templates_values = _read_systematic_templates_stacked(
                ra=selected_data["RA"], dec=selected_data["DEC"], sys_tab=sys_tab,
                use_maps=local_fit_maps, nside=templates_maps_nside, nest=templates_maps_nested,
            )

            data_we = jnp.ones(len(selected_data), dtype=jnp.float64)
            rand_we = jnp.ones(len(selected_randoms), dtype=jnp.float64)
            data_we *= (
                np.asarray(selected_data["WEIGHT"]) * np.asarray(selected_data["WEIGHT_FKP"])
                / np.asarray(selected_data["WEIGHT_SYS"]) / np.asarray(selected_data["WEIGHT_ZFAIL"])
            )
            rand_we *= (
                np.asarray(selected_randoms["WEIGHT"]) * np.asarray(selected_randoms["WEIGHT_FKP"])
                / np.asarray(selected_randoms["WEIGHT_SYS"]) / np.asarray(selected_randoms["WEIGHT_ZFAIL"])
            )
            data_we *= jnp.asarray(selected_data["WEIGHT_ZFAIL"], dtype=jnp.float64)

            regressor = LinearRegressor.from_stacked_templates(
                data_weights=data_we, random_weights=rand_we,
                template_values_data=data_templates_values,
                template_values_randoms=selected_randoms_templates_values,
                template_names=local_fit_maps, loglevel=loglevel,
            )
            regressor.cut_outliers(tail=tail)
            regressor.prepare(nbins=nbins)
            optimized_parameters = regressor.regress_minuit()
            logger.info("Optimized parameters: %s", optimized_parameters)

            if output_directory is not None:
                output_directory.mkdir(parents=True, exist_ok=True)
                stem = f"{tracer_type}_{region}_{z_range[0]:.1f}_{z_range[1]:.1f}"
                with open(output_directory / f"{stem}_linfitparam.txt", "w") as fo:
                    for par_name, par_value in optimized_parameters.items():
                        fo.write(f"{par_name} {par_value}\n")
                if save_summary_plots:
                    fig, _ = regressor.plot_overdensity(ylim=[0.7, 1.3])
                    fig.savefig(output_directory / f"{stem}_linimsysfit.png")

            weights_imlin[np.asarray(selection_data)] = regressor.export_weights()
            logger.info("Region %s, z-bin %s done in %.1fs", region, z_range, time() - t1)

    logger.info("All linear regressions done, took %ds.", int(time() - time_start))
    return weights_imlin
