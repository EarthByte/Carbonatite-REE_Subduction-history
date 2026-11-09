# Data

## Released with the paper: `data/released/`

- `released/carbonatite_deposits.csv`
- `released/points_measures.csv.gz`
- `released/measure_stages.csv`
- `released/points_masks.csv`
- `released/points_craton_distance.csv`
- `released/points_subducted_volatiles.csv.gz`
- `released/points_subduction_history.csv.gz`
- `released/points_mantle_reservoirs.csv.gz`
- `released/mantle_memory_capped_steps.csv`
- `released/points_dynamic_topography.csv.gz`
- `released/comparisons_and_model_skill.csv`
- `released/measure_contributions.csv.gz`
- `released/fertilised_domains.csv`
- `released/numbers_through_time.csv`
- `released/recurrence.csv`
- `released/isotope_districts.csv`
- `released/further_checks.csv`
- `released/aireee_settings.csv`
- `released/reveal_lid_thickness.nc`
- `released/llsvp_vote_4models.npz`
- `released/llsvp_model_agreement.csv`

**Deposit compilation** (`carbonatite_deposits.csv`). Carbonatites, alkaline-silicate rare-earth deposits,
kimberlites and porphyry copper deposits with their sources, ages, plate IDs, the begin age of the static polygon
holding each (`in_model_at_age` is false where the deposit is older than it) and their districts, as written by
`scripts/89_carbonatite_set.py`.

**Per-point tables** (`points_*.csv`, `points_*.csv.gz`). Every deposit and reference point, keyed by `pt_id`:
every measure of the paper labelled by mineral-system stage (`points_measures.csv.gz`, with `measure_stages.csv`
naming the stage of each column), the ice and exploration masks, craton-edge distance at the age of the point,
the subducted water and carbon sampled beneath it, the deliveries integrated along its plate path and its
subduction-episode history, the mantle reservoirs sampled beneath it (with the capped single-step deliveries
listed in `mantle_memory_capped_steps.csv`) and its dynamic topography through time.

**Results** (the remaining tables). Every comparison and the random-forest skill of every model
(`comparisons_and_model_skill.csv`), the contribution of each measure to each site's score
(`measure_contributions.csv.gz`), the subduction-fertilised domains against reference points
(`fertilised_domains.csv`), district numbers against global subduction (`numbers_through_time.csv`), recurrence
of districts at a site (`recurrence.csv`) and the initial neodymium and strontium isotope ratios of the districts
(`isotope_districts.csv`, medians of GEOROC samples; the samples themselves are under the GEOROC licence and are
not redistributed).

**Tectonic settings** (`aireee_settings.csv`). The tectonic setting of every alkaline-silicate rare-earth complex
in the classes of Beard et al. (2023), with the literature basis of each, read by `scripts/84`.

**Lithosphere and deep mantle** (`reveal_lid_thickness.nc`, `llsvp_vote_4models.npz`, `llsvp_model_agreement.csv`).
The thickness of the fast lithospheric lid at three thresholds and its gradient, computed from REVEAL by
`scripts/87_reveal_lithosphere.py`; and, on a 1 degree grid, the number of four independent tomographic models
(REVEAL, GLAD-M35, SPiRaL, SEMUCB-WM1) that place each cell in the slowest 20% of shear-wave speed averaged over
2,600-2,880 km depth, with the agreement between every pair of models. `rifts/llsvp.py` expects the vote grid at
`data/external/LLSVP_vote_4models/province_vote.npz`.

## Archived on Zenodo

The cumulative subducted water and carbon grids are too large for a repository and are archived on Zenodo (DOI
to be added on publication).

## Not distributed: fetched or read by the workflow

`data/raw/` - the deposit compilations, the dynamic-topography grids and the GEOROC files. `data/external/` -
craton polygons, crustal and sediment thickness, the fertilised-domain polygons, large igneous provinces, the REVEAL
grids and ice extent, all cited in `docs/datasets.md`. `data/plate_models/` - the plate model cache, populated
through the plate model manager. `data/derived/` - everything the workflow writes when it is run.
