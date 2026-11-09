# Lithospheric structure and subduction history distinguish rare-earth mineral systems

Code and data behind the paper. Carbonatites (the dated occurrences of Humphreys-Williams & Zahirovic, 2021, with
the USGS carbonatite and rare-earth deposit tables), alkaline-silicate rare-earth deposits (Orris & Grauch, 2002),
kimberlites (Gernon et al., 2023) and porphyry copper deposits (Singer et al., 2008) are reconstructed to their
formation ages through the past 1.8 billion years in a mantle reference frame (Müller et al., 2025) and compared
with continental lithosphere of the same ages. Along the plate path of every site the water and carbon carried into
subduction zones are integrated (Müller et al., 2024; Mather et al., 2026), both as delivered and as held in a
mantle reservoir that spreads what slabs deliver by horizontal diffusion; the time since the last subduction episode
beneath the overriding plate, the subduction-fertilised domains of Spandler et al. (2026), the thickness of the fast
lithospheric lid and its gradient in the REVEAL tomography model (Thrastarson et al., 2024), craton-edge distance
(Shirmard et al., 2025), dynamic topography from mantle-flow models (Dhungana & Flament, 2026; Young et al., 2022),
palaeolatitude, the large low-shear-velocity provinces and the neodymium and strontium isotopes of the districts
(GEOROC) are measured for the same points. Porphyry copper deposits pass through every comparison as the control
for a subduction signal. The manuscript is kept outside the repository.

## How a difference is tested

Each class is compared with 2,000 reference points placed on present-day continental crust, given ages drawn from
the district-weighted ages of the class, kept only where the plate model holds the crust at that age, and removed
beneath present-day ice. Deposits less than 50 km apart form one district and each district counts once. A
comparison is the probability that a value at a district exceeds the value at a reference point of the same age;
95% intervals and two-sided p-values come from resampling entire districts and reference points, p-values are
Holm-adjusted within each comparison set and mineral-system stage, and every comparison is repeated at equal
craton-edge distance and equal exploration density. Random forests trained with and without each stage of the
mineral system, scored on spatial blocks excluded from training, measure what each stage adds.

## Workflow

| Script | Output |
|---|---|
| `scripts/00_fetch_data.py`, `scripts/01_build_deposits.py` | the open deposit compilations and the harmonised table that supplies the porphyry copper control |
| `scripts/44_shirmard_cratons.py`, `scripts/51_ecm1_crust.py` | craton outlines cut by the static polygons of the plate model; crustal thickness as a grid |
| `scripts/72_kimberlite_set.py`, `scripts/89_carbonatite_set.py` | `deposits_carb26`: carbonatites, alkaline-silicate REE deposits, kimberlites and porphyry copper deposits with plate IDs and districts |
| `scripts/02_sample_kinematics.py` | `features_carb26`: every site and its reference points, with plate kinematics and boundary distances at formation |
| `scripts/01b_geological_context.py`, `scripts/15_rift_fill.py`, `scripts/17_null_masks.py`, `scripts/20_craton.py` | basin context, sediment and crustal thickness, ice and exploration masks, craton-edge distance |
| `scripts/03_water.py` | subducted water and carbon sampled beneath every point at formation and in fixed windows |
| `scripts/87_reveal_lithosphere.py` | `reveal_lid_thickness.nc`: thickness of the fast lithospheric lid and its gradient from REVEAL |
| `scripts/90_mantle_memory.py` | the diffusing mantle reservoirs of subducted carbon and water, sampled beneath every point |
| `scripts/82_primer_history.py` | delivery of slab water and carbon integrated along every plate path; time on an overriding plate and since the last subduction episode |
| `scripts/88_dynamic_topography.py` | dynamic topography beneath every point through time from six mantle-flow models |
| `scripts/83_carbonatite_features.py` | `carbonatite_features_carb26`: every measure of every point, labelled by stage of the mineral system |
| `scripts/81_spandler_reassessment.py` | the subduction-fertilised domains against reference points, by host rock and domain age |
| `scripts/84_carbonatite_systems.py` | every comparison and the random-forest skill with and without each stage |
| `scripts/85_carbonatite_secular_recurrence.py` | district numbers through time against global subduction; recurrence of districts at a site |
| `scripts/86_carbonatite_isotopes.py` | initial neodymium and strontium isotope ratios of the districts against their subduction history |
| `scripts/96_carbonatite_review_checks.py` | `further_checks.csv`: the episode measure under other radii and durations, persistence after formation, the fertilised domains under joint matching, paired model skill, isotope q-values |
| `scripts/91_carbonatite_figures.py` | Figures 1 to 6 |
| `tools/carbonatite_tables.py` | the tables of the supplement, from the data files |

## Running

The scripts need numpy, pandas, scipy, matplotlib, geopandas, shapely, cartopy, xarray and pygplates (with the
plate model manager for the plate models). Paths to the subducted-volatile grids and the external data are set in
`config/config.yaml`, where they appear as placeholders; scripts 01b and 15 skip their rift parts when the rift
outlines of the companion papers are absent.

    python scripts/00_fetch_data.py && python scripts/01_build_deposits.py
    python scripts/44_shirmard_cratons.py && python scripts/51_ecm1_crust.py
    python scripts/72_kimberlite_set.py && python scripts/89_carbonatite_set.py
    python scripts/02_sample_kinematics.py --deposits deposits_carb26 --classes CARB AIREE KIMB PCU \
        --nulls-per-class 2000 --null-weights district --tag carb26
    python scripts/01b_geological_context.py --tag carb26
    python scripts/03_water.py --tag carb26
    python scripts/15_rift_fill.py --tag carb26
    python scripts/17_null_masks.py --tag carb26 && python scripts/20_craton.py --tag carb26
    python scripts/87_reveal_lithosphere.py
    python scripts/90_mantle_memory.py --tags carb26
    python scripts/82_primer_history.py --tag carb26
    python scripts/88_dynamic_topography.py --tag carb26
    python scripts/83_carbonatite_features.py --tag carb26
    python scripts/81_spandler_reassessment.py --tag carb26
    python scripts/84_carbonatite_systems.py --tag carb26
    python scripts/85_carbonatite_secular_recurrence.py --tag carb26
    python scripts/86_carbonatite_isotopes.py --tag carb26
    python scripts/96_carbonatite_review_checks.py --tag carb26
    python scripts/91_carbonatite_figures.py --tag carb26
    python tools/carbonatite_tables.py all       # the tables of the supplement

Scripts 84 to 86 take fewer resamples (`--boot 200`) for a pilot run before a full one. Figures follow the
journal's artwork rules (no titles, a lettering floor) and every figure script checks its own artwork for
overlapping text before saving.

## Data

`data/released/` holds the deposit compilation as used, every measure of every deposit and reference point, the
comparison, skill, domain, secular, recurrence and isotope tables behind every figure and table of the paper, the
lid-thickness grid derived from REVEAL and the map of agreement between the four tomographic models from which
the large low-shear-velocity provinces are taken; `data/README.md` lists them. The compilations of
Humphreys-Williams & Zahirovic (2021), Berger et al. (2009), Orris & Grauch (2002), Gernon et al. (2023) and
Singer et al. (2008), the craton outlines of Shirmard et al. (2025), the fertilised-domain polygons of Spandler et
al. (2026), the Earth Crustal Model 1 (Mooney et al., 2023), the GST-1 sediment thickness (Bird & Mooney), the
dynamic topography of Dhungana & Flament (2026) and Young et al. (2022), the REVEAL model (Thrastarson et al.,
2024) and the GEOROC precompiled files are read from `data/raw/` and `data/external/`, obtained from their
sources as listed in `docs/datasets.md`. The subducted water and carbon grids are archived on Zenodo.

## Layout

    config/        configuration (machine-specific paths are placeholders) and the tectonic-setting table
    rifts/         the package modules the scripts use
    scripts/       one step each, each writing one file
    tools/         the tables of the supplement
    tests/         unit tests of the geometry and statistics
    docs/          data sources
    data/released/ the tables behind the paper

## Licence

MIT (see `LICENSE`). Third-party inputs keep their own licences, listed in `docs/datasets.md`.
