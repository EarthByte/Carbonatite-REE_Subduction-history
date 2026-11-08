# Open datasets: rift-related mineral deposits in a 1.8 Gyr plate-tectonic context

Compiled 2026-09-17 for the *rifting* project. Machine-readable version: `datasets.yaml` (same ids).

**How the URLs were checked:** HTTP 200 via curl, a successful `git clone`, or DOI resolution (302 from doi.org). The USGS mrdata CSV bundles, the Hoggard OSF table, the LIP XLSX, the Johansson LIP zip index and the EarthByte repos were downloaded or listed and their columns inspected. Zenodo API/record pages timed out from the sandbox (apart from two WebFetch successes), and ScienceBase file links and github.com web pages returned proxy 403s. Those entries are marked accordingly.

## GPlately PlateModelManager model names (from `https://repo.gplates.org/webdav/pmm/config/models_v2.json`, PMM 1.3.2)

| PMM name | Time span | Layers | Time-dependent rasters | Model DOI (PMM `Version`) |
|---|---|---|---|---|
| `muller2025` | 1800–0 Ma | Rotations, StaticPolygons, Coastlines, ContinentalPolygons, Topologies, COBs | AgeGrids, SpreadingRate | 10.5281/zenodo.15233548 (17142287) |
| `cao2024` | 1800–0 Ma | Rotations, StaticPolygons, Coastlines, ContinentalPolygons, Topologies, COBs | none | 10.5281/zenodo.11536686 (13628813) |

Neither model has a `Cratons` layer (`.../pmm/muller2025/Cratons.zip` returns 404). Craton outlines have to come from the EarthByte `Craton_Boundaries.shp`, Hasterok `cratons.shp`, or `muller2022`/`merdith2021` (1 Ga only). Usage: `PlateModelManager().get_model('muller2025', data_dir=...)`, then `get_rotation_model()`, `get_layer('Topologies')`, and so on.

Other keys in the config: default, shirmard2025, alfonso2024, cao2023, muller2022, zahirovic2022, merdith2021, clennett2020(+_m2019,_s2013), muller2019, scotese_and_wright2018, young2018, torsvikcocks2017, matthews2016(+_mantle_ref,_pmag_ref), muller2016, scotese2016, zahirovic2016, gibbons2015, domeier2014, zahirovic2014, shephard2013, gurnis2012, seton2012, li2008, muller2008, golonka, paleomap, rodinia.

## Quick ranking (priority 1)

| id | records | age field | type field |
|---|---|---|---|
| `craton_edges_earthbyte` | Craton_Deposits.shp 265 records | Age__Ga_ (Ga), Age_Uncert, Age_Type, Age (Ma, integer) | Type (PbZn-CD, PbZn-MVT, Cu-Sed, IOCG) |
| `sed_mpm_earthbyte` | Craton_Deposits.shp 289 (PbZn-MVT 90, Cu-Sed 89, PbZn-CD 86, IOCG 24); deposit_data.csv 204 rows x 44 columns with extracted kinematic features | Age__Ga_, Age_Uncert, 'age (Ma)', TOAGE | Type |
| `hoggard2020_deposits` | PbZn-CD 109, PbZn-MVT 147, Cu-sed 139, Magmatic Ni 108, VMS 947, Cu-por 691, IOCG 25 | 'Age (Ga)' (numeric; filled for 107/109 CD, 78/147 MVT, 139/139 Cu-sed, 108 Ni, 946 VMS, 594 porphyry, 15 IOCG); 'Age Uncertainty (Ga)' and 'Age Type' (direct vs inferred, method) for CD, MVT and Ni sheets | sheet name |
| `usgs_sedznpb` | 506 | agemy (numeric Ma, 366 filled; some are 0 for 'Alpine'); depage (free text, 393 filled) | deptype (e.g. CAam), prevtype (MVT/SEDEX) |
| `usgs_sedexmvt` | 240 | hostage (text, all rows); minagemeth (mineralisation age + method, only 63 rows); refage | deptype (MVT/CD), deptypea/b/c |
| `usgs_sedcu_2003` | 785 | agemy (numeric Ma, 778 filled; mostly host-rock stratigraphic age); depage (text) | deptype (Redbed Cu, Reduced Cu, Kupferschiefer...) |
| `usgs_sir2010_5090z` | sedcu_deps_pros 1200; sedcu_tracts 48; pcu_deps_pros 2199; pcu_precambrian 21; pcu_tracts 192 | sedcu: age_ma (all rows; host age), age_host (text). pcu: age_ma (-9999 = none), age_method, age_range, strat_age | dep_type, dep_subtype (e.g. Sandstone copper, Reduced-facies, Kupferschiefer; Cu-Au/Cu-Mo) |
| `usgs_potash` | 981 sites (+ basin tracts) | max_age/min_age (hierarchical chronostratigraphic strings, 953 filled); age_era/period/epoch/stage | dep_type (evaporite, halokinetic salt, brine); basin_type (rift, foreland, intracratonic...) |
| `gernon2023_kimberlite` | not verified | kimberlite eruption ages (from Tappe et al. 2018 compilation) | - |
| `hasterok2022_provinces` | global_gprv 914 provinces | lastorogen (named orogeny, not numeric) | prov_type (wide rift 29, narrow rift 26, passive margin 55, back-arc basin, basin, craton, shield...) |
| `humphreys_williams2021` | 609 (387 dated) | emplacement age (Ma) | - |

## Plate models and EarthByte workflow data

### Muller2025 mantle-reference-frame plate model (1.8 Ga), PlateModelManager name 'muller2025'  
`pmm_muller2025` · priority 1

- **Citation:** Müller et al. (2025) mantle-frame version of Cao et al. (2024); Zenodo record 'Muller2025 Mantle Reference Frame Plate Model based on Cao et al. (2024)' v1.3
- **DOI:** 10.5281/zenodo.15233548 (concept); PMM pins version 10.5281/zenodo.17142287
- **URL:** https://repo.gplates.org/webdav/pmm/muller2025/Rotations.zip (plus Topologies.zip, StaticPolygons.zip, Coastlines.zip, ContinentalPolygons.zip, COBs.zip under the same folder); config https://repo.gplates.org/webdav/pmm/config/models_v2.json
- **Format / licence:** GPlates .rot/.gpml(z) in zip; NetCDF age and spreading-rate grids / Open (Zenodo; licence not shown on record page - check); plate-model-manager is GPL-2.0
- **Records:** BigTime 1800 Ma, SmallTime 0
- **Age field:** n/a
- **Type field:** n/a
- **Notes:** PMM layers: Rotations, StaticPolygons, Coastlines, ContinentalPolygons, Topologies, COBs. TimeDepRasters: AgeGrids (Muller_etal_2025/SeafloorAge/SEAFLOOR_AGE_grid_{:.2f}Ma.nc) and SpreadingRate (Cao2024_SPREADING_RATE_grid_{:.2f}Ma.nc). No Cratons layer (Cratons.zip returns 404). Use for absolute plate/trench/rift velocities.

### Cao et al. 2024 1.8 Ga palaeomagnetic-frame plate model, PlateModelManager name 'cao2024'  
`pmm_cao2024` · priority 1

- **Citation:** Cao, X., et al. (2024) Geoscience Frontiers 15, 101922
- **DOI:** 10.1016/j.gsf.2024.101922; data 10.5281/zenodo.11536686 (PMM version 10.5281/zenodo.13628813)
- **URL:** https://repo.gplates.org/webdav/pmm/cao2024/Rotations.zip (plus Topologies, StaticPolygons, Coastlines, ContinentalPolygons, COBs)
- **Format / licence:** GPlates files in zip / Open (Zenodo; check record)
- **Records:** BigTime 1800 Ma
- **Age field:** n/a
- **Type field:** n/a
- **Notes:** Same layers as muller2025 but TimeDepRasters is empty (no age grids in PMM). No Cratons layer. Use for palaeolatitude only; longitudes are not meaningful in a palaeomagnetic frame.

### Other PMM models for cross-checks (muller2022, merdith2021, zahirovic2022, cao2023, alfonso2024, shirmard2025)  
`pmm_other` · priority 3

- **Citation:** See models_v2.json Description/URL fields
- **DOI:** e.g. merdith2021 10.5281/zenodo.13635864
- **URL:** https://repo.gplates.org/webdav/pmm/config/models_v2.json
- **Format / licence:** JSON config + zips / varies (mostly CC BY)
- **Records:** 33 model keys
- **Age field:** n/a
- **Type field:** n/a
- **Notes:** muller2022 (1000 Ma) and merdith2021 (1000 Ma) include Cratons and PaleomagneticData (muller2022) layers; useful as a 1 Ga comparison with a pmag/mantle-frame pair.

### EarthByte Craton-deposits_Tectonic-evolution repo (craton boundaries, deposits, 1.8 Ga optimised mantle-frame model)  
`craton_edges_earthbyte` · priority 1

- **Citation:** Farahbakhsh/Shirmard et al. (EarthByte); related Shirmard et al. 2025 Geosci. Frontiers 10.1016/j.gsf.2025.102176
- **DOI:** 10.5281/zenodo.15015718
- **URL:** https://github.com/EarthByte/Craton-deposits_Tectonic-evolution (git clone works)
- **Format / licence:** Shapefiles (Craton_Boundaries, Craton_Deposits, Random_Points), notebooks, xlsx / GPL-3.0
- **Records:** Craton_Deposits.shp 265 records
- **Age field:** Age__Ga_ (Ga), Age_Uncert, Age_Type, Age (Ma, integer)
- **Type field:** Type (PbZn-CD, PbZn-MVT, Cu-Sed, IOCG)
- **Notes:** Deposits derived from Hoggard et al. 2020 table; already has PLATEID1 and distance-to-craton-edge fields. Ready-made notebooks sample plate kinematics at deposit age and averaged over 10 Myr before it.

### EarthByte Sed-hosted-Spatio-Temporal-MPM (Inputs.zip)  
`sed_mpm_earthbyte` · priority 1

- **Citation:** EarthByte (Farahbakhsh et al.); builds on Shirmard et al. 2025/2026 and Farahbakhsh et al. 2023 (10.1016/j.oregeorev.2023.105665), 2025 (10.1029/2024TC008362)
- **DOI:** 10.5281/zenodo.20806346
- **URL:** https://github.com/EarthByte/Sed-hosted-Spatio-Temporal-MPM (git clone works)
- **Format / licence:** Shapefiles + CSV (Inputs.zip, 1.8 MB) / not stated in repo (check)
- **Records:** Craton_Deposits.shp 289 (PbZn-MVT 90, Cu-Sed 89, PbZn-CD 86, IOCG 24); deposit_data.csv 204 rows x 44 columns with extracted kinematic features
- **Age field:** Age__Ga_, Age_Uncert, 'age (Ma)', TOAGE
- **Type field:** Type
- **Notes:** deposit_data.csv already carries reconstructed coordinates and features (cr_ve, dp_ve, sb_ve, tr_adv, lat, latabs...). unlabelled_data.csv (5.4 MB) holds background points. Directly reusable as a training/validation set.

### EarthByte water_subduction (subducted/stored H2O and slab outflux through time)  
`water_subduction` · priority 2

- **Citation:** Mather, B.R., Müller, R.D., Dutkiewicz, A. et al. (2026) Commun. Earth Environ. 7, 48
- **DOI:** 10.1038/s43247-025-03097-0
- **URL:** https://github.com/EarthByte/water_subduction (git clone works)
- **Format / licence:** Jupyter notebooks + Perple_X lookup tables (.dat, up to ~56 MB each) / not stated in repo (check)
- **Records:** lookup tables for GLOSS sediments, altered volcanics, intrusives, serpentinised mantle (H2O, CO2, mineral modes)
- **Age field:** n/a (time-dependent grids produced by the notebooks)
- **Type field:** n/a
- **Notes:** Needs input grids (CarbonateSediment, ContinentalMasks, ...) not in the repo, and gplately + slabdip. The paper used muller2022 (540 Ma); extending to 1.8 Ga needs the muller2025 age grids. Slab outflux at ~125 km depth can be sampled along trenches near rifts at deposit age.

## Dated deposit compilations (core)

### Hoggard et al. 2020 global sediment-hosted base-metal deposit compilation  
`hoggard2020_deposits` · priority 1

- **Citation:** Hoggard, M.J., Czarnota, K., Richards, F.D., Huston, D.L., Jaques, A.L., Ghelichkhan, S. (2020) Global distribution of sediment-hosted metals controlled by craton edge stability. Nature Geoscience 13, 504-510
- **DOI:** 10.1038/s41561-020-0593-2
- **URL:** https://osf.io/download/r5mfz/ (base_metal_deposit_compilation.xls, 402 KB; OSF project https://osf.io/twksd/)
- **Format / licence:** XLS, 7 sheets / OSF project: no licence set (supplement of a Nature paper; cite)
- **Records:** PbZn-CD 109, PbZn-MVT 147, Cu-sed 139, Magmatic Ni 108, VMS 947, Cu-por 691, IOCG 25
- **Age field:** 'Age (Ga)' (numeric; filled for 107/109 CD, 78/147 MVT, 139/139 Cu-sed, 108 Ni, 946 VMS, 594 porphyry, 15 IOCG); 'Age Uncertainty (Ga)' and 'Age Type' (direct vs inferred, method) for CD, MVT and Ni sheets
- **Type field:** sheet name
- **Notes:** Best starting point: numeric ages in Ga, metal tonnages for weighting. MVT ages are often 'ND' (not determined). Ages for Cu-sed, VMS and porphyry have no uncertainty column. Includes contrast classes (VMS, porphyry, magmatic Ni).

### USGS Sediment-hosted zinc-lead deposits of the world (grade-tonnage database)  
`usgs_sedznpb` · priority 1

- **Citation:** Singer, D.A., Berger, V.I., Moring, B.C. (2009) USGS Open-File Report 2009-1252
- **DOI:** 10.3133/ofr20091252
- **URL:** https://mrdata.usgs.gov/sedznpb/sedznpb-csv.zip (321 KB)
- **Format / licence:** CSV (main, minerals, ref) + SHP + KML / US public domain
- **Records:** 506
- **Age field:** agemy (numeric Ma, 366 filled; some are 0 for 'Alpine'); depage (free text, 393 filled)
- **Type field:** deptype (e.g. CAam), prevtype (MVT/SEDEX)
- **Notes:** Also has tectset (tectonic setting), host rock and grades. agemy is often host-rock or orogenic age, so check it against depage.

### USGS Mineral resource data for MVT and clastic-dominated Pb-Zn deposits (Leach/Taylor compilation)  
`usgs_sedexmvt` · priority 1

- **Citation:** Taylor, R.D., Leach, D.L., Bradley, D.C., Pisarevsky, S.A. (2009) USGS OFR 2009-1297; companion to Leach et al. 2010 Econ. Geol. 105, 593 (10.2113/gsecongeo.105.3.593)
- **DOI:** 10.3133/ofr20091297
- **URL:** https://mrdata.usgs.gov/sedexmvt/sedexmvt-csv.zip (71 KB)
- **Format / licence:** CSV + SHP / US public domain
- **Records:** 240
- **Age field:** hostage (text, all rows); minagemeth (mineralisation age + method, only 63 rows); refage
- **Type field:** deptype (MVT/CD), deptypea/b/c
- **Notes:** Only openly released table behind the Leach et al. 2010 'Pb-Zn in Earth history' secular analysis. Mineralisation ages are sparse, so combine with Hoggard 2020.

### USGS Sediment-hosted copper deposits of the world  
`usgs_sedcu_2003` · priority 1

- **Citation:** Cox, D.P., Lindsey, D.A., Singer, D.A., Diggles, M.F. (2003) USGS OFR 03-107
- **DOI:** 10.3133/ofr2003107
- **URL:** https://mrdata.usgs.gov/sedcu/sedcu-csv.zip (120 KB)
- **Format / licence:** CSV + SHP + KML / US public domain
- **Records:** 785
- **Age field:** agemy (numeric Ma, 778 filled; mostly host-rock stratigraphic age); depage (text)
- **Type field:** deptype (Redbed Cu, Reduced Cu, Kupferschiefer...)
- **Notes:** Has grades and tonnage (cugrd, oreton). Ages are host-rock ages, not mineralisation ages.

### USGS Global assessment of undiscovered copper resources: GIS (includes sediment-hosted stratabound Cu, SIR 2010-5090-J, and porphyry Cu)  
`usgs_sir2010_5090z` · priority 1

- **Citation:** Dicken, C.L., Dunlap, P., Parks, H.L., Hammarstrom, J.M., Zientek, M.L., et al. (2016) USGS SIR 2010-5090-Z; sed-Cu model: Zientek, Hayes et al. (2014) SIR 2010-5090-J (10.3133/sir20105090J)
- **DOI:** 10.3133/sir20105090Z
- **URL:** https://mrdata.usgs.gov/sir20105090z/sir20105090z-csv.zip (64 MB)
- **Format / licence:** CSV + GIS / US public domain
- **Records:** sedcu_deps_pros 1200; sedcu_tracts 48; pcu_deps_pros 2199; pcu_precambrian 21; pcu_tracts 192
- **Age field:** sedcu: age_ma (all rows; host age), age_host (text). pcu: age_ma (-9999 = none), age_method, age_range, strat_age
- **Type field:** dep_type, dep_subtype (e.g. Sandstone copper, Reduced-facies, Kupferschiefer; Cu-Au/Cu-Mo)
- **Notes:** Most complete global sediment-hosted Cu table (1200 sites) with basin permissive tracts (polygons with basin name). Porphyry table is the contrast class, with isotopic age methods.

### USGS evaporite-related potash deposits (point locations and characteristics)  
`usgs_potash` · priority 1

- **Citation:** Orris, G.J., Cocker, M.D., et al. (2014) USGS SIR 2010-5090-S
- **DOI:** 10.3133/sir20105090S
- **URL:** https://mrdata.usgs.gov/potash/potash-csv.zip (363 KB); tracts https://mrdata.usgs.gov/potash/ktract.zip
- **Format / licence:** CSV + SHP + KML + XLSX / US public domain
- **Records:** 981 sites (+ basin tracts)
- **Age field:** max_age/min_age (hierarchical chronostratigraphic strings, 953 filled); age_era/period/epoch/stage
- **Type field:** dep_type (evaporite, halokinetic salt, brine); basin_type (rift, foreland, intracratonic...)
- **Notes:** Includes basin name and basin_type, which is directly useful for rift versus other basin settings. Map the stage names to Ma with the ICS/Macrostrat interval table.

### Humphreys-Williams & Zahirovic 2021 carbonatite compilation (609 occurrences, 387 dated)  
`humphreys_williams2021` · priority 1

- **Citation:** Humphreys-Williams, E.R., Zahirovic, S. (2021) Carbonatites and global tectonics. Elements 17(5), 339-344
- **DOI:** 10.2138/gselements.17.5.339
- **URL:** https://doi.org/10.5281/zenodo.5968095 (digital supplement, v2)
- **Format / licence:** xlsx (occurrence number, name, age, error, reference) + shapefile / CC BY 4.0
- **Records:** 609 (387 dated)
- **Age field:** emplacement age (Ma)
- **Type field:** -
- **Used:** data/raw/magmatic/HumphreysWilliams_Zahirovic_2021 (md5 checked against the record); occurrence numbers follow Woolley & Kjarsgaard 2008. The GPlates shapefile (593 located occurrences, 376 dated with error and reference) is read by `rifts.deposits.read_carbonatites_hwz2021` (source `hwz2021_carbonatite`; only occurrences flagged 'c'; an error of zero becomes 2%, at least 1 Myr) into the carbonatite paper's own set, `deposits_carb26` (scripts/89): 379 dated carbonatites in 271 districts. The shared compilation keeps Woolley & Kjarsgaard 2008.
- **Notes:** Already reconstructed with EarthByte models. The Woolley & Kjarsgaard 2008 GSC OF 5796 (527 carbonatites) is the underlying source; the geoscan.nrcan.gc.ca host was unreachable from the sandbox (DOI 10.4095/225115 resolves).

### USGS Carbonatites of the world, explored deposits of Nb and REE  
`usgs_carbonatite_nbree` · priority 2

- **Citation:** Berger, V.I., Singer, D.A., Orris, G.J. (2009) USGS OFR 2009-1139
- **DOI:** 10.3133/ofr20091139
- **URL:** https://mrdata.usgs.gov/carbonatite/carbonatite-csv.zip (51 KB)
- **Format / licence:** CSV + SHP + KML / US public domain
- **Records:** 60
- **Age field:** agemy (numeric Ma, all 60); depage (text with method, e.g. U-Pb zircon)
- **Type field:** deptype, mintype; tectset
- **Notes:** Small, but ages are high quality and records have tectonic setting. Explored deposits only.

### USGS Rare earth element mines, deposits and occurrences  
`usgs_ree` · priority 2

- **Citation:** Orris, G.J., Grauch, R.I. (2002) USGS OFR 02-189
- **DOI:** 10.3133/ofr02189
- **URL:** https://mrdata.usgs.gov/ree/ree-csv.zip (151 KB)
- **Format / licence:** CSV + SHP + KML / US public domain
- **Records:** 799
- **Age field:** age_info (free text, 358 filled; mixes isotopic ages and periods)
- **Type field:** deptype (carb, alkaline, placer, ...)
- **Notes:** deptype 'carb' gives carbonatite-hosted REE occurrences. Ages have to be parsed from text.

### USGS Volcanogenic massive sulfide deposits of the world  
`usgs_vms` · priority 2

- **Citation:** Mosier, D.L., Berger, V.I., Singer, D.A. (2009) USGS OFR 2009-1034
- **DOI:** 10.3133/ofr20091034
- **URL:** https://mrdata.usgs.gov/vms/vms-csv.zip (499 KB)
- **Format / licence:** CSV + SHP + KML / US public domain
- **Records:** 1090
- **Age field:** agemy (numeric Ma, 1087 filled), depage (text)
- **Type field:** deptype (Felsic, Bimodal-mafic, Mafic)
- **Notes:** Mafic and bimodal VMS form in rift and back-arc settings, so they are a useful contrast or complement.

## Deep magmatism: kimberlites and LIPs

### Gernon et al. 2023 supplementary files (kimberlite ages, rift timing)  
`gernon2023_kimberlite` · priority 1

- **Citation:** Gernon, T.M., Jones, S.M., Brune, S., et al. (2023) Rift-induced disruption of cratonic keels drives kimberlite volcanism. Nature 620, 344-350
- **DOI:** 10.5281/zenodo.7849142 (paper 10.1038/s41586-023-06193-3)
- **URL:** https://doi.org/10.5281/zenodo.7849142
- **Format / licence:** Zenodo archive (contents not verified; Zenodo timed out from the sandbox) / not verified
- **Records:** not verified
- **Age field:** kimberlite eruption ages (from Tappe et al. 2018 compilation)
- **Type field:** -
- **Notes:** Most directly relevant to 'rifting leads to deep magmatism'. The DOI resolves; open it in a browser to confirm the files.

### Tappe et al. 2018 global kimberlite age database  
`tappe2018_kimberlite` · priority 2

- **Citation:** Tappe, S., Smart, K., Torsvik, T., Massuyeau, M., de Wit, M. (2018) Geodynamics of kimberlites on a cooling Earth. EPSL 484, 1-14
- **DOI:** 10.1016/j.epsl.2017.12.013
- **URL:** https://doi.org/10.1016/j.epsl.2017.12.013
- **Format / licence:** Elsevier supplementary (xlsx) / publisher supplement (check re-use terms)
- **Records:** ~1130 dated kimberlites (as used by Grabreck et al. 2021, PLOS ONE 10.1371/journal.pone.0268066)
- **Age field:** emplacement age (Ma) with method
- **Type field:** kimberlite class
- **Notes:** Supplement not downloaded (paywalled landing page). Gernon 2023 and Grabreck et al. reuse it.

### Large Igneous Provinces Commission global LIP listing (Ernst 2014 table, rev. 2016)  
`lip_commission_table` · priority 2

- **Citation:** Ernst, R.E. (2014) Large Igneous Provinces, CUP; Ernst & Youbi (2017) Palaeo3 478, 30-52 (10.1016/j.palaeo.2017.03.014)
- **DOI:** 10.1016/j.palaeo.2017.03.014
- **URL:** http://www.largeigneousprovinces.org/sites/default/files2/Global%20LIP%20Table%202.1%20REV%20FINAL.XLSX
- **Format / licence:** XLSX / none stated
- **Records:** about 212 LIP events (Archean to present)
- **Age field:** 'Age (Ga)'
- **Type field:** Type (e.g. flood basalt, dyke swarm)
- **Notes:** No coordinates, only location names, so it must be georeferenced (for example by joining to the Johansson polygons for Phanerozoic events). Covers the full 1.8 Ga window.

### Johansson et al. 2018 EarthByte LIP polygons (GPlates-ready)  
`johansson2018_lips` · priority 2

- **Citation:** Johansson, L., Zahirovic, S., Müller, R.D. (2018) GRL 45, 5380-5389
- **DOI:** 10.1029/2017GL076691
- **URL:** https://www.earthbyte.org/webdav/ftp/Data_Collections/Johansson_etal_2018_GRL/Johansson_etal_2018_GRL.zip (114 MB; mostly an MPEG animation)
- **Format / licence:** gpmlz + shapefile / CC BY 4.0
- **Records:** 2526 polygons (v2)
- **Age field:** FROMAGE/TOAGE (Ma)
- **Type field:** TYPE / NAME
- **Notes:** Plate IDs follow Matthews et al. 2016 (410-0 Ma), so reassign them with muller2025 StaticPolygons before reconstructing. Phanerozoic focus.
- **Used:** data/external/Johansson2018_LIPs_v2 (copy from the global_tectonics repository of Hasterok et al.). The kimberlite tests (scripts/74) drop seamount chains, hotspot tracks, ridges, rises, plateaus, volcanic margins, seaward-dipping reflectors and breakup basalts (by NAME, or DESCR where NAME is empty), group polygons into provinces, and give each vertex the muller2025 plate beneath it.

### GEOROC precompiled rock-type files (carbonatite, phoscorite, syenite)
`georoc_rocktypes_2026_09` · used

- **Citation:** DIGIS Team (2026) GEOROC Compilation: Rock Types, GRO.data, version 15 (generated 2026-09-01)
- **DOI:** 10.25625/2JETOA
- **Format / licence:** CSV, one row per analysis with location, material, major and trace elements and Sr-Nd-Pb-Hf isotopes / CC BY-SA 4.0
- **Records:** carbonatite 4,504 analyses (Sr isotopes 1,236, Nd 1,123, Pb 606); syenite 5,233
- **Used:** data/raw/georoc_2026-09 (md5 in README); scripts/86 matches samples to deposit districts within 25 km and computes initial Sr and Nd at the district age.

### Young et al. 2022 plate-frame dynamic topography (mantle-flow run gld428)
`young2022_dyntopo` · used

- **Citation:** Young, A. et al. (2022) Earth and Planetary Science Letters 584, 117451
- **Format:** one netCDF per age (`gld428_PlateFrameGrid<age>.nc`, variable `z`, m, 0.125 degrees), every 20 Myr from 0 to 980 Ma; continents only
- **Frame:** plate frame: the dynamic topography at a past age at the present-day coordinates of each continental block (Merdith et al. 2021 plate model, no-net-rotation frame), so a site is sampled at its present position at any age
- **Used:** read from the GPlately tutorial data bundle (zenodo_data/dynamic_topography_Young2022; path in config `dt_young2022_dir`) by scripts/88 as a sensitivity case (columns suffixed `_young`): dynamic topography at the age of each point, its uplift rate over the 20 and 60 Myr before and the 20 Myr after, and the rise since its lowest value in the 200 Myr before.

### Dhungana & Flament plate-frame dynamic topography of six mantle-flow cases
`dhungana2025_dyntopo` · used

- **Citation:** Dhungana, S., Flament, N. (2026) Deep Earth Origin of the Great Unconformity - Datasets and Scripts, v1.1.0, Zenodo 10.5281/zenodo.20603294 (CC BY 4.0)
- **Format:** one tarball per CitcomS case in data/raw/dynamic_topography_Dhungana2025 (config `dt_dhungana2025_dir`), each with 51 netCDF grids (`<run>PlateFrameGrid<age>.nc`, variable `z`, m, 0.5 degrees), every 20 Myr from 0 to 1000 Ma; cases M1 gld504, M2 gld582, M3 gld583 (paleomagnetic-frame twin of M2), M4 gld486, M5 gld434, M6 gld428
- **Frame:** plate frame, as for Young et al. (2022); all six cases processed alike (latitude correction and removal of the global mean at each age)
- **Used:** scripts/88 computes the same five measures for each case; the median over M1-M6 is the dynamic-topography feature set of scripts/83 and 84, each case alone is a sensitivity column, and `dt_uplift_20_agree` is the share of cases that agree in sign with the median uplift. Plume propensity (`dt_high_p500`, `dt_rise_p500`; 1,000 km as sensitivity) is the share of the grids from 60 Myr before to 20 Myr after formation in which dynamic topography, or its uplift rate, above the 90th percentile of the grid lies within the radius, averaged over the six cases: model plumes rise freely and are not assimilated, so they are compared with the record statistically, by neighbourhood and window. Extracted from the record's data.zip with tools/zenodo_zip_extract.py (README in the folder).

### LLSVPs agreed by four independent tomographic models
`llsvp_vote_4models` · used

- **Sources:** REVEAL (Thrastarson et al. 2024, BSSA 114, doi 10.1785/0120230273), GLAD-M35 (Cui et al. 2024, GJI 239, doi 10.1093/gji/ggae270), SPiRaL (Simmons et al. 2021, GJI 227, doi 10.1093/gji/ggab277), SEMUCB-WM1 (French & Romanowicz 2014, GJI 199, doi 10.1093/gji/ggu334)
- **Construction:** in each model the mean shear-velocity anomaly over 2,600-2,880 km depth is resampled to a 1-degree grid and thresholded at its slowest 20 per cent; connected regions of at least 60 cells are kept; `vote` counts the models in which each cell is province. Single models place the edges a median of about 380 km apart (`province_agreement.csv`); agreed by at least two, three and four models: 29, 24 and 17 per cent of the core-mantle boundary.
- **Format:** NumPy archive (`vote`, `lat`, `lon`, `n_families`, `thresholds`, `tags`) and a CSV of pairwise agreement
- **Used:** data/external/LLSVP_vote_4models (province_agreement.py of the plume study run on the four published models); read by rifts/llsvp.py, which takes cells agreed by at least three of the four models (24 per cent of the sphere; two of four 29 per cent, four of four 17 per cent) and drops regions under 60 cells. The provinces are held fixed in the mantle frame; distances are measured from a site's mantle-frame position at its age, negative inside (scripts/74, 76-78 and 83).

## Rifts, basins, lithosphere

### Hasterok et al. 2022 global geological provinces and plates (includes rift provinces)  
`hasterok2022_provinces` · priority 1

- **Citation:** Hasterok, D., Halpin, J., Collins, A., Hand, M., Kreemer, C., Gard, M., Glorie, S. (2022) Earth-Science Reviews 231, 104069
- **DOI:** 10.1016/j.earscirev.2022.104069; data 10.5281/zenodo.5093930
- **URL:** https://raw.githubusercontent.com/dhasterok/global_tectonics/main/README.md (repo github.com/dhasterok/global_tectonics)
- **Format / licence:** SHP, GMT, KML / GPL-3.0 (repo LICENSE, 35 KB)
- **Records:** global_gprv 914 provinces
- **Age field:** lastorogen (named orogeny, not numeric)
- **Type field:** prov_type (wide rift 29, narrow rift 26, passive margin 55, back-arc basin, basin, craton, shield...)
- **Notes:** Openly licensed rift and passive-margin polygons, plus cratons.shp. Can serve as a rift and basin mask to intersect with deposits. It has no rift age, so ages must come from basin or deposit data.

### Hoggard et al. 2020 global lithospheric thickness (LAB) maps  
`hoggard2020_lab` · priority 2

- **Citation:** Hoggard et al. (2020) Nature Geoscience 13, 504-510; see also Czarnota et al. 2020 (GA eCat 'Minerals on the edge')
- **DOI:** 10.1038/s41561-020-0593-2
- **URL:** https://osf.io/twksd/ (folder lithospheric_thickness_maps; also paper supplement)
- **Format / licence:** netCDF, GeoTIFF, CSV / none stated on OSF (cite)
- **Records:** global grids (several tomography models)
- **Age field:** n/a (present day)
- **Type field:** n/a
- **Notes:** The 170 km LAB contour marks craton edges. Present day only, so it must be treated as a fixed feature that moves with plate IDs.

### Evenick 2021 revised global sedimentary basin map  
`evenick2021_basins` · priority 2

- **Citation:** Evenick, J.C. (2021) Earth-Science Reviews 215, 103564
- **DOI:** 10.1016/j.earscirev.2021.103564
- **URL:** https://africa-knowledge-platform.ec.europa.eu/node/45805 (Africa subset layer)
- **Format / licence:** shapefile (Africa subset); global layer not openly downloadable / not stated
- **Records:** not stated (basins >2 km sediment thickness)
- **Age field:** basin age (per the JRC description)
- **Type field:** basin class/type
- **Notes:** The global shapefile is not openly released; request it from the author. Open fallbacks: USGS potash basin tracts, Hasterok rift/basin provinces, and the USGS sed-Cu basin tracts.

### Brune et al. 2016 rift velocity database (web app)  
`brune2016_riftvel` · priority 3

- **Citation:** Brune, S., Williams, S.E., Butterworth, N.P., Müller, R.D. (2016) Abrupt plate accelerations shape rifted continental margins. Nature 536, 201-204
- **DOI:** 10.1038/nature18319
- **URL:** https://portal.gplates.org/cesium/?view=rift_v
- **Format / licence:** interactive; per-point velocity histories downloadable / -
- **Records:** rifts since 240 Ma
- **Age field:** time series
- **Type field:** -
- **Notes:** Only 240 Ma to present. Methodology reference: for 1.8 Ga, compute divergence velocities directly from muller2025 topologies with gplately.

### Müller et al. 2019 deforming plate model (rifts and orogens since the Triassic)  
`muller2019_deforming` · priority 3

- **Citation:** Müller, R.D., et al. (2019) Tectonics 38, 1884-1907
- **DOI:** 10.5281/zenodo.10525287 (v1.2)
- **URL:** https://zenodo.org/records/10525287
- **Format / licence:** GPlates zip (37.5 MB) / open (Zenodo)
- **Records:** -
- **Age field:** -
- **Type field:** deforming networks (extension/compression)
- **Notes:** Mesozoic-Cenozoic only. Use its rift network extents as a template for rift-zone masks.

### Macrostrat API (stratigraphic columns, time-scale intervals)  
`macrostrat` · priority 2

- **Citation:** Peters, S.E., Husson, J.M., Czaplewski, J. (2018) G-cubed 19, 1393
- **DOI:** 10.1029/2018GC007467
- **URL:** https://macrostrat.org/api/v2/
- **Format / licence:** JSON/CSV API / CC BY 4.0
- **Records:** -
- **Age field:** t_age/b_age (Ma)
- **Type field:** lithology, environment
- **Notes:** Use /defs/intervals to convert text ages (USGS potash, phosphate, sed-Cu) to numeric Ma. Column coverage is mainly North America.

## Other deposit, geochemistry and contrast sources

### USGS Porphyry copper deposits of the world  
`usgs_porcu` · priority 3

- **Citation:** Singer, D.A., Berger, V.I., Moring, B.C. (2008) USGS OFR 2008-1155
- **DOI:** 10.3133/ofr20081155
- **URL:** https://mrdata.usgs.gov/porcu/porcu-csv.zip (267 KB)
- **Format / licence:** CSV + SHP / US public domain
- **Records:** 690
- **Age field:** agemy (630 filled), depage
- **Type field:** deptype (code), subtype via comments
- **Notes:** Convergent-margin contrast class.

### USGS World phosphate mines, deposits and occurrences  
`usgs_phosphate` · priority 3

- **Citation:** Chernoff, C.B., Orris, G.J. (2002) USGS OFR 2002-156
- **DOI:** 10.3133/ofr02156
- **URL:** https://mrdata.usgs.gov/phosphate/phosphate-csv.zip (239 KB)
- **Format / licence:** CSV + SHP / US public domain
- **Records:** 1635
- **Age field:** min_age_g (text, 1129 filled), min_age_s (stage, 349), host_age
- **Type field:** dep_type (marine chemical sediment, igneous...), sub_type
- **Notes:** Phosphorites; the igneous subset overlaps with carbonatites.

### USGS Global distribution of selected mines, deposits and districts of critical minerals (PP 1802 GIS)  
`usgs_pp1802` · priority 2

- **Citation:** Labay, K., Burger, M.H., Bellora, J.D., et al. (2017) USGS data release
- **DOI:** 10.5066/F7GH9GQR
- **URL:** https://mrdata.usgs.gov/pp1802/ (ScienceBase gdb/shapefile links; item https://data.usgs.gov/datacatalog/data/USGS:594d3c8ee4b062508e39b332)
- **Format / licence:** File geodatabase / shapefiles / US public domain
- **Records:** not verified (23 commodities incl. Li, Mn, REE, Nb, Co, PGE, Ba, fluorite)
- **Age field:** not verified
- **Type field:** deposit type per commodity chapter
- **Notes:** Main open global source for lithium (brine, pegmatite, clay) and manganese sites. ScienceBase file downloads returned 403 from this sandbox's proxy; the landing pages respond.

### USGS index to PGE-Ni-Cr deposits and occurrences  
`usgs_nicrpge` · priority 3

- **Citation:** Causey, J.D., Galloway, J.P., Zientek, M.L. (2009) USGS OFR 2009-1045
- **DOI:** 10.3133/ofr20091045
- **URL:** https://mrdata.usgs.gov/nicrpge/nicrpge-csv.zip (1.4 MB)
- **Format / licence:** CSV + SHP / US public domain
- **Records:** 17954
- **Age field:** none
- **Type field:** class, record_tp
- **Notes:** Bibliographic index only, with no ages. For magmatic Ni with ages, use Hoggard 2020 'Magmatic Ni' (108 records).

### USGS Major mineral deposits of the world  
`usgs_major_deposits` · priority 3

- **Citation:** Schulz, K.J., Briskey, J.A. (2005) USGS OFR 2005-1294
- **DOI:** none found
- **URL:** https://mrdata.usgs.gov/major-deposits/ofr20051294-csv.zip
- **Format / licence:** CSV + SHP / US public domain
- **Records:** 3168
- **Age field:** none
- **Type field:** dep_type, type_detail, model_code, model_name
- **Notes:** Only about 372 rows have a model code. Useful for locations and commodities.

### USGS Mineral Resources Data System (MRDS)  
`usgs_mrds` · priority 3

- **Citation:** U.S. Geological Survey (2005) Mineral Resources Data System, Reston VA
- **DOI:** none
- **URL:** https://mrdata.usgs.gov/mrds/mrds-csv.zip (25.8 MB)
- **Format / licence:** CSV / tab-delimited RDBMS / US public domain
- **Records:** ~300k sites (global, US-heavy)
- **Age field:** no systematic mineralisation age in the flat CSV (the relational tables carry some host-rock age data)
- **Type field:** dep_type, model (sparse)
- **Notes:** Legacy database, no longer maintained, with uneven quality. For the US, USMIN supersedes it.

### USGS USMIN Mineral Deposit Database  
`usgs_usmin` · priority 3

- **Citation:** USGS USMIN (Horton et al., various data releases)
- **DOI:** various (per commodity release)
- **URL:** https://mrdata.usgs.gov/usmin/ and https://mrdata.usgs.gov/deposit/
- **Format / licence:** gdb/shp/CSV per state and per commodity / US public domain
- **Records:** US only
- **Age field:** some commodity releases include age fields
- **Type field:** deposit type
- **Notes:** US only, so of limited use for a global analysis.

### USGS Ni-Co laterite deposits of the world  
`usgs_laterite` · priority 3

- **Citation:** Berger, V.I., Singer, D.A., Bliss, J.D., Moring, B.C. (2011) USGS OFR 2011-1058
- **DOI:** 10.3133/ofr20111058
- **URL:** https://mrdata.usgs.gov/laterite/
- **Format / licence:** CSV + SHP / US public domain
- **Records:** not checked
- **Age field:** -
- **Type field:** -
- **Notes:** Weathering deposits, not rift-related. Listed for completeness only.

### Geoscience Australia OZMIN Mineral Deposits (data.gov.au snapshot)  
`ga_ozmin` · priority 3

- **Citation:** Geoscience Australia (2013) OZMIN Mineral Deposits Database v02
- **DOI:** —
- **URL:** https://data.gov.au/data/dataset/8e8457e0-a66f-44ad-90b7-ffe795cbc94e/resource/847c9bfd-2db5-45cd-98bc-0fbbedb4ca79/download/34247a24-d3cf-4a98-bb9d-81671ddb99de.zip (122 KB)
- **Format / licence:** SHP/DBF + XLSX / CC BY 3.0 AU
- **Records:** small 2012 resources snapshot
- **Age field:** not in snapshot
- **Type field:** commodity/deposit
- **Notes:** The full OZMIN (with mineral-system and age info) is served through GA Portal/WFS (https://portal.ga.gov.au/). Use it for Australian deposits (e.g. McArthur/Mt Isa).

### IAEA UDEPO / UThDEPO world uranium and thorium deposits  
`udepo` · priority 3

- **Citation:** IAEA (2020) World Distribution of Uranium Deposits, 3rd ed.; IAEA-TECDOC-1629
- **DOI:** —
- **URL:** https://infcis.iaea.org/UTHDEPO
- **Format / licence:** online database (web queries, reports) / IAEA terms of use
- **Records:** ~2000+ deposits
- **Age field:** not verified (classification yes)
- **Type field:** IAEA U-deposit type/subtype (e.g. sandstone, unconformity, intrusive)
- **Notes:** Unconformity and sandstone U deposits are basin-hosted. There is no simple bulk export, and coordinates are approximate.

### GEOROC geochemistry of igneous rocks  
`georoc` · priority 3

- **Citation:** GEOROC (DIGIS, Göttingen)
- **DOI:** —
- **URL:** https://georoc.eu/
- **Format / licence:** CSV downloads / API / CC BY-SA 4.0 (as stated by GEOROC; confirm)
- **Records:** >1 M analyses
- **Age field:** age (sparse)
- **Type field:** rock type, tectonic setting (includes 'Rift volcanics', 'Continental flood basalt')
- **Notes:** Can supply carbonatite and alkaline magmatism ages and settings, but with heterogeneous quality.

### Gard et al. 2019 global whole-rock geochemical database (includes age.csv)  
`gard2019_geochem` · priority 3

- **Citation:** Gard, M., Hasterok, D., Halpin, J. (2019) ESSD 11, 1553-1566
- **DOI:** 10.5194/essd-11-1553-2019; data 10.5281/zenodo.3359791
- **URL:** https://zenodo.org/record/3359791
- **Format / licence:** CSV (11 files, 797 MB) / open (Zenodo; check)
- **Records:** ~1 M samples
- **Age field:** age.csv (1.7 MB)
- **Type field:** rock type
- **Notes:** An alternative source for dated alkaline and carbonatite samples.

### USGS/GA/GSC CMMI Global Geochemical Database for Critical Minerals in Archived Mine Samples  
`usgs_cmmi_geochem` · priority 3

- **Citation:** USGS data release (CMMI); deposit classification: Hofstra et al. 2021 USGS OFR 2021-1049
- **DOI:** 10.3133/ofr20211049 (classification)
- **URL:** https://data.usgs.gov/datacatalog/data/USGS:5e616381e4b01d509255c7b6
- **Format / licence:** CSV/Excel / US public domain
- **Records:** thousands of ore samples
- **Age field:** no
- **Type field:** CMMI deposit type / mineral system (e.g. 'basin brine path', 'basin evaporite')
- **Notes:** The CMMI mineral-system classification (it has explicit basin-hosted systems) is a good vocabulary for harmonising deposit types across sources.

### Commercial or restricted sources (flag only)  
`commercial_flags` · priority 3

- **Citation:** S&P Capital IQ Pro (Metals & Mining); Mindat (API needs a key; data CC BY-NC-SA); Mining Data Online; BGS World Mineral Statistics (production only)
- **DOI:** —
- **URL:** https://www.bgs.ac.uk/mineralsuk/statistics/world-mineral-statistics/
- **Format / licence:** - / commercial / non-commercial
- **Records:** -
- **Age field:** S&P: no geological age; Mindat: locality ages sparse
- **Type field:** -
- **Notes:** Avoid these as primary sources in an open publication. The Mindat and S&P web hosts return 403 to automated requests.

## Gaps and recommendations

- **Mineralisation ages vs host-rock ages:** the USGS `agemy` fields (sedcu, sedznpb, SIR-5090-Z sed-Cu) are mostly host-rock stratigraphic ages. Hoggard 2020 is the only open table with an `Age Type` flag (direct vs inferred). Carry the age type as a covariate, and sample plate kinematics over a window (for example 10 Myr before the age, as in the EarthByte notebooks) rather than at a single time.
- **Harmonise and deduplicate** Hoggard 2020, USGS sedznpb, sedexmvt, sedcu and SIR-5090-Z by name plus a distance threshold of about 5 km. The EarthByte `Craton_Deposits` (265/289 records) is already a curated Hoggard subset with plate IDs.
- **Text-to-numeric ages:** convert them with the Macrostrat `/defs/intervals` endpoint (potash `max_age`/`min_age`, phosphate, REE `age_info`).
- **No open global rift-age database was found.** Combine Hasterok rift and passive-margin provinces with basin tracts (USGS potash, sed-Cu) and derive rift activity from `muller2025` topologies (divergent boundary velocities and deforming extent) with gplately.
- **Carbonatites:** the Humphreys-Williams & Zahirovic 2021 compilation is public (Zenodo 10.5281/zenodo.5968095) and in use (scripts/89).
- **Kimberlites:** open Gernon et al. 2023 Zenodo 7849142 in a browser. It is the most direct rift–deep-magmatism dataset.
- **Li and Mn:** use USGS PP1802 GIS (download manually; ScienceBase was blocked here). No open global dated Li-brine or Mn compilation was found.
- **Flag:** `github.com/EarthByte/Craton_Deposits` returns 404 (private or renamed). The same inputs exist in `Craton-deposits_Tectonic-evolution` and `Sed-hosted-Spatio-Temporal-MPM`.


## Magmatic compilations in use

| Class | Source | Records | Dated | Notes |
|---|---|---|---|---|
| Kimberlite | Gernon, T.M., Jones, S.M., Brune, S., Hincks, T.K., Palmer, M.R., Schumacher, J.C., Primiceri, R.M., Field, M., Griffin, W.L., Pearson, N.J., Howarth, G., Glerum, A. (2023) Rift-induced disruption of cratonic keels drives kimberlite volcanism. *Nature* 620, 344–350; supplementary data Zenodo 10.5281/zenodo.7849142 (CC-BY 4.0) | 1,141 localities | all, with the mineral dated and the isotopic system | `data/raw/magmatic/kimberlites_Gernon_etal_2023.csv`; perovskite U-Th-Pb for 372, phlogopite Rb-Sr or Ar-Ar for 418, zircon for 203 |
| Kimberlite | Giuliani, A. & Pearson, D.G. (2019) Kimberlites: from deep Earth to diamond mines. *Elements* 15, 377–380, supplementary locality database (screened from Faure, 2010) | 3,468 | 527 measured; 400 estimated from cited age ranges | `data/raw/magmatic/kimberlites_Giuliani_Pearson_2019.xlsx` |
| Carbonatite | Woolley, A.R. & Kjarsgaard, B.A. (2008) Carbonatite occurrences of the world: map and database. Geological Survey of Canada Open File 5796, doi:10.4095/225115 | 527 | 274 | `data/raw/magmatic/carbonatites_Woolley_Kjarsgaard_2008.xls` |
| Lamproite | GEOROC (accessed August 2023), dated lamproite samples | 197 samples at 52 locations | all | `data/raw/magmatic/lamproites_georoc_2023.csv` |
| Lamproite, orangeite | Tappe, S. (2022) Global cratonic lamproite-orangeite and related potassic rock compositions, Göttingen Research Online, doi:10.25625/JEPY8L (CC-BY 4.0) | 429 + 179 | 52 orangeites (approximate) | locations for a future lamproite class |

## Subducted volatile grids

The grids themselves are unpublished. What is published is the methodology that produces
them: the water workflow in Mather et al. (2026), Communications Earth & Environment 7, 48,
doi:10.1038/s43247-025-03097-0, and the carbon workflow in Muller et al. (2024),
Geochemistry Geophysics Geosystems 25, e2024GC011713, doi:10.1029/2024GC011713. The grids
used here were computed with those workflows in the Muller2025 mantle frame back to 1.8 Ga,
so they are a product of this work and are released with the papers rather than cited to an
existing archive. Both papers must say so, and neither may cite the grids as published data.

## Subducted-carbon grids

Two sets are used. The primary set sums all plate reservoirs, is smoothed over 300 km like
the water grids and spans 0–1500 Ma (1,501 grids, 0.2°, t/m²). The second set is smoothed
over 500 km and spans 0–1800 Ma, with separate reservoirs:

Cumulative subducted carbon in the Muller2025 mantle frame, 0.2°, smoothed over 500 km,
min/mean/max, for the reservoirs Mantle (serpentinite, oceanic crust and lithosphere;
0–1800 Ma), Crust (0–1800 Ma), carbonate sediment (0–239 Ma) and organic sediment
(0–599 Ma); t/m². Built from Muller2025-frame seafloor-age and spreading-rate grids.


## Exploration and ice masks

| Use | Source | File |
|---|---|---|
| Present-day ice sheets, caps and glaciers | Natural Earth 10 m physical, glaciated areas (public domain) | `data/external/ne_10m_glaciated_areas.shp` |
| Antarctic ice shelves | Natural Earth 10 m physical, Antarctic ice shelves (public domain) | `data/external/ne_10m_antarctic_ice_shelves_polys.shp` |

## Lithospheric architecture

| Use | Source | File |
|---|---|---|
| Craton polygons, reconstructed to the age of each point | Shirmard et al. (2025), craton boundaries inferred from REVEAL tomography (Zenodo 10.5281/zenodo.15003746, GPL-3.0), cut along the Muller2025 static polygons by `scripts/44_shirmard_cratons.py` | `data/external/Craton_Boundaries_Shirmard2025/Craton_Boundaries_Inferred.shp` -> `data/external/cratons_Shirmard2025_partitioned.gpmlz` |
| Present-day total crustal thickness (Hc), and McKenzie's stretching factor against 32 km of unstretched crust, per point and along rift transects | Earth Crustal Model 1, ECM1 (Mooney et al. 2023, Earth-Science Reviews 243, 104493), 1 degree cells gridded to 0.1 degree by `scripts/51_ecm1_crust.py` | `data/external/Mooney2023_ECM1/Hc_ECM1.txt` -> `data/external/ECM1_Hc_Mooney2023_0p1deg.nc` |
| Present-day shear-wave velocity at 150 km and its gradient | REVEAL full-waveform tomography (Thrastarson et al. 2024), sampled at 150 km by Shirmard et al. (2025) and converted to NetCDF here | `data/external/reveal_vsh_150km.nc` |

The 20 Shirmard outlines are present-day polygons without plate IDs. `44_shirmard_cratons.py`
cuts each along the Muller2025 static polygons (static polygons valid only at 0 Ma left out;
where static polygons overlap, the one hit by most sample points takes precedence), giving 318
pieces on 84 plates. Distances are measured to the merged outer ring of each craton, so cut
lines are never counted as craton edges; present-day distances match the source outlines to
within 9 km. At least half the pieces reconstruct back to about 1,750 Ma; `20_craton.py`
reports the count at each time. The Merdith et al. outlines used up to draft v8
(`data/external/shapes_cratons_Merdith_etal.gpml`) are superseded.

The explored footprint is not a dataset. It is computed in `scripts/17_null_masks.py` as the
distance from each point to the nearest deposit of another family in the compilation itself,
so that the footprint of a class is never defined by deposits found while looking for that
class; the radius is `nulls.explored_radius_km` in the configuration, and the script prints
the share of each class's own deposits inside the radius, which is what makes a radius usable.
