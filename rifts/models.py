"""Plate-model access through GPlately's PlateModelManager.

Two versions of one plate model are used. ``muller2025`` places the plates in a mantle
reference frame and supplies absolute plate speeds, trench migration and the positions at
which slab-water grids are sampled. ``cao2024`` shares its relative plate motions but sits
in a palaeomagnetic frame and supplies palaeolatitude. Relative quantities (divergence,
convergence, obliquity, distance to a boundary) are frame-independent and are taken from
the mantle model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pygplates


@dataclass
class PlateModel:
    name: str
    data_dir: str
    anchor_plate: int = 0
    _snapshots: dict = field(default_factory=dict, repr=False)

    @cached_property
    def _local(self):
        """Files already in the download cache, so a run needs no network and is reproducible
        from the archived data alone."""
        from pathlib import Path
        root = Path(self.data_dir) / self.name
        if not root.is_dir():
            return None
        want = {"Rotations": (".rot",), "Topologies": (".gpml", ".gpmlz"),
                "StaticPolygons": (".gpml", ".gpmlz", ".shp"),
                "ContinentalPolygons": (".gpml", ".gpmlz", ".shp")}
        # each layer is taken from the local copy when it is there, so a model cached without
        # topologies (the palaeomagnetic one, used for plate IDs and palaeolatitude only) never
        # needs the PlateModelManager servers for the layers it does have
        out = {}
        for layer, ext in want.items():
            d = root / layer
            files = sorted(str(f) for f in d.glob("*") if f.suffix.lower() in ext) if d.is_dir() else []
            if files:
                out[layer] = files
        return out or None

    @cached_property
    def _pmm(self):
        from plate_model_manager import PlateModelManager
        return PlateModelManager().get_model(self.name, data_dir=str(self.data_dir))

    def _files(self, layer, getter):
        loc = self._local
        if loc and loc.get(layer):
            return loc[layer]
        return getter()

    @cached_property
    def rotation_model(self):
        return pygplates.RotationModel(
            self._files("Rotations", lambda: self._pmm.get_rotation_model()),
            default_anchor_plate_id=self.anchor_plate)

    @cached_property
    def topologies(self):
        return [pygplates.FeatureCollection(f)
                for f in self._files("Topologies", lambda: self._pmm.get_topologies())]

    @cached_property
    def static_polygons(self):
        return [pygplates.FeatureCollection(f)
                for f in self._files("StaticPolygons", lambda: self._pmm.get_static_polygons())]

    @cached_property
    def continental_polygons(self):
        return [pygplates.FeatureCollection(f) for f in
                self._files("ContinentalPolygons",
                            lambda: self._pmm.get_layer("ContinentalPolygons"))]

    # ------------------------------------------------------------------ plate IDs
    def assign_plate_ids(self, lon, lat):
        """Plate ID and valid-time window of the present-day static polygon under each point."""
        pts = [pygplates.PointOnSphere(float(la), float(lo)) for lo, la in zip(lon, lat)]
        partitioner = pygplates.PlatePartitioner(self.static_polygons, self.rotation_model)
        pid = np.zeros(len(pts), int)
        begin = np.full(len(pts), np.nan)
        end = np.full(len(pts), np.nan)
        for i, p in enumerate(pts):
            poly = partitioner.partition_point(p)
            if poly is not None:
                f = poly.get_feature()
                pid[i] = f.get_reconstruction_plate_id()
                b, e = f.get_valid_time()
                begin[i] = float(b) if np.isfinite(float(b)) else 4600.0
                end[i] = float(e) if np.isfinite(float(e)) else 0.0
        return pid, begin, end

    # ------------------------------------------------------------ reconstruction
    def _rot(self, pid, time):
        """Total rotation of ``pid`` to ``time``; None where the rotation file has no entry."""
        if int(pid) == 0:
            return None
        return self.rotation_model.get_rotation(float(time), int(pid),
                                                use_identity_for_missing_plate_ids=False)

    def reconstruct_points(self, lon, lat, pids, time):
        """Rigidly reconstruct present-day points to ``time`` (Ma).

        Points whose plate has no rotation at ``time`` are returned as NaN. Whether the
        static polygon itself exists at ``time`` is a separate test (see
        ``assign_plate_ids``), recorded by the callers as a validity flag.
        """
        out = np.full((len(lon), 2), np.nan)
        cache = {}
        for i, (lo, la, pid) in enumerate(zip(lon, lat, pids)):
            if pid not in cache:
                cache[pid] = self._rot(pid, time)
            if cache[pid] is None:
                continue
            p = cache[pid] * pygplates.PointOnSphere(float(la), float(lo))
            out[i] = p.to_lat_lon()[::-1]
        return out[:, 0], out[:, 1]

    def point_velocities(self, rlon, rlat, pids, time, delta=1.0):
        """Velocity (cm/yr) of reconstructed points moving with their plates, in the anchor frame.

        Returns east, north, magnitude and azimuth (degrees clockwise from north).
        """
        n = len(rlon)
        ve, vn = np.full(n, np.nan), np.full(n, np.nan)
        cache = {}
        for i, (lo, la, pid) in enumerate(zip(rlon, rlat, pids)):
            if not np.isfinite(lo):
                continue
            if pid not in cache:
                cache[pid] = self.rotation_model.get_rotation(
                    float(time), int(pid), float(time) + delta,
                    use_identity_for_missing_plate_ids=False)
            if cache[pid] is None:
                continue
            p = pygplates.PointOnSphere(float(la), float(lo))
            v = pygplates.calculate_velocities([p], cache[pid], delta, pygplates.VelocityUnits.cms_per_yr)[0]
            enu = pygplates.LocalCartesian.convert_from_geocentric_to_north_east_down([p], [v])[0]
            vn[i], ve[i] = enu.get_x(), enu.get_y()
        mag = np.hypot(ve, vn)
        az = np.degrees(np.arctan2(ve, vn)) % 360
        return ve, vn, mag, az

    # ------------------------------------------------------------- snapshots
    def topological_snapshot(self, time):
        """pygplates TopologicalSnapshot at ``time``, cached; at most nine are kept, the oldest dropped
        first."""
        t = float(time)
        if t not in self._snapshots:
            if len(self._snapshots) > 8:
                self._snapshots.pop(next(iter(self._snapshots)))
            self._snapshots[t] = pygplates.TopologicalSnapshot(
                self.topologies, self.rotation_model, t, anchor_plate_id=self.anchor_plate)
        return self._snapshots[t]

    def continental_partitioner(self, time):
        rec = []
        pygplates.reconstruct(self.continental_polygons, self.rotation_model, rec, float(time),
                              anchor_plate_id=self.anchor_plate)
        geoms = [r.get_reconstructed_geometry() for r in rec]
        return pygplates.PlatePartitioner(rec, self.rotation_model) if geoms else None


def load_models(cfg):
    d = cfg["paths"]["pmm_data_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return (PlateModel(cfg["models"]["mantle"]["name"], d, cfg["models"]["mantle"]["anchor_plate"]),
            PlateModel(cfg["models"]["paleomag"]["name"], d, cfg["models"]["paleomag"]["anchor_plate"]))
