"""Large low-shear-velocity provinces (LLSVPs) from the agreement of modern tomographic models.

The provinces are taken from the vote map of the plume study (Muller_mantle_tomography_subduction,
Paper_plumes, province_agreement.py), written to data/external/LLSVP_vote_4models/: in each of
REVEAL, GLAD-M35, SPiRaL and SEMUCB-WM1 (published models only) the
mean shear-velocity anomaly over 2,600-2,880 km depth is thresholded at its slowest 20 per cent,
connected regions of at least 60 one-degree cells are kept, and every cell counts the families in
which it is province. A province here is the set of cells agreed by at least ``votes`` of the four
families (three by default, as in the plume study); two and four bracket it. Single models place
the boundary a median of about 380 km apart (province_agreement.csv), the resolution to which a
distance from the edge is meaningful.

The provinces are held fixed in the mantle frame (Muller2025), and districts are placed at their
mantle-frame position at the age they formed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

R_E = 6371.0
VOTE_FILE = Path("data") / "external" / "LLSVP_vote_4models" / "province_vote.npz"


def _xyz(lon, lat):
    lo, la = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def _label_periodic(m):
    """Connected regions of a lat x lon mask, joined across the date line."""
    from scipy.ndimage import label
    lab, n = label(m)
    # merge labels that meet across the first and last longitude column
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in zip(lab[:, 0], lab[:, -1]):
        if a and b:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(k) for k in range(n + 1)])
    return roots[lab]


class Provinces:
    """Province mask at one vote level, with signed distance from its edge."""

    def __init__(self, root, votes=3, min_cells=60, path=None):
        """Load the vote grid of the LLSVP families and keep the connected provinces (longitude wraps) of
        at least ``min_cells`` cells where at least ``votes`` families agree. The edge cells are indexed
        for distance queries, and area_share is the share of the Earth's surface the provinces cover."""
        z = np.load(Path(root) / (path or VOTE_FILE))
        self.lat, self.lon, vote = z["lat"].astype(float), z["lon"].astype(float), z["vote"].astype(int)
        self.votes, self.n_families = votes, int(z["n_families"][0])
        lab = _label_periodic(vote >= votes)
        ids, counts = np.unique(lab[lab > 0], return_counts=True)
        self.mask = np.isin(lab, ids[counts >= min_cells])
        # inner boundary: province cells with a neighbour outside (longitude wraps, latitude does not)
        m = self.mask
        inner = np.roll(m, 1, 1) & np.roll(m, -1, 1)
        inner &= np.pad(m[:-1], ((1, 0), (0, 0)), constant_values=False)
        inner &= np.pad(m[1:], ((0, 1), (0, 0)), constant_values=False)
        edge = m & ~inner
        LA, LO = np.meshgrid(self.lat, self.lon, indexing="ij")
        from scipy.spatial import cKDTree
        self._tree = cKDTree(_xyz(LO[edge], LA[edge]))
        self.area_share = float((m * np.cos(np.radians(LA))).sum() / np.cos(np.radians(LA)).sum())

    def inside(self, lon, lat):
        lon = ((np.asarray(lon, float) + 180.0) % 360.0) - 180.0
        dlat, dlon = self.lat[1] - self.lat[0], self.lon[1] - self.lon[0]
        i = np.clip(np.round((np.asarray(lat, float) - self.lat[0]) / dlat).astype(int), 0, len(self.lat) - 1)
        j = np.round((lon - self.lon[0]) / dlon).astype(int) % len(self.lon)
        return self.mask[i, j]

    def signed_distance(self, lon, lat):
        """Great-circle distance (km) to the nearest province edge; negative inside, NaN where unknown."""
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        out = np.full(len(lon), np.nan)
        ok = np.isfinite(lon) & np.isfinite(lat)
        if ok.any():
            chord, _ = self._tree.query(_xyz(lon[ok], lat[ok]))
            d = 2 * np.arcsin(np.clip(chord / 2, 0, 1)) * R_E
            out[ok] = np.where(self.inside(lon[ok], lat[ok]), -d, d)
        return out
