"""Tests of AUC = 0.5 from the district bootstrap, and corrections for testing many at once.

A study that measures many things and reports the ones whose interval excludes 0.5 will find
some by chance. The correction is applied within a family - the set of tests that answer one
question - because that is the unit a reader weighs: the lithospheric measures together answer
whether kimberlites sit on distinctive lithosphere, and the rift-lag windows together answer
whether they sit near rifts of some antiquity. Holm's step-down procedure controls the chance
of any false positive in the family; the Benjamini-Hochberg procedure controls the expected
share of false positives among those reported, and is given alongside because it is the less
conservative of the two and the one most readers will recognise.
"""
from __future__ import annotations

import numpy as np


def boot_p(boot, null=0.5):
    """Two-sided percentile-bootstrap p-value for H0: AUC = ``null``.

    The bootstrap distribution is centred on the observed AUC, so the share of replicates on
    the far side of ``null`` estimates how often a separation this large would vanish under
    resampling. The floor is 1 / (n + 1): a finite bootstrap cannot certify a smaller value.
    """
    b = np.asarray(boot, float)
    b = b[np.isfinite(b)]
    if len(b) == 0:
        return np.nan
    lo = np.mean(b <= null)
    hi = np.mean(b >= null)
    return float(max(min(1.0, 2.0 * min(lo, hi)), 1.0 / (len(b) + 1)))


def holm(p):
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def bh(p):
    """Benjamini-Hochberg adjusted p-values (q-values), returned in the input order."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    q = np.empty(m)
    prev = 1.0
    for k in range(m - 1, -1, -1):
        prev = min(prev, p[order[k]] * m / (k + 1))
        q[k] = prev
    out[order] = np.minimum(q, 1.0)
    return out
