"""
Numerical core used by
:class:`cosipy.response.relative_irf_detector_response.RelativeIRFDetectorResponse`
to redistribute the bin contents of a relative-coordinate response
(:class:`cosipy.response.relative_irf_hist.IRFRelativeHistUnpolarized`)
into the bins of an absolute Compton Data Space (Em, Phi, PsiChi).

The redistribution is content conserving: the response density is
assumed constant within each relative bin, and each bin content is
split between the absolute bins in proportion to their overlap. No
interpolation between relative bins is performed.

This module only depends on numpy and scipy, so it can be tested in
isolation.
"""

from typing import Tuple

import numpy as np
from scipy import sparse

__all__ = ["subsample_edges", "find_bins", "overlap_matrix", "em_overlap_fractions"]


def subsample_edges(edges: np.ndarray, nsamples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Midpoint-rule sub-samples of each bin defined by ``edges``.

    Parameters
    ----------
    edges : numpy.ndarray
        Strictly increasing bin edges, shape (nbins + 1,).
    nsamples : int
        Number of equally spaced samples per bin (>= 1).

    Returns
    -------
    values : numpy.ndarray, shape (nbins * nsamples,)
        Sample positions, grouped by bin.
    bins : numpy.ndarray of int, shape (nbins * nsamples,)
        Index of the bin each sample belongs to.
    widths : numpy.ndarray, shape (nbins * nsamples,)
        Width represented by each sample (bin width / nsamples).
    """

    edges = np.asarray(edges, dtype=float)

    if nsamples < 1:
        raise ValueError("nsamples must be >= 1")

    lo = edges[:-1]
    widths = np.diff(edges)

    frac = (np.arange(nsamples) + 0.5) / nsamples

    values = (lo[:, None] + widths[:, None] * frac[None, :]).ravel()
    bins = np.repeat(np.arange(len(lo)), nsamples)
    sample_widths = np.repeat(widths / nsamples, nsamples)

    return values, bins, sample_widths


def find_bins(edges: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Index of the bin containing each value, -1 if outside the edges.
    Bins are closed on the left, open on the right, except the last one,
    which also includes the upper edge.
    """

    edges = np.asarray(edges, dtype=float)
    values = np.asarray(values, dtype=float)

    idx = np.searchsorted(edges, values, side='right') - 1

    # Include the upper edge in the last bin
    idx = np.where(values == edges[-1], len(edges) - 2, idx)

    return np.where((idx >= 0) & (idx < len(edges) - 1), idx, -1)


def overlap_matrix(sample_rel_bin: np.ndarray,
                   sample_abs_bin: np.ndarray,
                   sample_weight: np.ndarray,
                   n_rel: int,
                   n_abs: int,
                   fallback_abs_bin: np.ndarray = None) -> sparse.csr_matrix:
    """
    Fraction of each relative bin that falls in each absolute bin,
    estimated from weighted samples.

    Each sample is a point of the measurement space that belongs to
    the relative bin ``sample_rel_bin`` and to the absolute bin
    ``sample_abs_bin`` (-1 if it is outside of all absolute bins, e.g.
    outside of the measured Phi range), and represents the phase-space
    volume ``sample_weight``. The fraction of relative bin ``r`` in
    absolute bin ``a`` is the weight of the samples in both, divided
    by the weight of all samples in ``r`` (including those outside of
    the absolute bins). Each row therefore sums to 1 minus the fraction
    of the bin outside of the absolute bins, so that ``contents @ M``
    conserves the content within the measured range exactly.

    Relative bins without any sample (smaller than the sampling) are
    fully assigned to ``fallback_abs_bin[r]``, typically the absolute
    bin containing their center, if it is >= 0.

    Parameters
    ----------
    sample_rel_bin : numpy.ndarray of int, shape (nsamples,)
    sample_abs_bin : numpy.ndarray of int, shape (nsamples,)
    sample_weight : numpy.ndarray, shape (nsamples,)
    n_rel, n_abs : int
        Number of relative and absolute bins.
    fallback_abs_bin : numpy.ndarray of int, shape (n_rel,), optional

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n_rel, n_abs)
    """

    sample_rel_bin = np.asarray(sample_rel_bin, dtype=np.int64)
    sample_abs_bin = np.asarray(sample_abs_bin, dtype=np.int64)
    sample_weight = np.asarray(sample_weight, dtype=float)

    total_weight = np.bincount(sample_rel_bin, weights=sample_weight, minlength=n_rel)

    inside = sample_abs_bin >= 0

    rows = sample_rel_bin[inside]
    cols = sample_abs_bin[inside]
    data = sample_weight[inside] / total_weight[rows]

    if fallback_abs_bin is not None:

        fallback_abs_bin = np.asarray(fallback_abs_bin, dtype=np.int64)

        unsampled = np.nonzero((total_weight == 0) & (fallback_abs_bin >= 0))[0]

        rows = np.concatenate([rows, unsampled])
        cols = np.concatenate([cols, fallback_abs_bin[unsampled]])
        data = np.concatenate([data, np.ones(len(unsampled))])

    # Duplicates are summed when converting to CSR
    return sparse.coo_matrix((data, (rows, cols)), shape=(n_rel, n_abs)).tocsr()


def em_overlap_fractions(ei: float, eps_edges: np.ndarray, em_edges: np.ndarray) -> np.ndarray:
    """
    Fraction of each Epsilon bin that falls in each measured energy
    (Em) bin, for a fixed photon energy ``ei``.

    For a fixed ``Ei``, ``Em = Ei * (1 + Epsilon)`` is linear in
    ``Epsilon``, so a density constant within an Epsilon bin is also
    constant in Em over the interval ``[Ei*(1+eps_lo), Ei*(1+eps_hi)]``,
    and the fractions are exact.

    Parameters
    ----------
    ei : float
        Photon energy (keV).
    eps_edges : numpy.ndarray, shape (n_eps + 1,)
        Epsilon edges.
    em_edges : numpy.ndarray, shape (n_em + 1,)
        Measured energy edges (keV).

    Returns
    -------
    numpy.ndarray, shape (n_eps, n_em)
        Each row sums to the fraction of the Epsilon bin within the Em
        range (1 if fully inside).
    """

    eps_edges = np.asarray(eps_edges, dtype=float)
    em_edges = np.asarray(em_edges, dtype=float)

    lo = ei * (1 + eps_edges[:-1])
    hi = ei * (1 + eps_edges[1:])

    overlap = (np.minimum(hi[:, None], em_edges[None, 1:])
               - np.maximum(lo[:, None], em_edges[None, :-1]))

    overlap = np.clip(overlap, 0, None)

    width = hi - lo

    with np.errstate(invalid='ignore', divide='ignore'):
        fractions = np.where(width[:, None] > 0, overlap / width[:, None], 0)

    return fractions
