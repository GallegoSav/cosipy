"""
Numerical core used by
:class:`cosipy.response.relative_irf_detector_response.RelativeIRFDetectorResponse`
to integrate a relative-coordinate differential response
(:class:`cosipy.response.relative_irf_hist.IRFRelativeHistUnpolarized`)
over the bins of an absolute Compton Data Space (Em, Phi, PsiChi).

This module only depends on numpy and scipy, so the index bookkeeping
can be tested in isolation. The interpolation weights along each
relative axis are passed in as callables, so the same multilinear
model as :meth:`histpy.Axis.interp_weights` /
:meth:`histpy.Histogram.interp` is used.
"""

from typing import Callable, Tuple

import numpy as np
from scipy import sparse

__all__ = ["build_transfer_matrix", "integrate_em_bins", "subsample_edges"]


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
        Width represented by each sample (bin width / nsamples), so
        that ``sum(f(values) * widths)`` over one bin approximates the
        integral of ``f`` over that bin.
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


def build_transfer_matrix(phi_geo: np.ndarray,
                          zeta: np.ndarray,
                          psichi_bin: np.ndarray,
                          psichi_weight: np.ndarray,
                          phi_samples: np.ndarray,
                          phi_bin: np.ndarray,
                          phi_weight: np.ndarray,
                          n_psichi: int,
                          n_phi: int,
                          irf_shape: Tuple[int, int, int],
                          theta_range: Tuple[float, float],
                          phi_range: Tuple[float, float],
                          phi_interp_weights: Callable,
                          theta_interp_weights: Callable,
                          zeta_interp_weights: Callable,
                          max_pairs: int = 2_000_000) -> sparse.csr_matrix:
    """
    Sparse linear map from the relative (Phi, Theta, Zeta) bins of the
    IRF to the absolute (Phi, PsiChi) bins of the measurement, for one
    fixed source direction.

    For each pair (Phi sample, PsiChi sub-sample) the relative
    coordinates are ``Theta = phi_geo - Phi`` and ``Zeta``. The
    differential response at that point is the multilinear
    interpolation of the IRF bins (8 corners), and its contribution to
    the absolute bin is weighted by the phase-space volume represented
    by the pair (``phi_weight * psichi_weight``, i.e. rad * sr).
    Therefore, if ``D`` holds the IRF differential response with the
    (Phi, Theta, Zeta) axes flattened in C order (last axis), then
    ``D @ T`` is the response integrated over each (Phi, PsiChi) bin,
    flattened as ``phi_bin * n_psichi + psichi_bin``.

    Pairs whose ``Theta`` or ``Phi`` fall outside the IRF axis edges
    are dropped (zero response), rather than clamped.

    Parameters
    ----------
    phi_geo, zeta : numpy.ndarray, shape (nsub,)
        Geometric angular distance to the source direction and
        azimuth around it (rad, zeta in [0, 2pi)) of each PsiChi
        sub-sample.
    psichi_bin : numpy.ndarray of int, shape (nsub,)
        Absolute PsiChi bin each sub-sample belongs to.
    psichi_weight : numpy.ndarray, shape (nsub,)
        Solid angle (sr) represented by each sub-sample.
    phi_samples, phi_bin, phi_weight : numpy.ndarray, shape (nphis,)
        Kinematic scattering angle sub-samples (rad), the absolute
        Phi bin they belong to, and the width (rad) they represent.
    n_psichi, n_phi : int
        Number of absolute PsiChi and Phi bins.
    irf_shape : tuple of int
        Number of IRF bins along (Phi, Theta, Zeta).
    theta_range, phi_range : tuple of float
        (lower, upper) edges of the IRF Theta and Phi axes (rad).
    phi_interp_weights, theta_interp_weights, zeta_interp_weights : callable
        ``f(values) -> (bins, weights)`` with shapes (2, N), as
        :meth:`histpy.Axis.interp_weights`.
    max_pairs : int
        Maximum number of pairs processed at once, to bound memory.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (prod(irf_shape), n_phi * n_psichi)
    """

    n_phi_irf, n_theta_irf, n_zeta_irf = irf_shape
    n_rows = n_phi_irf * n_theta_irf * n_zeta_irf
    n_cols = n_phi * n_psichi

    phi_geo = np.asarray(phi_geo, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    psichi_bin = np.asarray(psichi_bin)
    psichi_weight = np.asarray(psichi_weight, dtype=float)

    phi_samples = np.asarray(phi_samples, dtype=float)
    phi_bin = np.asarray(phi_bin)
    phi_weight = np.asarray(phi_weight, dtype=float)

    # Phi samples outside the IRF range carry no response
    phi_ok = (phi_samples >= phi_range[0]) & (phi_samples <= phi_range[1])
    phi_samples = phi_samples[phi_ok]
    phi_bin = phi_bin[phi_ok]
    phi_weight = phi_weight[phi_ok]

    transfer = sparse.csr_matrix((n_rows, n_cols), dtype=float)

    if len(phi_samples) == 0 or len(phi_geo) == 0:
        return transfer

    # Interpolation along Phi only depends on the sample, not on the pair
    phi_ibins, phi_iweights = phi_interp_weights(phi_samples)
    phi_ibins = np.asarray(phi_ibins)
    phi_iweights = np.asarray(phi_iweights)

    # Zeta only depends on the PsiChi sub-sample
    zeta_ibins, zeta_iweights = zeta_interp_weights(zeta)
    zeta_ibins = np.asarray(zeta_ibins)
    zeta_iweights = np.asarray(zeta_iweights)

    # Process blocks of Phi samples so the (nphis, nsub) mask fits in memory
    block = max(1, int(max_pairs // max(1, len(phi_geo))))

    for start in range(0, len(phi_samples), block):

        sl = slice(start, start + block)

        theta = phi_geo[None, :] - phi_samples[sl, None]

        mask = (theta >= theta_range[0]) & (theta <= theta_range[1])

        i_phi, i_sub = np.nonzero(mask)

        if len(i_phi) == 0:
            continue

        i_phi_global = i_phi + start

        theta_ibins, theta_iweights = theta_interp_weights(theta[i_phi, i_sub])
        theta_ibins = np.asarray(theta_ibins)
        theta_iweights = np.asarray(theta_iweights)

        pair_weight = phi_weight[i_phi_global] * psichi_weight[i_sub]
        cols = phi_bin[i_phi_global] * n_psichi + psichi_bin[i_sub]

        rows_list = []
        data_list = []

        for a in range(2):
            for b in range(2):
                for c in range(2):
                    rows_list.append((phi_ibins[a][i_phi_global] * n_theta_irf
                                      + theta_ibins[b]) * n_zeta_irf
                                     + zeta_ibins[c][i_sub])
                    data_list.append(phi_iweights[a][i_phi_global]
                                     * theta_iweights[b]
                                     * zeta_iweights[c][i_sub]
                                     * pair_weight)

        rows = np.concatenate(rows_list)
        data = np.concatenate(data_list)
        cols8 = np.tile(cols, 8)

        # Duplicates are summed when converting to CSR
        transfer = transfer + sparse.coo_matrix((data, (rows, cols8)),
                                                shape=(n_rows, n_cols)).tocsr()

    return transfer


def integrate_em_bins(density: np.ndarray,
                      ei: float,
                      em_edges: np.ndarray,
                      eps_centers: np.ndarray,
                      eps_edges: np.ndarray,
                      integrate_piecewise_linear: Callable) -> np.ndarray:
    """
    Integrate a response that is differential in measured energy over
    each measured energy (Em) bin, for a fixed photon energy ``ei``.

    The response is known as a function of ``Epsilon = Em/Ei - 1``
    (piecewise linear between ``eps_centers``, flat up to
    ``eps_edges``, zero outside), so
    ``int f dEm = Ei * int f dEpsilon`` over
    ``[Em_lo/Ei - 1, Em_hi/Ei - 1]`` clipped to the Epsilon edges.

    Parameters
    ----------
    density : numpy.ndarray, shape (..., n_eps)
        Response per unit Em (keV^-1), as a function of Epsilon along
        the last axis.
    ei : float
        Photon energy (keV).
    em_edges : numpy.ndarray
        Measured energy edges (keV).
    eps_centers, eps_edges : numpy.ndarray
        IRF Epsilon axis centers and edges.
    integrate_piecewise_linear : callable
        ``f(values, x_centers, x_lo, x_hi)``, e.g.
        :meth:`IRFRelativeHistUnpolarized._integrate_piecewise_linear`.

    Returns
    -------
    numpy.ndarray, shape (n_em, ...)
    """

    em_edges = np.asarray(em_edges, dtype=float)

    out = np.zeros((len(em_edges) - 1,) + density.shape[:-1])

    if ei <= 0:
        return out

    for m, (lo, hi) in enumerate(zip(em_edges[:-1], em_edges[1:])):

        eps_lo = np.clip(lo / ei - 1, eps_edges[0], eps_edges[-1])
        eps_hi = np.clip(hi / ei - 1, eps_edges[0], eps_edges[-1])

        if eps_hi <= eps_lo:
            continue

        out[m] = ei * integrate_piecewise_linear(density, eps_centers, eps_lo, eps_hi)

    return out
