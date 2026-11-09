import numpy as np
import pytest

from cosipy.response._relative_irf_binning import build_transfer_matrix, integrate_em_bins, subsample_edges


def _interp_weights(edges):
    """Linear interpolation between bin centers, flat beyond the first/last
    center -- the model of histpy.Axis.interp_weights."""

    centers = 0.5 * (edges[1:] + edges[:-1])

    def f(values):
        values = np.asarray(values, dtype=float)
        idx = np.clip(np.searchsorted(centers, values) - 1, 0, len(centers) - 2)
        w1 = np.clip((values - centers[idx]) / (centers[idx + 1] - centers[idx]), 0, 1)
        return np.array([idx, idx + 1]), np.array([1 - w1, w1])

    return f


def _numeric_piecewise_integral(values, x_centers, x_lo, x_hi):
    x = np.linspace(x_lo, x_hi, 200001)
    v = np.stack([np.interp(x, x_centers, row) for row in values.reshape(-1, values.shape[-1])])
    return np.trapezoid(v, x, axis=-1).reshape(values.shape[:-1])


def test_subsample_edges():

    values, bins, widths = subsample_edges(np.array([0., 1., 3.]), 2)

    np.testing.assert_allclose(values, [0.25, 0.75, 1.5, 2.5])
    np.testing.assert_array_equal(bins, [0, 0, 1, 1])
    np.testing.assert_allclose(widths, [0.5, 0.5, 1, 1])

    with pytest.raises(ValueError):
        subsample_edges(np.array([0., 1.]), 0)


@pytest.mark.parametrize('max_pairs', [10 ** 9, 700])
def test_transfer_matrix_brute_force(max_pairs):

    rng = np.random.default_rng(1)

    phi_edges = np.linspace(0, np.pi, 7)
    theta_edges = np.linspace(-1, 1, 6)
    zeta_edges = np.linspace(0, 2 * np.pi, 9)
    shape = (6, 5, 8)

    irf = rng.random(shape)

    nsub, n_psichi, n_phi = 500, 12, 4

    phi_geo = rng.uniform(0, np.pi, nsub)
    zeta = rng.uniform(0, 2 * np.pi, nsub)
    psichi_bin = rng.integers(0, n_psichi, nsub)
    psichi_weight = rng.random(nsub)

    # Last measured Phi bin partially outside the IRF range
    phis, phi_bin, phi_weight = subsample_edges(np.linspace(0, 1.1 * np.pi, n_phi + 1), 3)

    fphi, ftheta, fzeta = (_interp_weights(e) for e in (phi_edges, theta_edges, zeta_edges))

    transfer = build_transfer_matrix(phi_geo, zeta, psichi_bin, psichi_weight,
                                     phis, phi_bin, phi_weight,
                                     n_psichi, n_phi, shape,
                                     (theta_edges[0], theta_edges[-1]),
                                     (phi_edges[0], phi_edges[-1]),
                                     fphi, ftheta, fzeta, max_pairs)

    assert transfer.shape == (np.prod(shape), n_phi * n_psichi)

    got = irf.ravel() @ transfer

    expected = np.zeros(n_phi * n_psichi)

    for i, phi in enumerate(phis):

        if not phi_edges[0] <= phi <= phi_edges[-1]:
            continue

        for s in range(nsub):

            theta = phi_geo[s] - phi

            if not theta_edges[0] <= theta <= theta_edges[-1]:
                continue

            bp, wp = fphi([phi])
            bt, wt = ftheta([theta])
            bz, wz = fzeta([zeta[s]])

            value = sum(wp[a, 0] * wt[b, 0] * wz[c, 0] * irf[bp[a, 0], bt[b, 0], bz[c, 0]]
                        for a in range(2) for b in range(2) for c in range(2))

            expected[phi_bin[i] * n_psichi + psichi_bin[s]] += value * phi_weight[i] * psichi_weight[s]

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)


def test_integrate_em_bins():

    rng = np.random.default_rng(2)

    eps_edges = np.linspace(-0.5, 0.3, 9)
    eps_centers = 0.5 * (eps_edges[1:] + eps_edges[:-1])

    density = rng.random((3, 8))
    ei = 500.

    # First bin partially and last bin fully outside the Epsilon range
    em_edges = np.array([100., 240., 300., 420., 600., 660., 700.])

    got = integrate_em_bins(density, ei, em_edges, eps_centers, eps_edges, _numeric_piecewise_integral)

    assert got.shape == (6, 3)

    for m, (lo, hi) in enumerate(zip(em_edges[:-1], em_edges[1:])):

        em = np.linspace(lo, hi, 400001)
        eps = em / ei - 1
        inside = (eps >= eps_edges[0]) & (eps <= eps_edges[-1])

        values = np.stack([np.where(inside, np.interp(eps, eps_centers, d), 0) for d in density])

        np.testing.assert_allclose(got[m], np.trapezoid(values, em, axis=-1), rtol=1e-5, atol=1e-9)

    assert np.all(got[-1] == 0)
