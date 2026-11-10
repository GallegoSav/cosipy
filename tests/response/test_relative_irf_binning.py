import numpy as np
import pytest

from cosipy.response._relative_irf_binning import (subsample_edges, find_bins, overlap_matrix,
                                                   em_overlap_fractions)


def test_subsample_edges():

    values, bins, widths = subsample_edges(np.array([0., 1., 3.]), 2)

    np.testing.assert_allclose(values, [0.25, 0.75, 1.5, 2.5])
    np.testing.assert_array_equal(bins, [0, 0, 1, 1])
    np.testing.assert_allclose(widths, [0.5, 0.5, 1, 1])

    with pytest.raises(ValueError):
        subsample_edges(np.array([0., 1.]), 0)


def test_find_bins():

    edges = np.array([0., 1., 2., 4.])

    np.testing.assert_array_equal(find_bins(edges, [-0.1, 0, 0.5, 1, 3.9, 4, 4.1]),
                                  [-1, 0, 0, 1, 2, 2, -1])


def test_overlap_matrix():

    # 3 relative bins, 4 absolute bins
    # rel 0: weights 1 (abs 0), 3 (abs 1)            -> 0.25, 0.75
    # rel 1: weights 2 (abs 2), 2 (outside, abs -1)  -> 0.5 (half outside the range)
    # rel 2: no samples, fallback to abs 3           -> 1
    rel = np.array([0, 0, 1, 1])
    absolute = np.array([0, 1, 2, -1])
    weight = np.array([1., 3., 2., 2.])

    m = overlap_matrix(rel, absolute, weight, 3, 4, fallback_abs_bin=np.array([-1, -1, 3])).toarray()

    np.testing.assert_allclose(m, [[0.25, 0.75, 0, 0],
                                   [0, 0, 0.5, 0],
                                   [0, 0, 0, 1]])

    # Without fallback the unsampled bin is dropped
    m = overlap_matrix(rel, absolute, weight, 3, 4).toarray()
    np.testing.assert_allclose(m[2], 0)


def test_overlap_matrix_conserves_content():

    rng = np.random.default_rng(0)

    n_rel, n_abs, n = 50, 30, 5000

    rel = rng.integers(0, n_rel, n)
    absolute = rng.integers(0, n_abs, n)
    weight = rng.random(n)

    m = overlap_matrix(rel, absolute, weight, n_rel, n_abs)

    contents = rng.random(n_rel)

    np.testing.assert_allclose((contents @ m).sum(), contents.sum())


def test_em_overlap_fractions():

    ei = 500.
    eps_edges = np.array([-0.5, -0.2, 0., 0.2])     # Em: [250, 400, 500, 600]
    em_edges = np.array([300., 450., 550., 700.])

    f = em_overlap_fractions(ei, eps_edges, em_edges)

    assert f.shape == (3, 3)

    np.testing.assert_allclose(f, [[100 / 150, 0, 0],            # Em [250, 400]: 50 keV below the range
                                   [50 / 100, 50 / 100, 0],      # Em [400, 500]
                                   [0, 50 / 100, 50 / 100]])     # Em [500, 600]

    # Fully covering Em range: rows sum to 1
    f = em_overlap_fractions(ei, eps_edges, np.geomspace(10, 1e4, 30))
    np.testing.assert_allclose(f.sum(axis=1), 1)
