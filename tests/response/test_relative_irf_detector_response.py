import numpy as np
import pytest

import healpy as hp
import astropy.units as u
from astropy.coordinates import SkyCoord
from histpy import Axis, Axes, HealpixAxis, Histogram
from scoords import Attitude, SpacecraftFrame

from cosipy.data_io import EmCDSBinnedData
from cosipy.polarization import PolarizationAxis, StereographicConvention
from cosipy.response import FullDetectorResponse, BinnedInstrumentResponse
from cosipy.response.relative_coordinates import RelativeCDSCoordinates
from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized
from cosipy.response.relative_irf_detector_response import RelativeIRFDetectorResponse


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

IRF_EI_EDGES = np.geomspace(100, 1000, 4)


def _irf_axes(nside=1,
              ei_edges=IRF_EI_EDGES,
              eps_edges=np.linspace(-0.5, 0.5, 9),
              phi_edges_deg=np.linspace(0, 180, 7),
              theta_edges_deg=np.linspace(-90, 90, 5),
              zeta_edges_deg=np.linspace(0, 360, 9)):

    return Axes([
        HealpixAxis(nside=nside, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
        Axis(np.asarray(ei_edges) * u.keV, label='Ei', scale='log'),
        Axis(np.asarray(eps_edges), label='Epsilon'),
        Axis(np.asarray(phi_edges_deg) * u.deg, label='Phi'),
        Axis(np.asarray(theta_edges_deg) * u.deg, label='Theta'),
        PolarizationAxis(np.asarray(zeta_edges_deg) * u.deg, convention=StereographicConvention(), label='Zeta'),
    ])


def _make_random_irf(nside=1, seed=0, aeff=None, **kwargs):
    rng = np.random.default_rng(seed)
    axes = _irf_axes(nside=nside, **kwargs)
    return IRFRelativeHistUnpolarized(Histogram(axes, contents=rng.random(axes.nbins), unit=u.cm * u.cm),
                                      aeff=aeff)


def _histogram_total(irf):
    """
    Content (cm2) of the IRF histogram per (NuLambda, Ei), recomputed
    independently from the differential effective area: density times
    phase space, summed over Epsilon, Phi, Theta and Zeta. Unphysical
    bins are zero.
    """

    d = irf.differential_effective_area_hist
    ax = d.axes

    phi_m, theta_m, zeta_m = np.meshgrid(ax['Phi'].edges, ax['Theta'].edges, ax['Zeta'].edges, indexing='ij')

    ps_cds = RelativeCDSCoordinates.get_relative_cds_phase_space(phi_m[:-1, :-1, :-1], phi_m[1:, :-1, :-1],
                                                                 theta_m[:-1, :-1, :-1], theta_m[:-1, 1:, :-1],
                                                                 zeta_m[:-1, :-1, :-1], zeta_m[:-1, :-1, 1:])

    ps_em = ax['Ei'].centers[:, None] * ax['Epsilon'].widths[None, :]

    return np.einsum('pieabz,ie,abz->pi', np.asarray(d.contents), ps_em, ps_cds)


def _measurement_axes(em_edges=np.geomspace(10, 5000, 31),
                      phi_edges_deg=np.linspace(0, 180, 7),
                      nside=4,
                      coordsys=SpacecraftFrame(),
                      order=('Em', 'Phi', 'PsiChi')):
    """Defaults cover the full IRF range: Ei*(1+Epsilon) in [50, 1500] keV."""

    axes = {'Em': Axis(np.asarray(em_edges) * u.keV, label='Em', scale='log'),
            'Phi': Axis(np.asarray(phi_edges_deg) * u.deg, label='Phi'),
            'PsiChi': HealpixAxis(nside=nside, scheme='ring', coordsys=coordsys, label='PsiChi')}

    return Axes([axes[label] for label in order])


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

class TestConstruction:

    def test_axes(self):

        irf = _make_random_irf()
        meas = _measurement_axes(coordsys='galactic', order=('Phi', 'PsiChi', 'Em'))

        dr = RelativeIRFDetectorResponse(irf, meas)

        assert isinstance(dr, FullDetectorResponse)
        assert list(dr.axes.labels) == ['NuLambda', 'Ei', 'Phi', 'PsiChi', 'Em']
        assert dr.unit == u.cm ** 2
        assert not dr.is_relative_cds
        assert dr.pa_convention is None

        # NuLambda and Ei are the IRF ones
        assert dr.nside == irf.axes['NuLambda'].nside
        np.testing.assert_allclose(dr.axes['Ei'].edges.to_value(u.keV), irf.axes['Ei'].edges)

        # PsiChi always in local coordinates, same pixelization
        assert isinstance(dr.axes['PsiChi'].coordsys, SpacecraftFrame)
        assert dr.axes['PsiChi'].nside == meas['PsiChi'].nside

        # Em and Phi are the measurement ones
        assert dr.measurement_axes['Em'] == meas['Em']
        assert dr.measurement_axes['Phi'] == meas['Phi']

        assert dr[0].contents.shape == tuple(dr.axes[1:].nbins)

    def test_bad_inputs(self):

        irf = _make_random_irf()

        with pytest.raises(TypeError):
            RelativeIRFDetectorResponse("not an irf", _measurement_axes())

        bad = Axes([Axis(np.geomspace(20, 2000, 5) * u.keV, label='Em'),
                    Axis(np.linspace(0, 180, 7) * u.deg, label='Phi')])

        with pytest.raises(ValueError):
            RelativeIRFDetectorResponse(irf, bad)

        with pytest.raises(ValueError):
            RelativeIRFDetectorResponse(irf, _measurement_axes(), phi_samples=0)

        with pytest.raises(NotImplementedError):
            RelativeIRFDetectorResponse.open("some_file.h5")


class TestContentConservation:

    @pytest.mark.parametrize('phi_edges_deg', [np.linspace(0, 180, 7),     # IRF binning
                                               np.linspace(0, 180, 19)])   # Finer binning
    @pytest.mark.parametrize('nside, psichi_subdivision', [(2, 0), (4, 1)])
    def test_total_equals_histogram(self, phi_edges_deg, nside, psichi_subdivision):
        """
        With measurement axes covering the full IRF range, the response
        summed over all the measurement bins must equal the IRF
        histogram content exactly, for every NuLambda and Ei bin,
        independently of the sampling.
        """

        irf = _make_random_irf(seed=1)

        dr = RelativeIRFDetectorResponse(irf, _measurement_axes(phi_edges_deg=phi_edges_deg, nside=nside),
                                         psichi_subdivision=psichi_subdivision)

        expected = _histogram_total(irf)

        for pix in range(dr.npix):
            np.testing.assert_allclose(dr._get_pixel(pix).sum(axis=(1, 2, 3)), expected[pix], rtol=1e-10)

    def test_partial_ranges(self):
        """Restricted Em and Phi ranges only lose content."""

        irf = _make_random_irf(seed=2)

        meas = _measurement_axes(em_edges=np.geomspace(150, 600, 11), phi_edges_deg=np.linspace(30, 150, 5))

        dr = RelativeIRFDetectorResponse(irf, meas)

        expected = _histogram_total(irf)

        for pix in range(dr.npix):
            total = dr._get_pixel(pix).sum(axis=(1, 2, 3))
            assert np.all(total < expected[pix])
            assert np.all(total > 0)

    def test_em_exact(self):
        """
        Summed over Phi and PsiChi, the Em distribution is the exact
        Epsilon -> Em redistribution of the histogram contents.
        """

        irf = _make_random_irf(seed=3)

        em_edges = np.geomspace(10, 5000, 31)
        dr = RelativeIRFDetectorResponse(irf, _measurement_axes(em_edges=em_edges))

        d = irf.differential_effective_area_hist
        ax = d.axes

        phi_m, theta_m, zeta_m = np.meshgrid(ax['Phi'].edges, ax['Theta'].edges, ax['Zeta'].edges, indexing='ij')
        ps_cds = RelativeCDSCoordinates.get_relative_cds_phase_space(phi_m[:-1, :-1, :-1], phi_m[1:, :-1, :-1],
                                                                     theta_m[:-1, :-1, :-1], theta_m[:-1, 1:, :-1],
                                                                     zeta_m[:-1, :-1, :-1], zeta_m[:-1, :-1, 1:])

        pix = 3
        content_eps = np.einsum('ieabz,abz->ie', np.asarray(d.contents[pix]), ps_cds) \
                      * ax['Ei'].centers[:, None] * ax['Epsilon'].widths[None, :]

        response_em = dr._get_pixel(pix).sum(axis=(2, 3))

        for i, ei in enumerate(ax['Ei'].centers):

            em_lo = ei * (1 + ax['Epsilon'].edges[:-1])
            em_hi = ei * (1 + ax['Epsilon'].edges[1:])

            expected = np.zeros(len(em_edges) - 1)

            for c, lo, hi in zip(content_eps[i], em_lo, em_hi):
                overlap = np.clip(np.minimum(hi, em_edges[1:]) - np.maximum(lo, em_edges[:-1]), 0, None)
                expected += c * overlap / (hi - lo)

            np.testing.assert_allclose(response_em[i], expected, rtol=1e-10, atol=1e-12)


class TestConventions:

    def test_single_bin_lands_at_expected_direction(self):
        """
        An IRF with a single non-empty relative bin must produce a
        response concentrated in the data Phi bin of that IRF bin and in
        the Em bin of Ei*(1+Epsilon), with PsiChi pixels centered on the
        direction of the bin center (Phi + Theta from the source, at
        azimuth Zeta in the IRF polarization convention).
        """

        axes = _irf_axes(nside=1,
                         phi_edges_deg=np.linspace(0, 180, 19),
                         theta_edges_deg=np.linspace(-90, 90, 19),
                         zeta_edges_deg=np.linspace(0, 360, 17))

        contents = np.zeros(axes.nbins)

        i_ei, i_eps, a, b, c = 1, 4, 6, 10, 3   # Phi 60-70 deg, Theta 10-20 deg, Zeta 67.5-90 deg
        contents[:, i_ei, i_eps, a, b, c] = 1

        irf = IRFRelativeHistUnpolarized(Histogram(axes, contents=contents, unit=u.cm * u.cm))

        meas = _measurement_axes(phi_edges_deg=np.linspace(0, 180, 19), nside=16)

        dr = RelativeIRFDetectorResponse(irf, meas, psichi_subdivision=1)

        pix = 5
        r = dr._get_pixel(pix)    # (Ei, Em, Phi, PsiChi)

        # Total conserved: content of one bin = 1 cm2
        np.testing.assert_allclose(r.sum(), 1, rtol=1e-10)

        # Only the right Ei and data Phi bin
        np.testing.assert_allclose(r[i_ei].sum(), 1, rtol=1e-10)
        np.testing.assert_allclose(r[i_ei, :, a].sum(), 1, rtol=1e-10)

        # Em: Ei * (1 + Epsilon) for that bin
        ei = irf.axes['Ei'].centers[i_ei]
        eps_lo, eps_hi = irf.axes['Epsilon'].edges[[i_eps, i_eps + 1]]
        em_edges = meas['Em'].edges.to_value(u.keV)
        em_bins = np.nonzero(r[i_ei].sum(axis=(1, 2)) > 0)[0]
        assert em_edges[em_bins[0]] < ei * (1 + eps_lo) + 1e-9
        assert em_edges[em_bins[-1] + 1] > ei * (1 + eps_hi) - 1e-9

        # PsiChi: content-weighted mean direction close to the bin center direction
        src_vec = np.asarray(dr.pix2vec(pix))
        relcoords = RelativeCDSCoordinates(src_vec[:, None], irf.pol_convention)

        phi_c = np.deg2rad(65 + 15)
        zeta_c = np.deg2rad(78.75)
        expected_vec = np.asarray(relcoords.to_cds(np.array([phi_c]), np.array([zeta_c]))).reshape(3)

        psichi_weights = r[i_ei].sum(axis=(0, 1))
        psichi_vecs = np.asarray(hp.pix2vec(16, np.arange(hp.nside2npix(16))))
        mean_vec = psichi_vecs @ psichi_weights
        mean_vec /= np.linalg.norm(mean_vec)

        angle = np.rad2deg(np.arccos(np.clip(mean_vec @ expected_vec, -1, 1)))

        assert angle < 5


class TestAeffNormalization:

    @staticmethod
    def _nested_aeff(seed=5):
        """AEFF with nside 2 (IRF: 1) and each IRF Ei bin split in 2."""

        rng = np.random.default_rng(seed)

        ei_edges = np.sort(np.concatenate([IRF_EI_EDGES, np.sqrt(IRF_EI_EDGES[1:] * IRF_EI_EDGES[:-1])]))

        axes = Axes([HealpixAxis(nside=2, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
                     Axis(ei_edges * u.keV, label='Ei', scale='log')])

        return Histogram(axes, contents=rng.random(axes.nbins) * 100, unit=u.cm * u.cm)

    @staticmethod
    def _expected_average(aeff):

        contents = np.asarray(aeff.to(u.cm ** 2).contents.value)
        ei_widths = np.diff(aeff.axes['Ei'].edges.to_value(u.keV))

        # Width-weighted average over the 2 sub-bins of each IRF Ei bin
        avg_ei = (contents[:, 0::2] * ei_widths[0::2] + contents[:, 1::2] * ei_widths[1::2]) \
                 / (ei_widths[0::2] + ei_widths[1::2])

        # Mean over the 4 children of each nside 1 pixel
        parent = hp.nest2ring(1, hp.ring2nest(2, np.arange(48)) // 4)

        avg = np.zeros((12, avg_ei.shape[1]))
        np.add.at(avg, parent, avg_ei / 4)

        return avg

    def test_normalize_to_aeff(self):

        aeff = self._nested_aeff()
        irf = _make_random_irf(seed=4, aeff=aeff)

        expected = self._expected_average(aeff)

        dr = RelativeIRFDetectorResponse(irf, _measurement_axes(nside=2), normalize_to_aeff=True)

        for pix in range(dr.npix):
            np.testing.assert_allclose(dr._get_pixel(pix).sum(axis=(1, 2, 3)), expected[pix], rtol=1e-10)

        # Ratio helper
        ratio = dr.aeff_ratio()
        assert list(ratio.axes.labels) == ['NuLambda', 'Ei']
        np.testing.assert_allclose(ratio.contents, expected / _histogram_total(irf), rtol=1e-10)

        # The shape is not changed
        dr_raw = RelativeIRFDetectorResponse(irf, _measurement_axes(nside=2))
        pix = 7
        raw = dr_raw._get_pixel(pix)
        norm = dr._get_pixel(pix)
        np.testing.assert_allclose(norm, raw * ratio.contents[pix][:, None, None, None], rtol=1e-10)

    def test_non_nested_aeff(self):

        rng = np.random.default_rng(0)
        axes = Axes([HealpixAxis(nside=2, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
                     Axis(np.geomspace(50, 2000, 6) * u.keV, label='Ei', scale='log')])
        aeff = Histogram(axes, contents=rng.random(axes.nbins), unit=u.cm * u.cm)

        irf = _make_random_irf(aeff=aeff)

        # Fine without normalization
        RelativeIRFDetectorResponse(irf, _measurement_axes(nside=2))

        with pytest.raises(ValueError):
            RelativeIRFDetectorResponse(irf, _measurement_axes(nside=2), normalize_to_aeff=True)


class TestCacheAndAccess:

    def test_cache(self):

        irf = _make_random_irf()
        dr = RelativeIRFDetectorResponse(irf, _measurement_axes(nside=2), cache_size=1)

        a = dr._get_pixel(0)
        assert list(dr._cache) == [0]

        dr._get_pixel(1)
        assert list(dr._cache) == [1]

        # Weight doesn't modify the cache
        np.testing.assert_allclose(dr._get_pixel(0, weight=3 * u.s), 3 * a)
        np.testing.assert_allclose(dr._get_pixel(0), a)

        # Pixel given as tuple / array, as done by BinnedInstrumentResponse
        np.testing.assert_allclose(dr._get_pixel((np.int64(0),)), a)
        np.testing.assert_allclose(dr._get_pixel(np.array([0])), a)

        dr.clear_cache()
        assert len(dr._cache) == 0

        dr_nocache = RelativeIRFDetectorResponse(irf, _measurement_axes(nside=2), cache_size=0)
        dr_nocache._get_pixel(0)
        assert len(dr_nocache._cache) == 0

        with pytest.raises(IndexError):
            dr._get_pixel(dr.npix)

    def test_getitem(self):

        irf = _make_random_irf()
        dr = RelativeIRFDetectorResponse(irf, _measurement_axes(nside=2))

        r = dr[2]
        assert r.unit == u.cm ** 2
        np.testing.assert_allclose(r.contents.to_value(u.cm ** 2), dr._get_pixel(2))


class TestDumpAndToDr:

    def test_dump_roundtrip(self, tmp_path):

        irf = _make_random_irf()
        dr = RelativeIRFDetectorResponse(irf, _measurement_axes(nside=2))

        path = tmp_path / "rsp.h5"
        dr.dump(path, dtype=np.float64)

        with pytest.raises(FileExistsError):
            dr.dump(path)

        with FullDetectorResponse.open(path) as fdr:

            assert fdr.axes == dr.axes
            assert fdr.unit == dr.unit

            for pix in range(dr.npix):
                np.testing.assert_allclose(fdr._get_pixel(pix), dr._get_pixel(pix))

    def test_to_dr(self):

        irf = _make_random_irf()
        dr = RelativeIRFDetectorResponse(irf, _measurement_axes(nside=2))

        full = dr.to_dr()

        assert full.axes == dr.axes
        np.testing.assert_allclose(full.contents.to_value(u.cm ** 2)[3], dr._get_pixel(3))


class TestBinnedInstrumentResponse:

    def test_inertial_identity_attitude(self):
        """
        With PsiChi in an inertial frame and an identity attitude, the
        BinnedInstrumentResponse rotation is trivial, and the result
        must be the response of the pixel containing the direction.
        """

        irf = _make_random_irf()
        meas = _measurement_axes(nside=2, coordsys='icrs')

        data = EmCDSBinnedData(Histogram(meas))

        dr = RelativeIRFDetectorResponse(irf, meas)

        response = BinnedInstrumentResponse(dr, data)

        pix = 7
        vec = dr.pix2vec(pix)
        direction = SkyCoord(x=vec[0], y=vec[1], z=vec[2], representation_type='cartesian',
                             frame='icrs')

        out = response.differential_effective_area(direction,
                                                   dr.axes['Ei'].centers,
                                                   attitude=Attitude.from_matrix(np.eye(3), frame='icrs'),
                                                   weight=2 * u.s)

        np.testing.assert_allclose(u.Quantity(out).to_value(u.cm ** 2 * u.s), 2 * dr._get_pixel(pix))
