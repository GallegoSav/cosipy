import numpy as np
import pytest

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

def _irf_axes(nside=1,
              ei_edges=np.geomspace(100, 1000, 4),
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


def _make_random_irf(nside=1, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    axes = _irf_axes(nside=nside, **kwargs)
    return IRFRelativeHistUnpolarized(Histogram(axes, contents=rng.random(axes.nbins), unit=u.cm * u.cm))


def _make_constant_density_irf(density=2.0, nside=1):
    """
    IRF whose differential effective area is exactly ``density``
    (cm2/sr/rad/keV) everywhere within its axes ranges. The Phi and
    Theta ranges are chosen so that no bin is in the unphysical
    region (Phi + Theta outside [0, 180] deg). Then the multilinear
    interpolation is exactly constant, and the integral over the full
    measurement space must equal the total effective area.
    """

    axes = _irf_axes(nside=nside,
                     phi_edges_deg=np.linspace(20, 160, 8),
                     theta_edges_deg=np.linspace(-10, 10, 5))

    phi = axes['Phi'].edges.to_value(u.rad)
    theta = axes['Theta'].edges.to_value(u.rad)
    zeta = axes['Zeta'].edges.angle.to_value(u.rad)

    phi_m, theta_m, zeta_m = np.meshgrid(phi, theta, zeta, indexing='ij')

    ps_cds = RelativeCDSCoordinates.get_relative_cds_phase_space(phi_m[:-1, :-1, :-1], phi_m[1:, :-1, :-1],
                                                                 theta_m[:-1, :-1, :-1], theta_m[:-1, 1:, :-1],
                                                                 zeta_m[:-1, :-1, :-1], zeta_m[:-1, :-1, 1:])

    ei_c = axes['Ei'].centers.to_value(u.keV)
    eps_w = axes['Epsilon'].widths

    ps_em = ei_c[:, None] * eps_w[None, :]

    contents = density * ps_em[None, :, :, None, None, None] * ps_cds[None, None, None, :, :, :]
    contents = np.broadcast_to(contents, axes.nbins).copy()

    return IRFRelativeHistUnpolarized(Histogram(axes, contents=contents, unit=u.cm * u.cm))


def _measurement_axes(em_edges=np.geomspace(20, 2000, 11),
                      phi_edges_deg=np.linspace(0, 180, 7),
                      nside=4,
                      coordsys=SpacecraftFrame(),
                      order=('Em', 'Phi', 'PsiChi')):

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
        assert dr.nside == irf.axes['NuLambda'].nside
        assert dr.unit == u.cm ** 2
        assert not dr.is_relative_cds
        assert dr.pa_convention is None

        # PsiChi always in local coordinates, same pixelization
        assert isinstance(dr.axes['PsiChi'].coordsys, SpacecraftFrame)
        assert dr.axes['PsiChi'].nside == meas['PsiChi'].nside

        # Em and Phi are the measurement ones
        assert dr.measurement_axes['Em'] == meas['Em']
        assert dr.measurement_axes['Phi'] == meas['Phi']

        # Default energy axis is the IRF's one, with units
        np.testing.assert_allclose(dr.axes['Ei'].edges.to_value(u.keV), irf.axes['Ei'].edges)

        # Pixel shape follows the rest axes
        assert dr[0].contents.shape == tuple(dr.axes[1:].nbins)

    def test_custom_energy_and_nside(self):

        irf = _make_random_irf(nside=1)
        ei = Axis(np.geomspace(150, 900, 6) * u.keV, label='Ei', scale='log')

        dr = RelativeIRFDetectorResponse(irf, _measurement_axes(), energy_axis=ei, nside=2)

        assert dr.nside == 2
        assert dr.axes['Ei'] == ei
        assert dr._get_pixel(5).shape == tuple(dr.axes[1:].nbins)

    def test_bad_inputs(self):

        irf = _make_random_irf()

        with pytest.raises(TypeError):
            RelativeIRFDetectorResponse("not an irf", _measurement_axes())

        bad = Axes([Axis(np.geomspace(20, 2000, 5) * u.keV, label='Em'),
                    Axis(np.linspace(0, 180, 7) * u.deg, label='Phi')])

        with pytest.raises(ValueError):
            RelativeIRFDetectorResponse(irf, bad)

        with pytest.raises(ValueError):
            RelativeIRFDetectorResponse(irf, _measurement_axes(),
                                        energy_axis=Axis(np.geomspace(100, 1000, 4) * u.keV, label='E'))

        with pytest.raises(NotImplementedError):
            RelativeIRFDetectorResponse.open("some_file.h5")


class TestNormalization:

    @pytest.mark.parametrize('psichi_subdivision, rtol', [(1, 0.06), (2, 0.03)])
    def test_total_effective_area(self, psichi_subdivision, rtol):
        """
        For a constant-density IRF, integrating over measurement axes
        that cover the whole IRF range must recover the total
        effective area, up to the PsiChi quadrature error at the edges
        of the Theta (ARM) band.
        """

        irf = _make_constant_density_irf()

        # Em covers Ei * (1 + Epsilon) for all Ei centers. Phi edges
        # aligned with the IRF ones.
        meas = _measurement_axes(em_edges=np.geomspace(20, 2000, 13),
                                 phi_edges_deg=np.linspace(20, 160, 8),
                                 nside=8)

        dr = RelativeIRFDetectorResponse(irf, meas, phi_samples=3, psichi_subdivision=psichi_subdivision)

        tot_aeff = np.asarray(irf._tot_aeff.contents)  # cm2, (NuLambda, Ei)

        for pix in [0, 5, 11]:
            binned = dr._get_pixel(pix).sum(axis=(1, 2, 3))
            np.testing.assert_allclose(binned, tot_aeff[pix], rtol=rtol)

    def test_convergence_with_subdivision(self):

        irf = _make_constant_density_irf()
        meas = _measurement_axes(em_edges=np.geomspace(20, 2000, 13),
                                 phi_edges_deg=np.linspace(20, 160, 8),
                                 nside=4)

        tot_aeff = np.asarray(irf._tot_aeff.contents)[3]

        errors = []
        for sub in (0, 1, 2):
            dr = RelativeIRFDetectorResponse(irf, meas, psichi_subdivision=sub)
            errors.append(np.max(np.abs(dr._get_pixel(3).sum(axis=(1, 2, 3)) / tot_aeff - 1)))

        assert errors[2] < errors[0]

    def test_energy_outside_irf_is_zero(self):

        irf = _make_random_irf()
        ei = Axis([10, 50, 300, 5000, 9000] * u.keV, label='Ei')

        dr = RelativeIRFDetectorResponse(irf, _measurement_axes(), energy_axis=ei)

        data = dr._get_pixel(0)

        assert np.all(data[0] == 0)    # Ei center 30 keV < 100 keV
        assert np.all(data[3] == 0)    # Ei center 7000 keV > 1000 keV
        assert np.any(data[1] > 0)


class TestConsistencyWithUnbinned:

    def test_matches_brute_force_integration(self):
        """
        Compare a few bins with a brute force integration of the IRF
        unbinned differential effective area (same Phi and PsiChi
        sub-samples, fine grid in Em). This checks the coordinate
        conventions (Epsilon, Theta = phi_geo - Phi, Zeta) and the
        phase-space normalization.
        """

        irf = _make_random_irf(nside=1, seed=3, theta_edges_deg=np.linspace(-40, 40, 9))

        meas = _measurement_axes(em_edges=np.array([150., 210., 260., 330.]),
                                 phi_edges_deg=np.linspace(0, 180, 7),
                                 nside=2)

        dr = RelativeIRFDetectorResponse(irf, meas, phi_samples=2, psichi_subdivision=1)

        pix = 4
        response = dr._get_pixel(pix)  # (Ei, Em, Phi, PsiChi)

        ei_index = 1
        ei = dr._ei_keV[ei_index]

        src_vec = np.asarray(dr.pix2vec(pix))
        src_lon = np.arctan2(src_vec[1], src_vec[0])
        src_lat = np.arcsin(src_vec[2])

        theta_lo, theta_hi = np.deg2rad(-40), np.deg2rad(40)

        n_em = 2000

        for em_bin, phi_bin, psichi_bin in [(0, 1, 3), (1, 2, 10), (2, 3, 25), (1, 4, 40)]:

            em_lo, em_hi = meas['Em'].edges.to_value(u.keV)[[em_bin, em_bin + 1]]
            em = em_lo + (np.arange(n_em) + 0.5) / n_em * (em_hi - em_lo)

            phi_sel = dr._phi_s_bin == phi_bin
            phis = dr._phi_s[phi_sel]
            phiw = dr._phi_s_width[phi_sel]

            sub_sel = dr._psichi_sub_bin == psichi_bin
            sub_vec = dr._psichi_sub_vec[:, sub_sel]
            sub_w = dr._psichi_sub_weight[sub_sel]

            # Cartesian product of samples
            em_m, phi_m, sub_m = np.meshgrid(np.arange(n_em), np.arange(len(phis)), np.arange(sub_vec.shape[1]),
                                             indexing='ij')
            em_m, phi_m, sub_m = em_m.ravel(), phi_m.ravel(), sub_m.ravel()

            v = sub_vec[:, sub_m]

            class Photons:
                direction_lon_rad_sc = np.full(len(em_m), src_lon)
                direction_lat_rad_sc = np.full(len(em_m), src_lat)
                energy_keV = np.full(len(em_m), ei)

            class Events:
                scattered_lon_rad_sc = np.arctan2(v[1], v[0])
                scattered_lat_rad_sc = np.arcsin(np.clip(v[2], -1, 1))
                scattering_angle_rad = phis[phi_m]
                energy_keV = em[em_m]

            values = np.asarray(irf._differential_effective_area_cm2(Photons, Events))

            # The binned response is zero outside the IRF Theta and
            # Epsilon ranges, the unbinned evaluation clamps instead
            phi_geo, _ = RelativeCDSCoordinates(src_vec[:, None], irf.pol_convention).to_relative(v)
            theta = phi_geo.to_value(u.rad).ravel() - phis[phi_m]
            eps = em[em_m] / ei - 1
            eps_edges = irf.axes['Epsilon'].edges
            inside = ((theta >= theta_lo) & (theta <= theta_hi)
                      & (eps >= eps_edges[0]) & (eps <= eps_edges[-1]))
            values = np.where(inside, values, 0)

            expected = np.sum(values * phiw[phi_m] * sub_w[sub_m]) * (em_hi - em_lo) / n_em

            got = response[ei_index, em_bin, phi_bin, psichi_bin]

            assert expected > 0
            np.testing.assert_allclose(got, expected, rtol=2e-3)


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

    def test_getitem_and_interp(self):

        irf = _make_random_irf()
        dr = RelativeIRFDetectorResponse(irf, _measurement_axes(nside=2))

        r = dr[2]
        assert r.unit == u.cm ** 2
        np.testing.assert_allclose(r.contents.to_value(u.cm ** 2), dr._get_pixel(2))

        # At a pixel center, the interpolated response is that pixel
        coord = dr.pix2skycoord(2)
        np.testing.assert_allclose(dr.get_interp_response(coord).contents.to_value(u.cm ** 2),
                                   dr._get_pixel(2), rtol=1e-5)


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
