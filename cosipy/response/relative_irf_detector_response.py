"""
Binned (absolute Compton Data Space) view of a relative-coordinate
histogram IRF, usable everywhere a :class:`FullDetectorResponse` is
expected -- in particular with :class:`BinnedInstrumentResponse` for
binned likelihood analyses.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np

import h5py as h5
import healpy as hp

import astropy.units as u
from astropy.units import Quantity

from mhealpy import HealpixBase
from histpy import Axis, Axes, HealpixAxis
from scoords import SpacecraftFrame

from .FullDetectorResponse import FullDetectorResponse
from .DetectorResponse import DetectorResponse
from .relative_irf_hist import IRFRelativeHistUnpolarized
from ._relative_irf_binning import build_transfer_matrix, integrate_em_bins, subsample_edges

import logging
logger = logging.getLogger(__name__)

__all__ = ["RelativeIRFDetectorResponse"]


def _is_nested(axis) -> bool:
    return str(axis.scheme).upper().startswith('NEST')


class RelativeIRFDetectorResponse(FullDetectorResponse):
    """
    A :class:`FullDetectorResponse` computed on the fly from a
    relative-coordinate histogram IRF
    (:class:`~cosipy.response.relative_irf_hist.IRFRelativeHistUnpolarized`).

    The relative IRF describes the differential effective area as a
    function of the photon direction (``NuLambda``) and energy
    (``Ei``), and of the measured quantities expressed relative to the
    photon (``Epsilon``, ``Phi``, ``Theta``, ``Zeta``). This class
    integrates it over the bins of an absolute Compton Data Space
    (``Em``, ``Phi``, ``PsiChi``), chosen by the user -- typically the
    axes of the binned data -- producing, for each ``NuLambda`` pixel,
    the same ``[Ei, Em, Phi, PsiChi]`` matrix (in cm^2) that a
    :class:`FullDetectorResponse` read from file would provide. All
    the algorithms of :class:`FullDetectorResponse` that only rely on
    those per-pixel matrices are inherited as is
    (:meth:`get_interp_response`, :meth:`get_point_source_response`,
    including the scatt-map/inertial-frame path,
    :meth:`get_extended_source_response`, ...), and an instance can be
    passed to :class:`~cosipy.response.BinnedInstrumentResponse`.

    Earth occultation is not a property of the response: as for a
    regular :class:`FullDetectorResponse`, it is handled by the
    ``scatt_map`` (``SpacecraftHistory.get_scatt_map(earth_occ=True)``).

    Integration
    -----------
    For a ``NuLambda`` pixel the source direction is the pixel center,
    and the IRF is bilinearly interpolated to it (HEALPix
    interpolation). The photon energy is evaluated at the ``Ei`` bin
    centers (linear interpolation of the IRF in ``Ei``), consistent
    with how :class:`FullDetectorResponse` contents are used.

    - ``Em``: integrated exactly, consistent with the piecewise-linear
      interpolation of the IRF along ``Epsilon = Em/Ei - 1``.
    - ``Phi``: midpoint rule with ``phi_samples`` samples per bin.
    - ``PsiChi``: each pixel is split into ``4**psichi_subdivision``
      HEALPix sub-pixels (midpoint rule on the sphere).

    At each (Phi, PsiChi) sample the IRF is multilinearly interpolated
    in (Phi, Theta, Zeta), the same model used by the unbinned
    evaluation of the IRF. Points outside of the IRF axes ranges have
    zero response. Increase ``phi_samples`` and ``psichi_subdivision``
    if the Phi or PsiChi bins are coarse compared to the IRF ``Theta``
    resolution (angular resolution).

    The per-pixel computation is cached. Alternatively, all pixels
    can be precomputed with :meth:`to_dr`, or written to a standard
    response file with :meth:`dump` and reopened later with
    :meth:`FullDetectorResponse.open`.

    Parameters
    ----------
    irf : IRFRelativeHistUnpolarized
        The relative-coordinate histogram IRF.
    measurement_axes : histpy.Axes
        Measured axes ``Em``, ``Phi`` and ``PsiChi`` (in any order,
        which is preserved), e.g. ``binned_data.axes``. The ``PsiChi``
        axis can be in any frame (e.g. galactic): only its nside and
        scheme are used, and the response ``PsiChi`` axis is always in
        the spacecraft frame.
    energy_axis : histpy.Axis, optional
        Photon energy binning (``Ei``). Defaults to the IRF ``Ei``
        axis. When used through :class:`BinnedThreeMLPointSourceResponse`
        you must pass this same axis as its ``energy_axis``.
    nside : int, optional
        NSIDE of the ``NuLambda`` axis. Defaults to that of the IRF.
    scheme : str, optional
        HEALPix scheme of the ``NuLambda`` axis. Defaults to that of
        the IRF.
    phi_samples : int, optional
        Number of samples per ``Phi`` bin. Default 3.
    psichi_subdivision : int, optional
        Each ``PsiChi`` pixel is split into ``4**psichi_subdivision``
        sub-pixels. Default 1.
    cache_size : int or None, optional
        Maximum number of ``NuLambda`` pixels kept in memory (least
        recently used are dropped). ``None`` (default) means no limit,
        ``0`` disables the cache. Each pixel takes
        ``8 * nEi * nEm * nPhi * nPsiChi`` bytes.
    max_pairs : int, optional
        Maximum number of (Phi, PsiChi) sample pairs processed at once,
        to bound the memory used during the computation.
    dtype : numpy dtype, optional
        Data type of the returned contents. Default float64.
    """

    def __init__(self,
                 irf: IRFRelativeHistUnpolarized,
                 measurement_axes: Axes,
                 energy_axis: Axis = None,
                 nside: int = None,
                 scheme: str = None,
                 phi_samples: int = 3,
                 psichi_subdivision: int = 1,
                 cache_size: int = None,
                 max_pairs: int = 2_000_000,
                 dtype=np.float64):

        if not isinstance(irf, IRFRelativeHistUnpolarized):
            raise TypeError(f"irf must be an IRFRelativeHistUnpolarized, got {type(irf)}")

        if int(phi_samples) < 1:
            raise ValueError("phi_samples must be >= 1")

        if int(psichi_subdivision) < 0:
            raise ValueError("psichi_subdivision must be >= 0")

        self._irf = irf
        self._phi_samples = int(phi_samples)
        self._psichi_subdivision = int(psichi_subdivision)
        self._max_pairs = int(max_pairs)
        self._cache_size = cache_size
        self._cache = OrderedDict()

        irf_axes = irf.axes

        # ---- Measurement axes ----
        if not isinstance(measurement_axes, Axes):
            measurement_axes = Axes(measurement_axes)

        if set(measurement_axes.labels) != {'Em', 'Phi', 'PsiChi'}:
            raise ValueError("measurement_axes must have exactly the labels 'Em', 'Phi' and 'PsiChi', "
                             f"got {list(measurement_axes.labels)}")

        em_axis = measurement_axes['Em']
        phi_axis = measurement_axes['Phi']
        psichi_axis = measurement_axes['PsiChi']

        if em_axis.unit is None or not em_axis.unit.is_equivalent(u.keV):
            raise ValueError("Em axis must have units of energy")

        if phi_axis.unit is None or not phi_axis.unit.is_equivalent(u.rad):
            raise ValueError("Phi axis must have units of angle")

        if not isinstance(psichi_axis, HealpixAxis):
            raise ValueError("PsiChi axis must be a HealpixAxis")

        # Response PsiChi is always in local (spacecraft) coordinates.
        # The rotation to an inertial frame is done by the inherited
        # get_point_source_response / BinnedInstrumentResponse.
        local_psichi_axis = psichi_axis.copy()
        local_psichi_axis.coordsys = SpacecraftFrame()

        # ---- Photon energy axis ----
        if energy_axis is None:
            irf_ei = irf_axes['Ei']
            energy_axis = Axis(irf_ei.edges * u.keV, label='Ei', scale=irf_ei.axis_scale)

        if energy_axis.label != 'Ei':
            raise ValueError(f"energy_axis must have label 'Ei', got '{energy_axis.label}'")

        if energy_axis.unit is None or not energy_axis.unit.is_equivalent(u.keV):
            raise ValueError("energy_axis must have units of energy")

        # ---- NuLambda axis ----
        irf_nulambda = irf_axes['NuLambda']

        if nside is None:
            nside = irf_nulambda.nside

        if scheme is None:
            scheme = 'nested' if _is_nested(irf_nulambda) else 'ring'

        nulambda_axis = HealpixAxis(nside=nside, scheme=scheme,
                                    coordsys=SpacecraftFrame(), label='NuLambda')

        # ---- FullDetectorResponse state ----
        # Keep the measurement axes order, so the rest axes match the
        # data axes order as expected by BinnedInstrumentResponse
        rest_axes = [energy_axis]
        for label in measurement_axes.labels:
            rest_axes.append(local_psichi_axis if label == 'PsiChi' else measurement_axes[label])

        self._axes = Axes([nulambda_axis] + rest_axes)
        self._rest_axes = self._axes[1:]
        self._cds_axes = self._axes[('Em', 'Phi', 'PsiChi')]
        self._is_relative_cds = False
        self._pa_convention = None
        self._unit = u.Unit('cm2')
        self._eff_area = np.ones(energy_axis.nbins, dtype=dtype)

        HealpixBase.__init__(self, base=nulambda_axis, coordsys=SpacecraftFrame())

        # Order to go from the computation order (Ei, Em, Phi, PsiChi)
        # to the rest axes order
        computation_labels = ['Ei', 'Em', 'Phi', 'PsiChi']
        self._transpose = [computation_labels.index(label) for label in self._rest_axes.labels]

        # ---- Precomputed quantities, independent of NuLambda ----

        # Photon energy evaluation points and IRF interpolation weights
        self._ei_keV = np.asarray(energy_axis.centers.to_value(u.keV), dtype=float)

        irf_ei_edges = np.asarray(irf_axes['Ei'].edges, dtype=float)
        self._ei_inside = (self._ei_keV >= irf_ei_edges[0]) & (self._ei_keV <= irf_ei_edges[-1])

        if irf_axes['Ei'].nbins > 1:
            ei_bins, ei_weights = irf_axes['Ei'].interp_weights(self._ei_keV)
            self._ei_ibins = np.asarray(ei_bins)
            self._ei_iweights = np.asarray(ei_weights)
        else:
            self._ei_ibins = np.zeros((2, len(self._ei_keV)), dtype=int)
            self._ei_iweights = np.vstack([np.ones(len(self._ei_keV)), np.zeros(len(self._ei_keV))])

        # Measured energy
        self._em_edges_keV = np.asarray(em_axis.edges.to_value(u.keV), dtype=float)

        # Phi samples (rad)
        phi_edges_rad = np.asarray(phi_axis.edges.to_value(u.rad), dtype=float)
        self._phi_s, self._phi_s_bin, self._phi_s_width = subsample_edges(phi_edges_rad, self._phi_samples)

        # PsiChi sub-pixels: unit vectors (local frame), parent bin and solid angle
        nside_psichi = psichi_axis.nside
        npix_psichi = psichi_axis.npix
        nsub = 4 ** self._psichi_subdivision
        nside_sub = nside_psichi * 2 ** self._psichi_subdivision

        parents = np.arange(npix_psichi)
        parents_nest = parents if _is_nested(psichi_axis) else hp.ring2nest(nside_psichi, parents)

        sub_pixels = (parents_nest[:, None] * nsub + np.arange(nsub)[None, :]).ravel()

        self._psichi_sub_vec = np.asarray(hp.pix2vec(nside_sub, sub_pixels, nest=True))
        self._psichi_sub_bin = np.repeat(parents, nsub)
        self._psichi_sub_weight = np.full(len(sub_pixels), 4 * np.pi / hp.nside2npix(nside_sub))

        self._n_phi = phi_axis.nbins
        self._n_psichi = npix_psichi

        # IRF relative axes
        self._irf_nside = irf_nulambda.nside
        self._irf_nest = _is_nested(irf_nulambda)
        self._eps_centers = np.asarray(irf_axes['Epsilon'].centers, dtype=float)
        self._eps_edges = np.asarray(irf_axes['Epsilon'].edges, dtype=float)

        irf_phi_edges = np.asarray(irf_axes['Phi'].edges, dtype=float)
        irf_theta_edges = np.asarray(irf_axes['Theta'].edges, dtype=float)
        self._irf_phi_range = (irf_phi_edges[0], irf_phi_edges[-1])
        self._irf_theta_range = (irf_theta_edges[0], irf_theta_edges[-1])

        self._irf_rel_shape = (irf_axes['Phi'].nbins, irf_axes['Theta'].nbins, irf_axes['Zeta'].nbins)

    # ------------------------------------------------------------------
    # Properties / FullDetectorResponse overrides
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, *args, **kwargs):
        raise NotImplementedError("RelativeIRFDetectorResponse is built from an IRF, not opened from a file. "
                                  "Use RelativeIRFDetectorResponse(irf, ...), or "
                                  "FullDetectorResponse.open() on a file written by dump().")

    @property
    def irf(self) -> IRFRelativeHistUnpolarized:
        """The underlying relative-coordinate IRF."""
        return self._irf

    @property
    def filename(self):
        """No backing file. Returns None."""
        return None

    @property
    def headers(self):
        return {}

    @property
    def cache_size(self):
        return self._cache_size

    def clear_cache(self):
        """Drop all cached NuLambda pixels."""
        self._cache.clear()

    def close(self):
        """No file to close; clears the cache."""
        self.clear_cache()

    def get_counts(self, pix, em_slice=None):
        raise NotImplementedError("RelativeIRFDetectorResponse has no raw counts. Use _get_pixel() or [].")

    def __str__(self):
        return (f"{self.__class__.__name__}(nside = {self.nside}, "
                f"phi_samples = {self._phi_samples}, psichi_subdivision = {self._psichi_subdivision})")

    def __repr__(self):

        output = f"{self}\nAXES:\n"

        for axis in self._axes:

            output += f"  {axis.label}:\n"

            if isinstance(axis, HealpixAxis):
                output += (f"    TYPE: 'healpix'\n"
                           f"    NPIX: {axis.npix}\n"
                           f"    NSIDE: {axis.nside}\n"
                           f"    SCHEME: '{axis.scheme}'\n")
            else:
                output += (f"    TYPE: '{axis.axis_scale}'\n"
                           f"    UNIT: '{axis.unit}'\n"
                           f"    NBINS: {axis.nbins}\n")

        return output

    # ------------------------------------------------------------------
    # Per-pixel response
    # ------------------------------------------------------------------

    @staticmethod
    def _standardize_pixel(pix) -> int:
        # FullDetectorResponse/BinnedInstrumentResponse can pass the
        # pixel as an int, a 0-d array or a 1-element tuple/array
        pix_arr = np.asarray(pix)

        if pix_arr.size != 1:
            raise IndexError(f"Expected a single NuLambda pixel, got {pix}")

        return int(pix_arr.reshape(-1)[0])

    def _get_pixel(self, pix, weight=None):
        """
        Response matrix (cm^2) for a single NuLambda pixel, with the
        rest axes order, optionally weighted.

        Parameters
        ----------
        pix : int
            NuLambda pixel
        weight : float or Quantity, optional
            Weight to apply to the response slice

        Returns
        -------
        numpy.ndarray
        """

        pix = self._standardize_pixel(pix)

        if pix < 0 or pix >= self.npix:
            raise IndexError(f"Pixel {pix} out of range for nside {self.nside}")

        data = self._cache.get(pix)

        if data is None:

            data = self._compute_pixel(pix)

            if self._cache_size is None or self._cache_size > 0:
                self._cache[pix] = data

                if self._cache_size is not None:
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(pix)

        if weight is not None:
            if isinstance(weight, Quantity):
                weight = weight.value

            # Don't modify the cache in place
            data = data * weight

        return data

    def _interp_irf_nulambda(self, src_vec: np.ndarray) -> np.ndarray:
        """
        IRF differential effective area bilinearly interpolated at the
        source direction (HEALPix interpolation).

        Returns
        -------
        numpy.ndarray, shape (nEi_irf, nEpsilon, nPhi_irf, nTheta, nZeta)
        """

        theta, phi = hp.vec2ang(src_vec)
        theta = float(np.ravel(theta)[0])
        phi = float(np.ravel(phi)[0])

        pixels, weights = hp.get_interp_weights(self._irf_nside, theta, phi, nest=self._irf_nest)
        pixels = np.ravel(pixels)
        weights = np.ravel(weights)

        contents = self._irf.differential_effective_area_hist.contents

        if hasattr(contents, 'todense'):
            # Sparse contents
            slab = sum(w * np.asarray(contents[p].todense()) for p, w in zip(pixels, weights) if w != 0)
        else:
            contents = np.asarray(contents)
            slab = np.tensordot(weights, contents[pixels], axes=(0, 0))

        return slab

    def _transfer_matrix(self, src_vec: np.ndarray):
        """
        Sparse map from the IRF relative (Phi, Theta, Zeta) bins to the
        absolute (Phi, PsiChi) bins for a given source direction. See
        :func:`cosipy.response._relative_irf_binning.build_transfer_matrix`.
        """

        from .relative_coordinates import RelativeCDSCoordinates

        relcoords = RelativeCDSCoordinates(src_vec[:, None], pol_convention=self._irf.pol_convention)

        phi_geo, az = relcoords.to_relative(self._psichi_sub_vec)

        phi_geo = np.asarray(phi_geo.to_value(u.rad), dtype=float).ravel()

        # to_relative returns az in (-pi, pi]; the IRF Zeta axis spans [0, 2pi)
        zeta = np.asarray(az.to_value(u.rad), dtype=float).ravel() % (2 * np.pi)

        irf_axes = self._irf.axes

        return build_transfer_matrix(phi_geo=phi_geo,
                                     zeta=zeta,
                                     psichi_bin=self._psichi_sub_bin,
                                     psichi_weight=self._psichi_sub_weight,
                                     phi_samples=self._phi_s,
                                     phi_bin=self._phi_s_bin,
                                     phi_weight=self._phi_s_width,
                                     n_psichi=self._n_psichi,
                                     n_phi=self._n_phi,
                                     irf_shape=self._irf_rel_shape,
                                     theta_range=self._irf_theta_range,
                                     phi_range=self._irf_phi_range,
                                     phi_interp_weights=irf_axes['Phi'].interp_weights,
                                     theta_interp_weights=irf_axes['Theta'].interp_weights,
                                     zeta_interp_weights=irf_axes['Zeta'].interp_weights,
                                     max_pairs=self._max_pairs)

    def _compute_pixel(self, pix: int) -> np.ndarray:
        """
        Integrate the IRF over the absolute measurement bins for the
        direction at the center of NuLambda pixel ``pix``.

        Returns
        -------
        numpy.ndarray
            cm^2, with shape and order of the rest axes.
        """

        n_ei = len(self._ei_keV)
        n_em = len(self._em_edges_keV) - 1
        n_eps = len(self._eps_centers)
        n_cds = self._n_phi * self._n_psichi

        result = np.zeros((n_ei, n_em, n_cds))

        if np.any(self._ei_inside):

            src_vec = np.asarray(hp.pix2vec(self.nside, pix, nest=_is_nested(self._axes['NuLambda'])),
                                 dtype=float)

            slab = self._interp_irf_nulambda(src_vec)

            transfer_t = self._transfer_matrix(src_vec).T.tocsr()

            # Response integrated over each (Phi, PsiChi) bin, still
            # differential in Em, for each IRF Ei bin that is needed.
            # Shape (n_cds, n_eps). It's linear in the IRF contents, so
            # Ei can be interpolated afterwards.
            density_cache = {}

            def density(i_ei):
                if i_ei not in density_cache:
                    d = slab[i_ei].reshape(n_eps, -1)
                    density_cache[i_ei] = np.asarray(transfer_t @ d.T)
                return density_cache[i_ei]

            for j in range(n_ei):

                if not self._ei_inside[j]:
                    continue

                dens = None

                for b, w in zip(self._ei_ibins[:, j], self._ei_iweights[:, j]):
                    if w == 0:
                        continue
                    contrib = w * density(int(b))
                    dens = contrib if dens is None else dens + contrib

                if dens is None:
                    continue

                result[j] = integrate_em_bins(dens,
                                              self._ei_keV[j],
                                              self._em_edges_keV,
                                              self._eps_centers,
                                              self._eps_edges,
                                              IRFRelativeHistUnpolarized._integrate_piecewise_linear)

        result = result.reshape(n_ei, n_em, self._n_phi, self._n_psichi)

        return np.ascontiguousarray(result.transpose(self._transpose), dtype=self.dtype)

    # ------------------------------------------------------------------
    # Full precomputation
    # ------------------------------------------------------------------

    def _pixel_iterator(self, show_progress):

        pixels = range(self.npix)

        if show_progress:
            from tqdm.autonotebook import tqdm
            pixels = tqdm(pixels)

        return pixels

    def to_dr(self, show_progress: bool = False) -> DetectorResponse:
        """
        Compute the response for all NuLambda pixels and load it in
        memory. The cache is neither used nor filled.

        Returns
        -------
        DetectorResponse
        """

        contents = np.empty(self._axes.shape, dtype=self.dtype)

        for pix in self._pixel_iterator(show_progress):
            cached = self._cache.get(pix)
            contents[pix] = cached if cached is not None else self._compute_pixel(pix)

        return DetectorResponse(self._axes,
                                contents=contents,
                                unit=self._unit,
                                copy_contents=False)

    def dump(self, filename, dtype=np.float32, compress: bool = True,
             overwrite: bool = False, show_progress: bool = False):
        """
        Compute the response for all NuLambda pixels and write it as a
        standard (version 2) detector response HDF5 file, which can be
        reopened with :meth:`FullDetectorResponse.open`. Pixels are
        computed and written one at a time.

        Parameters
        ----------
        filename : str or path-like
            Output file. Must have the ``.h5`` extension.
        dtype : numpy dtype, optional
            Data type of the stored contents. Default float32.
        compress : bool, optional
            Use bitshuffle compression (needs ``hdf5plugin``).
        overwrite : bool, optional
            Overwrite an existing file.
        show_progress : bool, optional
            Show a progress bar.
        """

        from .RspConverter import RspConverter

        filename = Path(filename)

        if filename.suffix != '.h5':
            raise ValueError("Output file must have the .h5 extension")

        if filename.exists() and not overwrite:
            raise FileExistsError(f"{filename} exists. Use overwrite=True.")

        if compress:
            import hdf5plugin
            compression = hdf5plugin.Bitshuffle()
        else:
            compression = None

        # Headers, written in alphabetical order so that HEADER_ORDER
        # is the identity (see RspConverter._create_h5)
        headers = {'GENERATOR': self.__class__.__name__,
                   'PHI_SAMPLES': str(self._phi_samples),
                   'PSICHI_SUBDIVISION': str(self._psichi_subdivision)}

        with h5.File(filename, mode='w') as f:

            drm = f.create_group('DRM')
            drm.attrs['VERSION'] = FullDetectorResponse.rsp_version

            header_group = drm.create_group('HEADERS')
            for key in sorted(headers):
                header_group.attrs[key] = headers[key]
            drm.attrs['HEADER_ORDER'] = np.arange(len(headers))

            drm.attrs['UNIT'] = self._unit.to_string()
            drm.attrs['SPARSE'] = False

            axes_group = drm.create_group('AXES')
            self._axes.write(axes_group)

            axes_desc_group = drm.create_group('AXIS_DESCRIPTIONS')
            for label in self._axes.labels:
                axes_desc_group.attrs[label] = RspConverter.axis_description.get(label, "")

            drm.create_dataset('EFF_AREA', data=np.ones(self._axes['Ei'].nbins, dtype=dtype),
                               track_times=False)

            # One chunk per (NuLambda, Ei), as in RspConverter
            chunks = tuple(1 if axis.label in ('NuLambda', 'Ei') else axis.nbins
                           for axis in self._axes)

            counts = drm.create_dataset('COUNTS',
                                        self._axes.shape,
                                        dtype=dtype,
                                        chunks=chunks,
                                        compression=compression,
                                        track_times=False)

            for pix in self._pixel_iterator(show_progress):
                cached = self._cache.get(pix)
                data = cached if cached is not None else self._compute_pixel(pix)
                counts[pix] = data.astype(dtype, copy=False)
