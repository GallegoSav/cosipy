"""
Binned (absolute Compton Data Space) view of a relative-coordinate
histogram IRF, usable everywhere a :class:`FullDetectorResponse` is
expected -- in particular with :class:`BinnedInstrumentResponse` for
binned likelihood analyses.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy import sparse

import h5py as h5
import healpy as hp

import astropy.units as u
from astropy.units import Quantity

from mhealpy import HealpixBase
from histpy import Axis, Axes, HealpixAxis, Histogram
from scoords import SpacecraftFrame

from .FullDetectorResponse import FullDetectorResponse
from .DetectorResponse import DetectorResponse
from .relative_irf_hist import IRFRelativeHistUnpolarized
from .relative_coordinates import RelativeCDSCoordinates
from ._relative_irf_binning import subsample_edges, find_bins, overlap_matrix, em_overlap_fractions

import logging
logger = logging.getLogger(__name__)

__all__ = ["RelativeIRFDetectorResponse"]


def _is_nested(axis) -> bool:
    return str(axis.scheme).upper().startswith('NEST')


class RelativeIRFDetectorResponse(FullDetectorResponse):
    """
    A :class:`FullDetectorResponse` built from the bins of a
    relative-coordinate histogram IRF
    (:class:`~cosipy.response.relative_irf_hist.IRFRelativeHistUnpolarized`),
    without interpolation.

    The relative IRF bins the response in photon direction
    (``NuLambda``) and energy (``Ei``), and in measured quantities
    expressed relative to the photon (``Epsilon``, ``Phi``, ``Theta``,
    ``Zeta``). This class keeps the IRF ``NuLambda`` and ``Ei`` axes as
    they are, and redistributes the content of each relative bin into
    the bins of an absolute Compton Data Space (``Em``, ``Phi``,
    ``PsiChi``) chosen by the user -- typically the axes of the binned
    data. For each ``NuLambda`` pixel this gives the same
    ``[Ei, Em, Phi, PsiChi]`` matrix (in cm^2) that a
    :class:`FullDetectorResponse` read from file provides, so all the
    algorithms of :class:`FullDetectorResponse` that rely on those
    per-pixel matrices are inherited as is
    (:meth:`get_point_source_response`, including the
    scatt-map/inertial-frame path, :meth:`get_extended_source_response`,
    ...), and an instance can be passed to
    :class:`~cosipy.response.BinnedInstrumentResponse`.

    Earth occultation is not a property of the response: as for a
    regular :class:`FullDetectorResponse`, it is handled by the
    ``scatt_map`` (``SpacecraftHistory.get_scatt_map(earth_occ=True)``).

    Redistribution
    --------------
    The response density is assumed constant within each relative bin
    (the same assumption used to store it), and each bin content is
    split in proportion to its overlap with the absolute bins. The
    content is therefore conserved: summed over all the absolute bins,
    the response equals the IRF histogram content of that
    ``(NuLambda, Ei)`` bin, minus the part outside of the measured
    ``Em`` and ``Phi`` ranges. The source direction is the center of
    the ``NuLambda`` pixel, and the photon energy is the ``Ei`` bin
    center (as used by the IRF to compute its ``Epsilon`` phase space).

    - ``Em``: the overlap with each ``Epsilon`` bin
      (``Em = Ei * (1 + Epsilon)``) is computed exactly.
    - ``Phi``, ``PsiChi``: a relative (``Phi``, ``Theta``, ``Zeta``)
      bin is a patch of the sky (a slice of a ring around the source
      direction) whose boundaries never match HEALPix pixels. Its
      overlap with each (``Phi``, ``PsiChi``) bin is estimated by
      sampling: ``phi_samples`` values per IRF ``Phi`` bin and
      ``4**psichi_subdivision`` sub-pixels per ``PsiChi`` pixel. The
      sampled overlaps are normalized per relative bin, so the
      sampling only affects how the content is split between
      neighboring bins, not the total. Relative bins too small to be
      hit by any sample are assigned to the bin containing their
      center. Increase the sampling if the ``PsiChi`` pixels or the
      data ``Phi`` bins are coarse compared to the IRF ``Theta`` bins.

    Normalization
    -------------
    By default only the IRF differential histogram (``IRF`` group of
    the file) is used. If the IRF was loaded with a separate total
    effective area (``AEFF`` group), ``normalize_to_aeff=True``
    rescales each ``(NuLambda, Ei)`` slice so that its total over the
    full measurement space equals the ``AEFF`` effective area averaged
    over that IRF bin -- the normalization used by the unbinned
    likelihood. The ``AEFF`` grid must be nested in the IRF one. See
    :meth:`aeff_ratio`.

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
        the spacecraft frame. Using the IRF ``Phi`` binning is
        recommended.
    phi_samples : int, optional
        Number of samples per IRF ``Phi`` bin. Default 3.
    psichi_subdivision : int, optional
        Each ``PsiChi`` pixel is sampled with ``4**psichi_subdivision``
        sub-pixels. Default 1.
    normalize_to_aeff : bool, optional
        Rescale to the IRF total effective area (``AEFF``). Default
        False.
    cache_size : int or None, optional
        Maximum number of ``NuLambda`` pixels kept in memory (least
        recently used are dropped). ``None`` (default) means no limit,
        ``0`` disables the cache. Each pixel takes
        ``8 * nEi * nEm * nPhi * nPsiChi`` bytes.
    max_pairs : int, optional
        Maximum number of (Phi, PsiChi) samples processed at once, to
        bound the memory used during the computation.
    dtype : numpy dtype, optional
        Data type of the returned contents. Default float64.
    """

    def __init__(self,
                 irf: IRFRelativeHistUnpolarized,
                 measurement_axes: Axes,
                 phi_samples: int = 3,
                 psichi_subdivision: int = 1,
                 normalize_to_aeff: bool = False,
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
        self._normalize_to_aeff = bool(normalize_to_aeff)
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

        # ---- NuLambda and Ei: the IRF ones ----
        irf_nulambda = irf_axes['NuLambda']
        irf_ei = irf_axes['Ei']

        nulambda_axis = HealpixAxis(nside=irf_nulambda.nside,
                                    scheme='nested' if _is_nested(irf_nulambda) else 'ring',
                                    coordsys=SpacecraftFrame(), label='NuLambda')

        energy_axis = Axis(np.asarray(irf_ei.edges, dtype=float) * u.keV, label='Ei', scale=irf_ei.axis_scale)

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

        # Photon energies (keV): same centers the IRF used for its phase space
        self._ei_keV = np.asarray(irf_ei.centers, dtype=float)

        # Measured energy
        self._em_edges_keV = np.asarray(em_axis.edges.to_value(u.keV), dtype=float)
        self._eps_edges = np.asarray(irf_axes['Epsilon'].edges, dtype=float)
        self._eps_widths = np.asarray(irf_axes['Epsilon'].widths, dtype=float)

        # Exact Epsilon -> Em fractions, per Ei. Shape (nEi, nEps, nEm)
        self._em_fractions = np.stack([em_overlap_fractions(ei, self._eps_edges, self._em_edges_keV)
                                       for ei in self._ei_keV])

        # IRF relative axes (rad)
        self._irf_phi_edges = np.asarray(irf_axes['Phi'].edges, dtype=float)
        self._irf_theta_edges = np.asarray(irf_axes['Theta'].edges, dtype=float)
        self._irf_zeta_edges = np.asarray(irf_axes['Zeta'].edges, dtype=float)

        self._irf_rel_shape = (len(self._irf_phi_edges) - 1,
                               len(self._irf_theta_edges) - 1,
                               len(self._irf_zeta_edges) - 1)
        self._n_rel = int(np.prod(self._irf_rel_shape))

        # Phase space of each relative bin (sr * rad), flattened in C order
        phi_m, theta_m, zeta_m = np.meshgrid(self._irf_phi_edges, self._irf_theta_edges, self._irf_zeta_edges,
                                             indexing='ij')

        self._ps_cds = np.asarray(RelativeCDSCoordinates.get_relative_cds_phase_space(
            phi_m[:-1, :-1, :-1], phi_m[1:, :-1, :-1],
            theta_m[:-1, :-1, :-1], theta_m[:-1, 1:, :-1],
            zeta_m[:-1, :-1, :-1], zeta_m[:-1, :-1, 1:]), dtype=float).reshape(-1)

        # Phi samples (rad), per IRF Phi bin, and the data Phi bin they fall in
        phi_s, phi_s_irf_bin, phi_s_width = subsample_edges(self._irf_phi_edges, self._phi_samples)

        data_phi_edges = np.asarray(phi_axis.edges.to_value(u.rad), dtype=float)

        self._phi_s = phi_s
        self._phi_s_irf_bin = phi_s_irf_bin
        self._phi_s_width = phi_s_width
        self._phi_s_data_bin = find_bins(data_phi_edges, phi_s)
        self._data_phi_edges = data_phi_edges

        # PsiChi sub-pixels: unit vectors (local frame), parent bin and solid angle
        nside_psichi = psichi_axis.nside
        npix_psichi = psichi_axis.npix
        nsub = 4 ** self._psichi_subdivision
        nside_sub = nside_psichi * 2 ** self._psichi_subdivision

        parents = np.arange(npix_psichi)
        parents_nest = parents if _is_nested(psichi_axis) else hp.ring2nest(nside_psichi, parents)

        sub_pixels = (parents_nest[:, None] * nsub + np.arange(nsub)[None, :]).ravel()

        self._psichi_nside = nside_psichi
        self._psichi_nest = _is_nested(psichi_axis)
        self._psichi_sub_vec = np.asarray(hp.pix2vec(nside_sub, sub_pixels, nest=True))
        self._psichi_sub_bin = np.repeat(parents, nsub)
        self._psichi_sub_weight = np.full(len(sub_pixels), 4 * np.pi / hp.nside2npix(nside_sub))

        self._n_phi = phi_axis.nbins
        self._n_psichi = npix_psichi

        # Total effective area used for normalize_to_aeff
        self._aeff_avg = self._average_aeff() if self._normalize_to_aeff else None

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
    def normalize_to_aeff(self) -> bool:
        return self._normalize_to_aeff

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
                f"phi_samples = {self._phi_samples}, psichi_subdivision = {self._psichi_subdivision}, "
                f"normalize_to_aeff = {self._normalize_to_aeff})")

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
    # IRF contents
    # ------------------------------------------------------------------

    def _irf_pixel_density(self, pix: int) -> np.ndarray:
        """
        IRF differential effective area for one NuLambda pixel,
        shape (nEi, nEpsilon, n_rel), cm2/sr/rad/keV.
        """

        contents = self._irf.differential_effective_area_hist.contents

        slab = contents[pix]

        if hasattr(slab, 'todense'):
            slab = slab.todense()

        slab = np.asarray(slab, dtype=float)

        return slab.reshape(slab.shape[0], slab.shape[1], -1)

    def _irf_pixel_contents_total(self, pix: int) -> np.ndarray:
        """
        Total IRF histogram content (cm2) per Ei for one NuLambda
        pixel, i.e. the differential effective area integrated over the
        full relative measurement space.
        """

        density = self._irf_pixel_density(pix)

        # Epsilon phase space: Ei * dEpsilon (keV)
        ps_em = self._ei_keV[:, None] * self._eps_widths[None, :]

        return np.einsum('ier,ie,r->i', density, ps_em, self._ps_cds)

    def _average_aeff(self) -> np.ndarray:
        """
        IRF total effective area (``AEFF`` if the IRF was loaded with
        one) averaged over each IRF (NuLambda, Ei) bin.

        Returns
        -------
        numpy.ndarray, shape (npix, nEi), cm2
        """

        tot = self._irf._tot_aeff

        tot_nulambda = tot.axes['NuLambda']
        tot_ei_edges = np.asarray(tot.axes['Ei'].edges, dtype=float)
        tot_contents = np.asarray(tot.contents, dtype=float)

        ei_edges = np.asarray(self._irf.axes['Ei'].edges, dtype=float)

        # ---- Ei: the AEFF edges must subdivide the IRF ones ----
        idx = np.argmin(np.abs(tot_ei_edges[None, :] - ei_edges[:, None]), axis=1)

        nested = np.allclose(tot_ei_edges[idx], ei_edges, rtol=1e-6, atol=0)

        if not nested:
            raise ValueError("normalize_to_aeff requires the AEFF Ei edges to include all the IRF Ei edges. "
                             f"IRF Ei edges: {ei_edges}, AEFF Ei edges: {tot_ei_edges}")

        tot_ei_widths = np.diff(tot_ei_edges)

        aeff_ei = np.empty((tot_contents.shape[0], len(ei_edges) - 1))

        for i in range(len(ei_edges) - 1):
            sl = slice(idx[i], idx[i + 1])
            aeff_ei[:, i] = (np.sum(tot_contents[:, sl] * tot_ei_widths[sl], axis=1)
                             / np.sum(tot_ei_widths[sl]))

        # ---- NuLambda: AEFF nside must be the IRF one times a power of 2 ----
        nside = self.nside
        tot_nside = tot_nulambda.nside

        ratio = tot_nside // nside

        if tot_nside % nside != 0 or ratio & (ratio - 1) != 0:
            raise ValueError(f"normalize_to_aeff requires the AEFF nside ({tot_nside}) to be the IRF nside "
                             f"({nside}) times a power of 2")

        nchildren = ratio ** 2

        # Parent (IRF) pixel of each AEFF pixel, in the IRF scheme
        tot_pix = np.arange(tot_nulambda.npix)
        tot_pix_nest = tot_pix if _is_nested(tot_nulambda) else hp.ring2nest(tot_nside, tot_pix)

        parent_nest = tot_pix_nest // nchildren
        parent = parent_nest if _is_nested(self._axes['NuLambda']) else hp.nest2ring(nside, parent_nest)

        aeff = np.zeros((self.npix, aeff_ei.shape[1]))
        np.add.at(aeff, parent, aeff_ei)

        return aeff / nchildren

    def aeff_ratio(self) -> Histogram:
        """
        Ratio between the IRF total effective area (``AEFF`` if the
        IRF was loaded with one) averaged over each IRF
        ``(NuLambda, Ei)`` bin, and the content of the IRF differential
        histogram in that bin. This is the factor applied by
        ``normalize_to_aeff=True``. NaN where the histogram is empty.

        Returns
        -------
        histpy.Histogram
            Axes ``NuLambda``, ``Ei``.
        """

        aeff = self._aeff_avg if self._aeff_avg is not None else self._average_aeff()

        hist_total = np.array([self._irf_pixel_contents_total(pix) for pix in range(self.npix)])

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(hist_total > 0, aeff / hist_total, np.nan)

        return Histogram(Axes([self._axes['NuLambda'], self._axes['Ei']]), contents=ratio)

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

    def _overlap_matrix(self, src_vec: np.ndarray):
        """
        Fraction of each IRF relative (Phi, Theta, Zeta) bin in each
        absolute (Phi, PsiChi) bin, for a given source direction.

        Returns
        -------
        scipy.sparse.csr_matrix, shape (n_rel, nPhi * nPsiChi)
        """

        n_theta, n_zeta = self._irf_rel_shape[1], self._irf_rel_shape[2]
        n_abs = self._n_phi * self._n_psichi

        relcoords = RelativeCDSCoordinates(src_vec[:, None], pol_convention=self._irf.pol_convention)

        # Geometry of the PsiChi sub-pixels relative to the source
        phi_geo, az = relcoords.to_relative(self._psichi_sub_vec)

        phi_geo = np.asarray(phi_geo.to_value(u.rad), dtype=float).ravel()

        # to_relative returns az in (-pi, pi]; the IRF Zeta axis spans [0, 2pi)
        zeta = np.asarray(az.to_value(u.rad), dtype=float).ravel() % (2 * np.pi)

        zeta_bin = find_bins(self._irf_zeta_edges, zeta)

        # Samples: (Phi sample, PsiChi sub-pixel) pairs, by blocks of Phi samples
        rel_list, abs_list, w_list = [], [], []

        block = max(1, self._max_pairs // max(1, len(phi_geo)))

        for start in range(0, len(self._phi_s), block):

            sl = slice(start, start + block)

            theta = phi_geo[None, :] - self._phi_s[sl, None]
            theta_bin = find_bins(self._irf_theta_edges, theta.ravel()).reshape(theta.shape)

            i_phi, i_sub = np.nonzero((theta_bin >= 0) & (zeta_bin[None, :] >= 0))

            if len(i_phi) == 0:
                continue

            i_phi_global = i_phi + start

            rel = ((self._phi_s_irf_bin[i_phi_global] * n_theta + theta_bin[i_phi, i_sub]) * n_zeta
                   + zeta_bin[i_sub])

            data_phi_bin = self._phi_s_data_bin[i_phi_global]

            absolute = np.where(data_phi_bin >= 0,
                                data_phi_bin * self._n_psichi + self._psichi_sub_bin[i_sub],
                                -1)

            rel_list.append(rel)
            abs_list.append(absolute)
            w_list.append(self._phi_s_width[i_phi_global] * self._psichi_sub_weight[i_sub])

        if rel_list:
            rel = np.concatenate(rel_list)
            absolute = np.concatenate(abs_list)
            weight = np.concatenate(w_list)
        else:
            rel = absolute = np.zeros(0, dtype=int)
            weight = np.zeros(0)

        # Fallback for physical relative bins not hit by any sample:
        # the absolute bin containing the bin center
        sampled = np.bincount(rel, minlength=self._n_rel) > 0
        unsampled = np.nonzero(~sampled & (self._ps_cds > 0))[0]

        fallback = np.full(self._n_rel, -1, dtype=np.int64)

        if len(unsampled):

            a, b, c = np.unravel_index(unsampled, self._irf_rel_shape)

            phi_c = 0.5 * (self._irf_phi_edges[a] + self._irf_phi_edges[a + 1])
            theta_c = 0.5 * (self._irf_theta_edges[b] + self._irf_theta_edges[b + 1])
            zeta_c = 0.5 * (self._irf_zeta_edges[c] + self._irf_zeta_edges[c + 1])

            phi_geo_c = np.clip(phi_c + theta_c, 0, np.pi)

            vec = np.asarray(relcoords.to_cds(phi_geo_c, zeta_c)).reshape(3, -1)

            psichi_pix = hp.vec2pix(self._psichi_nside, vec[0], vec[1], vec[2], nest=self._psichi_nest)

            data_phi_bin = find_bins(self._data_phi_edges, phi_c)

            fallback[unsampled] = np.where(data_phi_bin >= 0, data_phi_bin * self._n_psichi + psichi_pix, -1)

        return overlap_matrix(rel, absolute, weight, self._n_rel, n_abs, fallback)

    def _compute_pixel(self, pix: int) -> np.ndarray:
        """
        Redistribute the IRF contents of NuLambda pixel ``pix`` into
        the absolute measurement bins.

        Returns
        -------
        numpy.ndarray
            cm^2, with shape and order of the rest axes.
        """

        n_ei = len(self._ei_keV)
        n_em = len(self._em_edges_keV) - 1
        n_cds = self._n_phi * self._n_psichi

        density = self._irf_pixel_density(pix)  # (nEi, nEps, n_rel), per keV sr rad

        result = np.zeros((n_ei, n_em, n_cds))

        if np.any(density):

            src_vec = np.asarray(hp.pix2vec(self.nside, pix, nest=_is_nested(self._axes['NuLambda'])),
                                 dtype=float)

            # Content of each relative (Phi, Theta, Zeta) bin -> (Phi, PsiChi) bins,
            # with the (Phi, Theta, Zeta) phase space folded in
            overlap_t = (sparse.diags(self._ps_cds) @ self._overlap_matrix(src_vec)).T.tocsr()

            for i in range(n_ei):

                if not np.any(density[i]):
                    continue

                # Content per (Epsilon, absolute CDS) bin, still per keV of Em: (nEps, n_cds)
                content = np.asarray(overlap_t @ density[i].T).T

                # Epsilon phase space Ei * dEps (keV) and exact Epsilon -> Em fractions
                em_weights = self._em_fractions[i] * (self._ei_keV[i] * self._eps_widths)[:, None]

                result[i] = em_weights.T @ content

            if self._normalize_to_aeff:

                hist_total = self._irf_pixel_contents_total(pix)

                with np.errstate(divide='ignore', invalid='ignore'):
                    factor = np.where(hist_total > 0, self._aeff_avg[pix] / hist_total, 0)

                result *= factor[:, None, None]

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
        memory. The cache is used if available, but not filled.

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
                   'NORMALIZE_TO_AEFF': str(self._normalize_to_aeff),
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
