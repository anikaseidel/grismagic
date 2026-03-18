"""
Unified grism trace computation wrapping all grismagic readers.

Explictly not ported from grizli/grismconf.py:

- Sensitivity curve loading (get_beams, SENS) — tied to grizli's file layout and astropy.table
- JwstDispersionTransform — JWST coordinate rotation to align dispersion with +x; only needed if you're working in grizli's rotated convention
- NIRISS fwcpos filter wheel rotation correction — niche
- Instrument-specific empirical polynomial corrections (the V4/V8 NIRCam offsets in get_beam_trace) — grizli calibrations
- load_grism_config / load_nircam_sensitivity_curve — grizli infrastructure

"""

import warnings

import numpy as np
from .readers import aXeConfReader, GRISMCONFReader, CRDSReader, RomanConfReader
from .wavelengthrange import load_all_ranges


class GrismTrace:
    """
    Unified grism trace calculator.

    Wraps any of the four grismagic readers and exposes a single
    ``get_trace`` method that returns detector-frame trace positions and
    wavelengths for a given source location and set of pixel offsets.

    Parameters
    ----------
    reader : aXeConfReader | GRISMCONFReader | CRDSReader | RomanConfReader
        An already-initialised reader instance.
    filter_name : str, optional
        Filter name (e.g. ``'F200W'``).  When set, ``get_trace`` and
        ``dx_range`` automatically look up ``lam_min`` / ``lam_max`` from the
        CRDS wavelengthrange reference file so you do not have to pass them
        explicitly.  Explicit ``lam_min`` / ``lam_max`` arguments always take
        precedence.
    wavelengthrange_file : str or path-like, optional
        Path to a local wavelengthrange ASDF file.  Overrides the default
        CRDS cache (see :mod:`grismagic.wavelengthrange`).
    check_update : bool
        If ``True``, query CRDS on every use to see whether the operational
        context has changed and re-download the wavelengthrange reference if
        needed.  Default ``False`` (use the cached file without a network
        call).
    instrument : str
        JWST instrument name used when resolving the wavelengthrange
        reference.  Default ``'niriss'``.

    Examples
    --------
    >>> tr = GrismTrace.from_axe("WFC3.G141.conf")
    >>> offset = np.arange(-100, 200)
    >>> x_tr, y_tr, lam = tr.get_trace(507, 507, order="A", offset=offset)

    >>> tr = GrismTrace.from_grismconf("NIRCAM_F444W_modA_R.conf")
    >>> x_tr, y_tr, lam = tr.get_trace(1024, 1024, order="+1", offset=offset)

    >>> tr = GrismTrace.from_crds("specwcs.asdf", filter_name="F200W")
    >>> x_tr, y_tr, lam = tr.get_trace(1024, 1024, order="+1")
    # offset and lam_min / lam_max applied automatically from filter_name

    GrismTrace.from_axe("file.conf")
    GrismTrace.from_grismconf("file.conf")
    GrismTrace.from_crds("file.asdf", filter_name="F200W")
    GrismTrace.from_roman("file.yaml")
    GrismTrace.from_file("file.*")          # auto-detects by extension/content

    x_trace, y_trace, lam = tr.get_trace(x, y, order, offset)

    """

    def __init__(self, reader, filter_name=None, wavelengthrange_file=None,
                 check_update=False, instrument="niriss"):
        if isinstance(reader, aXeConfReader):
            self._kind = "axe"
        elif isinstance(reader, GRISMCONFReader):
            self._kind = "grismconf"
        elif isinstance(reader, CRDSReader):
            self._kind = "crds"
        elif isinstance(reader, RomanConfReader):
            self._kind = "roman"
        else:
            raise TypeError(f"Unsupported reader type: {type(reader).__name__}")
        self.reader = reader
        self.filter_name = filter_name
        self._wavelengthrange_file = wavelengthrange_file
        self._check_update = check_update
        self._instrument = instrument
        self._waverange_table = None  # loaded once on first use: {(filter, order): (lmin, lmax)}

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_axe(cls, conf_file, filter_name=None, wavelengthrange_file=None,
                 check_update=False, instrument="niriss"):
        """Create from an aXe text .conf file."""
        return cls(aXeConfReader(conf_file), filter_name=filter_name,
                   wavelengthrange_file=wavelengthrange_file,
                   check_update=check_update, instrument=instrument)

    @classmethod
    def from_grismconf(cls, conf_file, filter_name=None, wavelengthrange_file=None,
                       check_update=False, instrument="niriss"):
        """Create from a GRISMCONF text .conf file."""
        return cls(GRISMCONFReader(conf_file), filter_name=filter_name,
                   wavelengthrange_file=wavelengthrange_file,
                   check_update=check_update, instrument=instrument)

    @classmethod
    def from_crds(cls, asdf_file, filter_name=None, wavelengthrange_file=None,
                  check_update=False, instrument="niriss"):
        """Create from a JWST CRDS specwcs .asdf file."""
        return cls(CRDSReader(asdf_file), filter_name=filter_name,
                   wavelengthrange_file=wavelengthrange_file,
                   check_update=check_update, instrument=instrument)

    @classmethod
    def from_roman(cls, yaml_file, filter_name=None, wavelengthrange_file=None,
                   check_update=False, instrument="niriss"):
        """Create from a Roman WFI grism YAML file."""
        return cls(RomanConfReader(yaml_file), filter_name=filter_name,
                   wavelengthrange_file=wavelengthrange_file,
                   check_update=check_update, instrument=instrument)

    @classmethod
    def from_file(cls, path, filter_name=None, wavelengthrange_file=None,
                  check_update=False, instrument="niriss"):
        """
        Auto-detect reader type from file extension and content.

        * ``.asdf``            → CRDS
        * ``.yaml`` / ``.yml`` → Roman
        * ``.conf`` with ``DISPX_`` keywords → GRISMCONF
        * ``.conf`` otherwise  → aXe
        """
        kw = dict(filter_name=filter_name, wavelengthrange_file=wavelengthrange_file,
                  check_update=check_update, instrument=instrument)
        path_lower = str(path).lower()
        if path_lower.endswith(".asdf"):
            return cls.from_crds(path, **kw)
        if path_lower.endswith((".yaml", ".yml")):
            return cls.from_roman(path, **kw)
        with open(path) as fh:
            content = fh.read()
        if any(k in content for k in ("DISPX_", "DISPY_", "DISPL_")):
            return cls.from_grismconf(path, **kw)
        return cls.from_axe(path, **kw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def orders(self):
        """
        Available spectral orders.

        For aXe: beam names like ``'A'``, ``'B'``.
        For GRISMCONF / CRDS: signed strings like ``'+1'``, ``'0'``, ``'-1'``.
        For Roman: order strings from the YAML file.
        """
        if self._kind == "axe":
            return self.reader.beams
        return self.reader.orders

    @property
    def fwcpos_ref(self):
        """
        Reference filter wheel position angle (degrees) for NIRISS grisms,
        or ``None`` if the format does not carry this keyword.
        """
        return getattr(self.reader, "fwcpos_ref", None)

    def remove_beam(self, order):
        """
        Remove a spectral order from ``self.orders``.

        Parameters
        ----------
        order : str
            Order identifier as listed in ``self.orders``.
        """
        lst = self.reader.beams if self._kind == "axe" else self.reader.orders
        if order in lst:
            lst.remove(order)

    def _lam_range(self, order, lam_min, lam_max):
        """
        Return ``(lam_min, lam_max)``, filling in from the wavelengthrange
        reference when ``self.filter_name`` is set and neither limit was
        supplied explicitly.

        The reference file is read once on first call and the full table is
        cached on the instance; subsequent calls are pure dict lookups.
        """
        if (lam_min is None and lam_max is None) and self.filter_name is not None:
            if self._waverange_table is None:
                try:
                    self._waverange_table = load_all_ranges(
                        instrument=self._instrument,
                        wavelengthrange_file=self._wavelengthrange_file,
                        check_update=self._check_update,
                    )
                except Exception as exc:
                    warnings.warn(
                        f"Could not load wavelengthrange reference for "
                        f"instrument={self._instrument!r}: {exc}. "
                        "Traces will not be band-limited to the filter bandpass. "
                        "Pass wavelengthrange_file= explicitly, set "
                        "$GRISMAGIC_WAVELENGTHRANGE_FILE, or ensure $CRDS_PATH is set.",
                        stacklevel=4,
                    )
                    self._waverange_table = {}
            try:
                order_str = str(int(str(order).lstrip("+")))
            except ValueError:
                order_str = str(order)
            return self._waverange_table.get(
                (self.filter_name.upper(), order_str), (None, None)
            )
        return lam_min, lam_max

    def offset_range(self, order, x=None, y=None, nt=512, lam_min=None, lam_max=None):
        """
        Pixel extent of the trace along its primary dispersion axis.

        For aXe the range comes directly from the ``BEAM{order}`` entry and is
        position-independent; ``x`` and ``y`` are ignored.  For GRISMCONF /
        CRDS the t parameter is swept at ``(x, y)``; if ``lam_min`` /
        ``lam_max`` are given, the sweep is restricted to the corresponding t
        range via INVDISPL.  For Roman the full wavelength grid is used.
        ``x`` and ``y`` must be supplied for non-aXe formats.

        Parameters
        ----------
        order : str
            Spectral order identifier.
        x, y : float, optional
            Source position (pixels for GRISMCONF / CRDS; mm for Roman).
            Required for non-aXe formats.
        nt : int
            Number of points in the parameter sweep.
        lam_min, lam_max : float, optional
            Wavelength limits (same units as DISPL) used to restrict the t
            sweep to the physical filter bandpass.  If ``None`` and
            ``self.filter_name`` is set, the limits are looked up automatically
            from the wavelengthrange reference.

        Returns
        -------
        lo, hi : float
            Minimum and maximum pixel offset from the source along the primary
            dispersion axis.
        """
        lam_min, lam_max = self._lam_range(order, lam_min, lam_max)
        if self._kind == "axe":
            lo, hi = self.reader.beam_range[order]
            return float(lo), float(hi)
        if x is None or y is None:
            raise ValueError("x and y are required for GRISMCONF / CRDS / Roman")
        if self._kind in ("grismconf", "crds"):
            t = self._t_grid(order, x, y, nt, lam_min, lam_max)
            r = self.reader
            if self._primary_axis(order, x, y, lam_min, lam_max) == 'y':
                vals = r.DISPY(order, x, y, t)
            else:
                vals = r.DISPX(order, x, y, t)
            return float(vals.min()), float(vals.max())
        # Roman
        r = self.reader
        wl_grid = np.linspace(r.wl_min, r.wl_max, nt)
        dx_grid, _ = r.get_trace(order, x, y, wl_grid)
        return float(dx_grid.min()), float(dx_grid.max())

    def get_trace_at_wavelength(self, x, y, order, lam, n_interp=512):
        """
        Compute trace positions at specified wavelengths.

        Parameters
        ----------
        x, y : float
            Source position (pixels for aXe / GRISMCONF / CRDS; mm for Roman).
        order : str
            Spectral order identifier.
        lam : array-like
            Target wavelengths.  Units: Angstrom for aXe and GRISMCONF;
            micron for CRDS and Roman.
        n_interp : int
            Grid size for the numerical offset → wavelength inversion used by the
            aXe format.

        Returns
        -------
        x_trace : np.ndarray
            Absolute x pixel positions.
        y_trace : np.ndarray
            Absolute y pixel positions.
        """
        lam = np.asarray(lam, dtype=float)
        if self._kind == "axe":
            return self._axe_at_wavelength(x, y, order, lam, n_interp)
        if self._kind in ("grismconf", "crds"):
            r = self.reader
            t = r.INVDISPL(order, x, y, lam)
            return x + r.DISPX(order, x, y, t), y + r.DISPY(order, x, y, t)
        # Roman: ids() maps wavelength (micron) directly to trace offsets
        r = self.reader
        dy_mm = r.ids(order, lam, x, y)
        dx_mm = r.xmap(order, x, y) + r.crv(order, dy_mm, x, y)
        return (
            x + dx_mm * r.plate_scale,
            y + (r.ymap(order, x, y) + dy_mm) * r.plate_scale,
        )

    def get_traces(self, xs, ys, order, offset=None, n_lam_roman=512):
        """
        Compute traces for multiple source positions.

        Loops over ``(xs, ys)`` pairs and stacks the results.  See
        ``get_trace`` for parameter and return-value details.

        Parameters
        ----------
        xs, ys : array-like of float
            Source positions, one per source.
        order : str
            Spectral order identifier.
        offset : array-like, optional
            Pixel offsets along the primary dispersion axis (same grid for all
            sources).  If ``None``, derived automatically from ``offset_range``.
        n_lam_roman : int
            Wavelength grid size for Roman inversion.

        Returns
        -------
        x_traces : np.ndarray, shape (n_sources, n_offset)
        y_traces : np.ndarray, shape (n_sources, n_offset)
        lams : np.ndarray, shape (n_sources, n_offset)
        """
        results = [self.get_trace(x, y, order, offset, n_lam_roman) for x, y in zip(xs, ys)]
        return tuple(np.array([r[i] for r in results]) for i in range(3))

    def get_traces_at_wavelength(self, xs, ys, order, lam, n_interp=512):
        """
        Compute trace positions at specified wavelengths for multiple sources.

        Loops over ``(xs, ys)`` pairs and stacks the results.  See
        ``get_trace_at_wavelength`` for parameter and return-value details.

        Parameters
        ----------
        xs, ys : array-like of float
            Source positions, one per source.
        order : str
            Spectral order identifier.
        lam : array-like
            Target wavelengths (same grid for all sources).
        n_interp : int
            Grid size for the aXe numerical inversion.

        Returns
        -------
        x_traces : np.ndarray, shape (n_sources, n_lam)
        y_traces : np.ndarray, shape (n_sources, n_lam)
        """
        results = [self.get_trace_at_wavelength(x, y, order, lam, n_interp) for x, y in zip(xs, ys)]
        return tuple(np.array([r[i] for r in results]) for i in range(2))

    def get_trace(self, x, y, order, offset=None, lam_min=None, lam_max=None, n_lam_roman=512):
        """
        Compute the grism trace for a source at detector position ``(x, y)``.

        Parameters
        ----------
        x, y : float
            Source position.  Detector pixel coordinates for aXe / GRISMCONF /
            CRDS; FPA coordinates in mm for Roman.
        order : str
            Spectral order identifier as listed in ``self.orders``.
        offset : array-like, optional
            Pixel offsets from the source along the primary dispersion axis.
            If ``None``, the range is derived automatically from
            ``offset_range`` using the resolved wavelength limits and integer
            pixel steps are used.
        lam_min, lam_max : float, optional
            Wavelength limits (same units as DISPL) used to restrict the t
            inversion to the physical filter bandpass.  Strongly recommended
            for CRDS / GRISMCONF to avoid unphysical polynomial extrapolation.
            If ``None`` and ``self.filter_name`` is set, the limits are looked
            up automatically from the wavelengthrange reference.
        n_lam_roman : int
            Wavelength grid size used when inverting the Roman dispersion model.

        Returns
        -------
        x_trace : np.ndarray
            Absolute x pixel positions along the trace.
        y_trace : np.ndarray
            Absolute y pixel positions along the trace.
        lam : np.ndarray
            Wavelength along the trace.  Units: Angstrom for aXe and
            GRISMCONF; micron for CRDS and Roman.
        """
        lam_min, lam_max = self._lam_range(order, lam_min, lam_max)
        if offset is None:
            lo, hi = self.offset_range(order, x, y, lam_min=lam_min, lam_max=lam_max)
            if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) > 8192:
                raise ValueError(
                    f"offset_range for order={order!r} returned an unreasonable range "
                    f"({lo:.3g}, {hi:.3g}). This usually means the INVDISPL polynomial "
                    "is being evaluated outside its fitted domain (check that lam_min/"
                    "lam_max units match the reader's wavelength units, or pass "
                    "offset= explicitly)."
                )
            offset = np.arange(int(np.floor(lo)), int(np.ceil(hi)) + 1, dtype=float)
        else:
            offset = np.asarray(offset, dtype=float)
        if self._kind == "axe":
            return self._trace_axe(x, y, order, offset)
        if self._kind in ("grismconf", "crds"):
            return self._trace_grismconf(x, y, order, offset, lam_min, lam_max)
        return self._trace_roman(x, y, order, offset, n_lam=n_lam_roman)

    # ------------------------------------------------------------------
    # Per-format implementations
    # ------------------------------------------------------------------

    def _trace_axe(self, x, y, beam, offset):
        dy, lam = self.reader.get_beam_trace(x, y, offset, beam=beam)
        return x + offset, y + dy, lam

    def _t_grid(self, order, x, y, nt, lam_min, lam_max):
        """Return a t grid restricted to [lam_min, lam_max] via INVDISPL, or [0, 1] if unset."""
        if lam_min is not None and lam_max is not None:
            r = self.reader
            t0 = float(np.clip(r.INVDISPL(order, x, y, lam_min), 0.0, 1.0))
            t1 = float(np.clip(r.INVDISPL(order, x, y, lam_max), 0.0, 1.0))
            return np.linspace(min(t0, t1), max(t0, t1), nt)
        return np.linspace(0, 1, nt)

    def _primary_axis(self, order, x, y, lam_min=None, lam_max=None, nt=64):
        """Return 'x' (row grism) or 'y' (column grism) based on which DISP has larger range."""
        t = self._t_grid(order, x, y, nt, lam_min, lam_max)
        r = self.reader
        if np.ptp(r.DISPY(order, x, y, t)) > np.ptp(r.DISPX(order, x, y, t)):
            return 'y'
        return 'x'

    def _trace_grismconf(self, x, y, order, offset, lam_min=None, lam_max=None, nt=512):
        r = self.reader
        # Build the forward trace on a fine t grid, then directly interpolate
        # the secondary axis and wavelength from the primary axis.  This avoids
        # amplifying interpolation noise through a possibly non-monotone secondary
        # polynomial (e.g. the order-0 DISPY quadratic in GR150C).
        #
        # Note: INVDISPX and INVDISPY are intentionally NOT used here.  Neither
        # CRDS ASDF specwcs files nor GRISMCONF files store analytic inverses
        # for DISPX/DISPY (only INVDISPL is provided in either format), so any
        # inversion of DISPX/DISPY would be numerical anyway.  Additionally,
        # the forward polynomials can be non-monotone in t (e.g. the order-0
        # DISPY quadratic in GR150C), making numerical inversion via interp
        # unstable.  The tabular approach below (matching what the JWST pipeline
        # does via Tabular1D) is stable for all orders.
        #
        # If the input polynomials were guaranteed to be monotone in t (which
        # they are for most orders but not order 0), one could instead invert
        # analytically:
        #   t = r.INVDISPX(order, x, y, offset)   # x-primary
        #   y_trace = y + r.DISPY(order, x, y, t)
        #   lam     = r.DISPL(order, x, y, t)
        # or equivalently for y-primary using INVDISPY.  This would be faster
        # but relies on well-conditioned, monotone forward polynomials.
        # Restrict the t grid to the wavelength range of the filter.  Outside
        # that range the polynomials can be non-monotone, which causes argsort +
        # interp to produce erratic secondary-axis values for offsets that happen
        # to land in those non-physical regions.  _t_grid falls back to [0, 1]
        # when lam_min / lam_max are both None.
        t0 = self._t_grid(order, x, y, nt, lam_min, lam_max)
        dispx0 = r.DISPX(order, x, y, t0)
        dispy0 = r.DISPY(order, x, y, t0)
        displ0 = r.DISPL(order, x, y, t0)

        if self._primary_axis(order, x, y, lam_min, lam_max) == 'y':
            so = np.argsort(dispy0)
            x_trace = x + np.interp(offset, dispy0[so], dispx0[so])
            y_trace = y + offset
            lam = np.interp(offset, dispy0[so], displ0[so])
        else:
            so = np.argsort(dispx0)
            x_trace = x + offset
            y_trace = y + np.interp(offset, dispx0[so], dispy0[so])
            lam = np.interp(offset, dispx0[so], displ0[so])

        if lam_min is not None or lam_max is not None:
            outside = np.zeros(len(lam), dtype=bool)
            if lam_min is not None:
                outside |= lam < lam_min
            if lam_max is not None:
                outside |= lam > lam_max
            x_trace = np.where(outside, np.nan, x_trace)
            y_trace = np.where(outside, np.nan, y_trace)
            lam = np.where(outside, np.nan, lam)
        return x_trace, y_trace, lam

    def _axe_at_wavelength(self, x, y, beam, lam, n_interp):
        lo, hi = self.offset_range(beam)
        offset_grid = np.linspace(lo, hi, n_interp)
        _, _, lam_grid = self._trace_axe(x, y, beam, offset_grid)
        so = np.argsort(lam_grid)
        offset = np.interp(lam, lam_grid[so], offset_grid[so])
        x_trace, y_trace, _ = self._trace_axe(x, y, beam, offset)
        return x_trace, y_trace

    def _trace_roman(self, x, y, order, offset, n_lam=512):
        r = self.reader
        wl_grid = np.linspace(r.wl_min, r.wl_max, n_lam)
        dx_grid, dy_grid = r.get_trace(order, x, y, wl_grid)
        so = np.argsort(dx_grid)
        y_trace = y + np.interp(offset, dx_grid[so], dy_grid[so])
        lam = np.interp(offset, dx_grid[so], wl_grid[so])
        return x + offset, y_trace, lam
