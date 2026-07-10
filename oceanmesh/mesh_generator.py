import datetime
import logging
import os
import time

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import scipy.sparse as spsparse
from _constrained_delaunay_class import (
    ConstrainedDelaunayTriangulation as CDT,
)
from _delaunay_class import DelaunayTriangulation as DT
from _fast_geometry import unique_edges
from pyproj import CRS

from .clean import _external_topology
from .edgefx import multiscale_sizing_function
from .fix_mesh import fix_mesh, simp_qual
from .grid import Grid
from .region import (
    to_lat_lon,
    to_stereo,
    bbox_contains,
    validate_crs_compatible,
    get_crs_string,
    is_global_bbox,
)
from .signed_distance_function import Domain, multiscale_signed_distance_function

logger = logging.getLogger(__name__)

__all__ = [
    "generate_mesh",
    "generate_multiscale_mesh",
    "plot_mesh_connectivity",
    "plot_mesh_bathy",
    "write_to_fort14",
    "write_to_t3s",
]


def write_to_fort14(
    points,
    cells,
    filepath,
    topobathymetry=None,
    project_name="Created with oceanmesh",
    flip_bathymetry=False,
):
    """
    Parameters
    -----------
    points (numpy.ndarray): An array of shape (np, 2) containing the x, y coordinates of the mesh nodes.
    cells (numpy.ndarray): An array of shape (ne, 3) containing the indices of the nodes that form each mesh element.
    filepath (str): The file path to write the fort.14 file to.
    topobathymetry (numpy.ndarray): An array of shape (np, 1) containing the topobathymetry values at each node.
    project_name (str): The name of the project to be written to the fort.14 file.
    flip_bathymetry (bool): If True, the bathymetry values will be multiplied by -1.

    Returns:
    --------
    points (numpy.ndarray): An array of shape (np, 2) containing the x, y coordinates of the mesh nodes.
    cells (numpy.ndarray): An array of shape (ne, 3) containing the indices of the nodes that form each mesh element.
    filepath (str): The file path to write the fort.14 file to.
    """
    logger.info("Exporting mesh to fort.14 file...")

    # Calculate number of nodes and elements
    npoints = np.size(points, 0)
    nelements = np.size(cells, 0)

    if topobathymetry is not None:
        assert (
            len(topobathymetry) == npoints
        ), "topobathymetry must be the same length as points"
    else:
        topobathymetry = np.zeros((npoints, 1))

    if flip_bathymetry:
        topobathymetry *= -1

    # Shift cell indices by 1 (fort.14 uses 1-based indexing)
    cells += 1

    # Open file for writing
    with open(filepath, "w") as f_id:
        # Write mesh name
        if flip_bathymetry:
            f_id.write(f"{project_name} (bathymetry flipped) \n")
        else:
            f_id.write(f"{project_name} \n")

        # Write number of nodes and elements
        np.savetxt(
            f_id,
            np.column_stack((nelements, npoints)),
            delimiter=" ",
            fmt="%i",
            newline="\n",
        )

        # Write node coordinates
        for k in range(npoints):
            np.savetxt(
                f_id,
                np.column_stack((k + 1, points[k][0], points[k][1], topobathymetry[k])),
                delimiter=" ",
                fmt="%i %f %f %f",
                newline="\n",
            )

        # Write element connectivity
        for k in range(nelements):
            np.savetxt(
                f_id,
                np.column_stack((k + 1, 3, cells[k][0], cells[k][1], cells[k][2])),
                delimiter=" ",
                fmt="%i %i %i %i %i ",
                newline="\n",
            )

        # Write zero for each boundary condition (4 total)
        for k in range(4):
            f_id.write("%d \n" % 0)

    return f"Wrote the mesh to {filepath}..."


def write_to_t3s(points, cells, filepath):
    """
    Write mesh data to a t3s file.

    Parameters:
    points (numpy.ndarray): An array of shape (np, 2) containing the x, y coordinates of the mesh nodes.
    cells (numpy.ndarray): An array of shape (ne, 3) containing the indices of the nodes that form each mesh element.
    filepath (str): The file path to write the t3s file to.

    Returns:
    None
    """
    logger.info("Exporting mesh to t3s file...")

    # Calculate number of nodes and elements
    npoints = np.size(points, 0)
    nelements = np.size(cells, 0)

    # Open file for writing
    with open(filepath, "w") as f_id:
        # Write header
        today = datetime.datetime.now()
        date_time = today.strftime("%m/%d/%Y, %H:%M:%S")
        t3head = (
            """#########################################################################\n
        :FileType t3s ASCII EnSim 1.0\n
        # Canadian Hydraulics Centre/National Research Council (c) 1998-2004\n
        # DataType 2D T3 Scalar Mesh\n
        #
        :Application BlueKenue\n
        :Version 3.0.44\n
        :WrittenBy pyoceanmesh\n
        :CreationDate """
            + date_time
            + """\n
        #
        #------------------------------------------------------------------------\n
        #
        :Projection Cartesian\n
        :Ellipsoid Unknown\n
        #
        :NodeCount """
            + str(npoints)
            + """\n
        :ElementCount """
            + str(nelements)
            + """\n
        :ElementType T3\n
        #
        :EndHeader"""
        )  # END HEADER
        t3head = os.linesep.join([s for s in t3head.splitlines() if s])
        f_id.write(t3head)
        f_id.write("\n")

        # Write node coordinates
        for k in range(npoints):
            np.savetxt(
                f_id,
                np.column_stack((points[k][0], points[k][1], 0.0)),
                delimiter=" ",
                fmt="%f %f %f",
                newline="\n",
            )

        # Write element connectivity
        for k in range(nelements):
            np.savetxt(
                f_id,
                np.column_stack((cells[k][0], cells[k][1], cells[k][2])),
                delimiter=" ",
                fmt="%i %i %i ",
                newline="\n",
            )

    return f"Wrote the mesh to {filepath}..."


def plot_mesh_connectivity(points, cells, show_plot=True):
    """Plot the mesh connectivity using matplotlib's triplot function.
    Parameters
    ----------

    points : numpy.ndarray
        A 2D array containing the x and y coordinates of the points.
    cells : numpy.ndarray
        A 2D array containing the connectivity information for the triangles.
    show_plot : bool, optional
        Whether to show the plot or not. The default is True.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    triang = tri.Triangulation(points[:, 0], points[:, 1], cells)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.triplot(triang, lw=0.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Mesh connectivity")
    if show_plot:
        plt.show(block=False)
    return ax


def plot_mesh_bathy(points, bathymetry, connectivity, show_plot=True):
    """
    Create a tricontourf plot of the bathymetry data associated with the points,
    using the triangle connectivity information to plot the contours.

    Parameters
    ----------
    points : numpy.ndarray
        A 2D array containing the x and y coordinates of the points.
    bathymetry : numpy.ndarray
        A 1D array containing the bathymetry values associated with each point.
    connectivity : numpy.ndarray
        A 2D array containing the connectivity information for the triangles.
    show_plot : bool, optional
        Whether or not to display the plot. Default is True.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        The axis handle of the plot.

    """
    # Create a Triangulation object using the points and connectivity table
    triangulation = tri.Triangulation(points[:, 0], points[:, 1], connectivity)

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the tricontourf
    tricontourf = ax.tricontourf(triangulation, bathymetry, cmap="jet")

    # Add colorbar
    plt.colorbar(tricontourf)

    # Set axis labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Set title
    ax.set_title("Mesh Topobathymetry")

    # Show the plot if requested
    if show_plot:
        plt.show()

    return ax


def _parse_kwargs(kwargs):
    for key in kwargs:
        if key in {
            "nscreen",
            "max_iter",
            "seed",
            "pfix",
            "egfix",
            "points",
            "domain",
            "edge_length",
            "bbox",
            "min_edge_length",
            "plot",
            "blend_width",
        "enforce_min",
            "blend_polynomial",
            "blend_max_iter",
            "blend_nnear",
            "lock_boundary",
            "pseudo_dt",
            "stereo",
            "force_function",
            "improve",
            "improve_every",
            "qual_tol",
            "exit_quality",
            "max_stalls",
        "cleanup",
        "cpp",
            "heal_fixed_edges_every",
            "rewind_threshold",
        }:
            pass
        else:
            raise ValueError(
                "Option %s with parameter %s not recognized " % (key, kwargs[key])
            )


def _check_bbox(bbox):
    assert isinstance(bbox, tuple), "`bbox` must be a tuple"
    assert int(len(bbox) / 2), "`dim` must be 2"


def _find_global_domain(domains, errors):
    """Identify the global (stereo=True) domain if present.

    Returns
    -------
    global_domain: Domain | None
    """

    stereo_flags = [getattr(d, "stereo", False) for d in domains]
    global_indices = [i for i, s in enumerate(stereo_flags) if s]

    global_domain = None
    if len(global_indices) > 1:
        errors.append("Only one global (stereo=True) domain permitted.")
    elif len(global_indices) == 1:
        if global_indices[0] != 0:
            errors.append("Global domain must be the first (coarsest) domain in list.")
        global_domain = domains[global_indices[0]]

    return global_domain


def _detect_implicit_global_domain(domains):
    """Detect implicit global-like domain: EPSG:4326 + global bbox but stereo=False."""

    for i, d in enumerate(domains):
        dcrs = getattr(d, "crs", None)
        try:
            if (
                dcrs is not None
                and CRS.from_user_input(dcrs).to_epsg() == 4326
                and is_global_bbox(d.bbox)
            ):
                return i
        except Exception:
            # If CRS can't be parsed, skip implicit detection for this domain
            continue
    return None


def _has_any_crs_metadata(domains, edge_lengths):
    domain_crs_list = [getattr(d, "crs", None) for d in domains]
    edge_crs_list = [
        getattr(el, "crs", None) if hasattr(el, "crs") else None for el in edge_lengths
    ]
    return any(c is not None for c in domain_crs_list) or any(
        c is not None for c in edge_crs_list
    )


def _validate_crs_presence(domains, errors):
    for i, d in enumerate(domains):
        if getattr(d, "crs", None) is None:
            errors.append(
                f"Domain #{i} missing CRS metadata. Provide CRS via Shoreline(crs=...) to enable compatibility checks."
            )


def _validate_global_domain_and_regions(global_domain, domains, errors):
    """Validate CRS requirements, containment, and CRS mixing rules when a global domain is present."""

    gcrs = getattr(global_domain, "crs", None)
    if gcrs is not None:
        try:
            gcrs_str = get_crs_string(gcrs)
        except Exception:
            gcrs_str = str(gcrs)

        # Explicitly require global CRS be EPSG:4326
        try:
            parsed = CRS.from_user_input(gcrs)
            if parsed.to_epsg() != 4326:
                errors.append(
                    f"Global domain CRS {gcrs_str} must be EPSG:4326 for global+regional multiscale meshing."
                )
        except Exception:
            errors.append(
                f"Global domain CRS '{gcrs_str}' could not be parsed; expected EPSG:4326."
            )

    # Containment checks + regional stereo checks + CRS compatibility
    for i, d in enumerate(domains[1:], start=1):
        g_bbox = global_domain.bbox
        d_bbox = d.bbox
        try:
            if getattr(global_domain, "stereo", False):
                lon_min, lon_max, lat_min, lat_max = d_bbox
                reg_corners_lon = [lon_min, lon_max, lon_max, lon_min]
                reg_corners_lat = [lat_min, lat_min, lat_max, lat_max]
                sx, sy = to_stereo(np.array(reg_corners_lon), np.array(reg_corners_lat))
                stereo_reg_bbox = (
                    float(np.min(sx)),
                    float(np.max(sx)),
                    float(np.min(sy)),
                    float(np.max(sy)),
                )
                if not bbox_contains(g_bbox, stereo_reg_bbox):
                    errors.append(
                        f"Regional domain #{i} bbox {d_bbox} (lat/lon) not contained within global stereo bbox {g_bbox}."
                    )
            else:
                if not bbox_contains(g_bbox, d_bbox):
                    errors.append(
                        f"Regional domain #{i} bbox {d_bbox} not contained within global bbox {g_bbox}."
                    )
        except Exception:
            errors.append(
                f"Regional domain #{i} containment check failed due to transformation error; verify CRS and stereo settings."
            )

        if getattr(d, "stereo", False):
            errors.append(
                f"Regional domain #{i} has stereo=True; only the global domain may set stereo=True."
            )

        ok_crs, msg_crs = validate_crs_compatible(
            getattr(global_domain, "crs", None), getattr(d, "crs", None)
        )
        if not ok_crs:
            errors.append(msg_crs)


def _validate_implicit_global_mixing(domains, implicit_global_idx, errors):
    """Warn/error when a global-like EPSG:4326 domain is mixed with other CRSs without stereo=True."""

    ig = domains[implicit_global_idx]
    if getattr(ig, "stereo", False):
        return

    try:
        ig_crs = (
            CRS.from_user_input(ig.crs)
            if getattr(ig, "crs", None) is not None
            else None
        )
        for j, d in enumerate(domains):
            if j == implicit_global_idx:
                continue
            dcrs = getattr(d, "crs", None)
            if dcrs is None or ig_crs is None:
                continue
            if not ig_crs.equals(CRS.from_user_input(dcrs)):
                errors.append(
                    "Detected global-like EPSG:4326 domain without stereo=True mixed with other CRS. "
                    "Set stereo=True on the global domain and place it first in the list."
                )
                break
    except Exception:
        # If CRS parsing fails, skip this implicit enforcement
        return


def _validate_edge_length_crs(domains, edge_lengths, errors):
    for i, (d, el) in enumerate(zip(domains, edge_lengths)):
        if not hasattr(el, "crs"):
            continue

        el_crs = getattr(el, "crs", None)
        d_crs = getattr(d, "crs", None)
        if el_crs is None or d_crs is None:
            continue

        try:
            if not CRS.from_user_input(el_crs).equals(CRS.from_user_input(d_crs)):
                errors.append(
                    f"Edge length #{i} CRS {get_crs_string(el_crs)} does not match domain CRS {get_crs_string(d_crs)}."
                )
        except Exception:
            errors.append(
                f"Edge length #{i} CRS could not be compared to domain CRS (el={get_crs_string(el_crs)}, domain={get_crs_string(d_crs)})."
            )


def _validate_multiscale_domains(domains, edge_lengths):  # noqa: C901
    """Validate domain & sizing function compatibility for multiscale meshing.

    Checks performed:
      1. Presence of CRS on all domains.
      2. Global domain (stereo=True) ordering: must be first if present.
      3. Bbox containment for regional domains within global domain when mixing.
      4. CRS compatibility (global EPSG:4326, regional geographic or projected).
      5. Edge length Grid CRS matches corresponding domain CRS.
      6. Stereo flag usage: only global domain should have stereo=True.

    Returns
    -------
    (ok: bool, errors: list[str])
    """
    errors = []

    if len(domains) != len(edge_lengths):
        errors.append("Number of domains and edge_lengths differ.")
        return False, errors

    global_domain = _find_global_domain(domains, errors)
    implicit_global_idx = _detect_implicit_global_domain(domains)
    has_any_crs = _has_any_crs_metadata(domains, edge_lengths)

    # If no CRS anywhere and no implicit global detected, allow bbox-only flows without error
    if has_any_crs or implicit_global_idx is not None:
        _validate_crs_presence(domains, errors)

    # If we have a global domain, validate CRS and containment
    if global_domain is not None:
        _validate_global_domain_and_regions(global_domain, domains, errors)

    # Implicit global-like domain with EPSG:4326 but stereo=False mixing with different CRS
    if implicit_global_idx is not None:
        _validate_implicit_global_mixing(domains, implicit_global_idx, errors)

    # Edge length CRS matching
    _validate_edge_length_crs(domains, edge_lengths, errors)

    return len(errors) == 0, errors


def _sanitize_smoothed_sizing_grids(edge_lengths_smoothed):
    """Ensure each smoothed sizing Grid has a positive finite hmin.

    The multiscale sizing blender can produce Grid objects with missing/invalid
    `hmin`. This helper recomputes `hmin` from the underlying grid values.
    """

    for i, el in enumerate(edge_lengths_smoothed):
        _sanitize_smoothed_sizing_grid(i, el)


def _sanitize_smoothed_sizing_grid(i, el):
    if not isinstance(el, Grid):
        return

    hmin = getattr(el, "hmin", None)
    if hmin is not None and np.isfinite(hmin) and hmin > 0:
        return

    vals = el.values
    if np.ma.isMaskedArray(vals):
        vals = np.ma.filled(vals, np.nan)
    vals = np.asarray(vals)
    pos = vals[np.isfinite(vals) & (vals > 0)]
    if pos.size > 0:
        el.hmin = float(np.nanmin(pos))
        logger.warning(
            f"Sizing grid #{i} had invalid hmin; recomputed fallback hmin={el.hmin:.3f}"
        )
        return

    raise ValueError(
        f"Sizing grid #{i} contains no positive values to determine a minimum edge length."
    )


# NOTE: stereo-aware sizing wrapper removed per verification comment; sizing
# functions are always evaluated on lat/lon points supplied by generate_mesh.


def generate_multiscale_mesh(domains, edge_lengths, **kwargs):
    r"""Generate a 2D triangular mesh using callbacks to several
    sizing functions `edge_lengths` and several signed distance functions
    See the kwargs for `generate_mesh`.

    This function supports both regional multiscale meshing (multiple nested
    domains in the same projection) and global+regional multiscale meshing
    (a global domain in stereographic projection with one or more regional
    refinement zones defined in WGS84). For global+regional workflows,
    coordinate transformations between WGS84 (EPSG:4326) and stereographic
    space are handled automatically during mesh generation. Users define all
    sizing functions on latitude/longitude grids; the mesher manages the
    projection conversions transparently when the first (global) domain has
    `stereo=True`.

    Parameters
    ----------
    domains: A list of function objects.
        A list of functions that takes a point and returns the signed nearest distance to the domain boundary Ω.
    edge_lengths: A function object.
        A list of functions that can evalulate a point and return a mesh size.
    \**kwargs:
        See below for kwargs in addition to the ones available for `generate_mesh`

    Requirements for mixing global and regional domains
    ---------------------------------------------------
    - Global domain must be first in the list and must use EPSG:4326 with stereo=True
    - Regional domains must not set stereo=True
    - Each regional domain bbox must be fully contained by the global domain bbox
    - All domains and sizing Grid objects must supply CRS metadata; each Grid CRS must match its domain CRS
    - Global+regional CRS mixing supported only when global=EPSG:4326 and regional is geographic or projected
        - Coordinate transformation workflow: sizing functions are defined in EPSG:4326; the global mesh is generated in stereographic space; automatic conversions applied during sizing evaluation
        - The global domain requires two shoreline datasets: one in lat/lon (for sizing functions), one in stereographic (for the meshing boundary)

        Automatic coordinate handling for global+regional meshing
        -----------------------------------------------------------
        When the first domain has `stereo=True`, this function automatically:
            * Detects the global+regional mixing scenario during validation.
            * Transforms query points between stereographic and lat/lon when evaluating regional sizing grids.
            * Applies stereographic distortion corrections to sizing values where needed.
            * Propagates the `stereo=True` flag to the final blending/union mesh generation step.

        This ensures that regional sizing functions (defined in WGS84) interact correctly with a global mesh generated in stereographic space. Users do not need to manually handle coordinate conversions.

        Example
        -------
        See the README section 'Global mesh generation with regional refinement' for
        a complete example demonstrating how to merge a regional mesh (e.g., Australia)
        into a global mesh.

    :Keyword Arguments:
        * *blend_width* (``float``) --
                The width of the element size transition region between nest and parent
        * *blend_polynomial* (``int``) --
                The rate of transition scales with 1/dist^blend_polynomial
        * *blend_max_iter* (``int``) --
                The number of mesh generation iterations to blend the nest and parent.
        * *blend_nnear* (``int``) --
                The number of nearest neighbors in the IDW interpolation.
    * *stereo* (``bool``) --
        Note: The stereo parameter for the final union/blending step is inferred from the domain
        metadata (global domain first with stereo=True). Users typically should not set this
        explicitly for multiscale workflows.

    Notes
    -----
    * Regional-only multiscale meshing (no global domain) requires all domains share a compatible CRS.
    * Global+regional meshing follows a two-step workflow: sizing in EPSG:4326, global meshing in stereographic space.
    * Validation errors provide detailed guidance (CRS mismatches, bbox containment, stereo flag misuse).
    * Domain metadata (CRS, stereo flags) is collected internally to manage automatic coordinate transformations.

    """
    assert (
        len(domains) > 1 and len(edge_lengths) > 1
    ), "This function takes a list of domains and sizing functions"
    assert len(domains) == len(
        edge_lengths
    ), "The same number of domains must be passed as sizing functions"

    # Perform validation prior to any mesh generation steps
    ok, verrors = _validate_multiscale_domains(domains, edge_lengths)
    if not ok:
        formatted = "\n - " + "\n - ".join(verrors)
        raise ValueError(
            "Multiscale domain validation failed with the following issues:"
            + formatted
            + "\nGuidance: Ensure a single global domain (stereo=True, EPSG:4326) precedes regional domains; supply CRS metadata via Shoreline; regional bboxes must lie within global bbox; sizing Grid CRS must match domain CRS."
        )
    opts = {
        "max_iter": 100,
        "seed": 0,
        "pfix": None,
        "points": None,
        "min_edge_length": None,
        "plot": 999999,
        "blend_width": 2500,
        "blend_polynomial": 2,
        "blend_max_iter": 20,
        "blend_nnear": 256,
        "lock_boundary": True,
    }
    opts.update(kwargs)
    # pfix/egfix are forwarded ONLY to the first (outermost) nest
    # and to the final blending pass — injecting them into inner
    # nests whose domains do not contain them corrupts those
    # meshes.
    ms_pfix = kwargs.pop("pfix", None)
    ms_egfix = kwargs.pop("egfix", None)
    _ms_gradation = kwargs.pop("gradation", 0.15)
    _parse_kwargs(kwargs)

    # Build domain metadata for stereo/CRS awareness during blending
    domain_metadata = {
        "stereo_flags": [getattr(d, "stereo", False) for d in domains],
        "crs_list": [getattr(d, "crs", None) for d in domains],
        # Consider the first domain as potential global parent
        "global_stereo": bool(getattr(domains[0], "stereo", False)),
    }

    master_edge_length, edge_lengths_smoothed = multiscale_sizing_function(
        edge_lengths,
        blend_width=opts["blend_width"],
        nnear=opts["blend_nnear"],
        p=opts["blend_polynomial"],
        domain_metadata=domain_metadata,
        gradation=_ms_gradation,
        enforce_min=bool(kwargs.get("enforce_min", True)),
    )

    _sanitize_smoothed_sizing_grids(edge_lengths_smoothed)

    # ---- OM2D single-loop multiscale (meshgen.m) -----------------
    # One distmesh loop over ALL nests: composite fd assigns each
    # point the sdf of the DEEPEST box containing it (dpoly.m box
    # loop); composite fh evaluates bar midpoints with the deepest
    # box's (smooth_outer-relaxed) sizing; seeding runs per box
    # with each box's own h0 anchor, excluding deeper-box regions
    # (meshgen.m:664-748). No inter-nest merge pass exists in the
    # .m (split_bars.m is dead code there).
    grids = edge_lengths_smoothed
    boxes = [g.bbox for g in grids]
    h0s = np.array([float(g.hmin) for g in grids])

    from . import edges as _om_edges
    from .geometry import inpoly2 as _inpoly2

    _rings = [getattr(d, "boubox_ring", None) for d in domains]

    def _in_region(q, k):
        # dpoly.m membership: inpoly against the box's boubox ring
        # (polygon nests); rectangle fallback when no ring is
        # attached (create_bbox domains)
        ring = _rings[k]
        if ring is not None and len(ring) > 3:
            part = np.vstack([np.asarray(ring, float),
                              [[np.nan, np.nan]]])
            e = _om_edges.get_poly_edges(part)
            ins, _ = _inpoly2(q, np.nan_to_num(part), e)
            return ins
        x0, x1, y0, y1 = boxes[k]
        return ((q[:, 0] >= x0) & (q[:, 0] <= x1)
                & (q[:, 1] >= y0) & (q[:, 1] <= y1))

    for g in grids:
        g.extrapolate = True
        g.build_interpolant()

    def fh_comp(q):
        q = np.asarray(q, dtype=float)
        v = grids[0].eval(q)
        for k in range(1, len(grids)):
            m = _in_region(q, k)
            if m.any():
                v[m] = grids[k].eval(q[m])
        return v

    def fd_comp(q):
        q = np.asarray(q, dtype=float)
        d = np.full(len(q), 1.0)
        for k in range(len(domains)):
            m = _in_region(q, k)
            for kd in range(k + 1, len(domains)):
                m &= ~_in_region(q, kd)
            if m.any():
                d[m] = domains[k].eval(q[m])
        return d

    # per-box equilateral seeding (meshgen.m:664-748)
    geps_ms = 1e-12 * float(np.amin(h0s))
    rng_pts = []
    for k, (dom, g) in enumerate(zip(domains, grids)):
        x0, x1, y0, y1 = boxes[k]
        h = h0s[k]
        dxs = 2.0 / np.sqrt(3.0) * h
        ys = np.arange(y0, y1 + h, h)
        rows = []
        for i, yv in enumerate(ys):
            # meshgen.m:697-704 builds rows by METRIC length
            # (m_lldist): in longitude degrees the column step is
            # dxs/cos(lat); a plain-degree lattice over-seeds by
            # 1/cos(lat) (x1.32 at 40.75N on Example_3 nest 2)
            _c = max(np.cos(np.deg2rad(yv)), 0.05)
            dxr = dxs / _c
            xs = np.arange(x0 + (0.5 * dxr if i % 2 else 0.0),
                           x1 + dxr, dxr)
            rows.append(np.column_stack(
                [xs, np.full(len(xs), yv)]))
        p1 = np.vstack(rows)
        # exclude deeper-box regions (they seed themselves finer)
        excl = np.zeros(len(p1), dtype=bool)
        for kd in range(k + 1, len(domains)):
            excl |= _in_region(p1, kd)
        p1 = p1[~excl]
        p1 = p1[dom.eval(p1) < geps_ms]
        r0 = np.asarray(g.eval(p1), dtype=float)
        keep = np.random.rand(len(p1)) < (h / r0) ** 2
        p1 = p1[keep]
        logger.info(f"nest #{k}: seeded {len(p1)} points (h0={h})")
        rng_pts.append(p1)
        if k == 0:
            ring = getattr(dom, "boubox_ring", None)
            if ring is not None and len(ring):
                ring = np.asarray(ring, dtype=float)
                ring = ring[fd_comp(ring) < geps_ms]
                if len(ring):
                    r0 = np.asarray(g.eval(ring), dtype=float)
                    keep = np.random.rand(len(ring)) < (h / r0) ** 2
                    rng_pts.append(ring[keep])
    p0 = np.vstack(rng_pts)
    logger.info(f"single-loop multiscale: {len(p0)} initial points")

    _kwargs = dict(kwargs)
    if ms_pfix is not None:
        _kwargs["pfix"] = ms_pfix
        if ms_egfix is not None:
            _kwargs["egfix"] = ms_egfix
    _union_bbox = (min(b[0] for b in boxes),
                   max(b[1] for b in boxes),
                   min(b[2] for b in boxes),
                   max(b[3] for b in boxes))
    _p, _t = generate_mesh(
        domain=fd_comp,
        edge_length=fh_comp,
        bbox=_union_bbox,
        min_edge_length=h0s,
        points=p0,
        **_kwargs,
    )

    return _p, _t


def generate_mesh(domain, edge_length, **kwargs):
    r"""Generate a 2D triangular mesh using callbacks to a
        sizing function `edge_length` and signed distance function.

    Parameters
    ----------
    domain: A function object.
        A function that takes a point and returns the signed nearest distance to the domain boundary Ω.
    edge_length: A function object.
        A function that can evalulate a point and return a mesh size.
    \**kwargs:
        See below

    :Keyword Arguments:
        * *bbox* (``tuple``) --
            Bounding box containing domain extents. REQUIRED IF NOT USING :class:`edge_length`
        * *max_iter* (``float``) --
            Maximum number of meshing iterations. (default==50)
        * *seed* (``float`` or ``int``) --
            Pseudo-random seed to initialize meshing points. (default==0)
        * *pfix* (`array-like`) --
            An array of points to constrain in the mesh. (default==None)
        * *min_edge_length* (``float``) --
            The minimum element size in the domain. REQUIRED IF NOT USING :class:`edge_length`
        * *plot* (``int``) --
            The mesh is visualized every `plot` meshing iterations.
        * *pseudo_dt* (``float``) --
            The pseudo time step for the meshing algorithm. (default==0.2)
        * *stereo* (``bool``) --
            To mesh the whole world (default==False)

    Returns
    -------
    points: array-like
        vertex coordinates of mesh
    t: array-like
        mesh connectivity table.

    """
    _DIM = 2
    opts = {
        "max_iter": 100,           # OM2D itmax (meshgen.m:421)
        "seed": 0,
        "pfix": None,
        "egfix": None,
        "points": None,
        "min_edge_length": None,
        "plot": 999999,
        "lock_boundary": False,
        "pseudo_dt": 0.1,          # OM2D deltat (meshgen.m:654)
        "stereo": False,
        # OM2D meshgen.build parity:
        # the .m force law IS Bossen-Heckbert (meshgen.m:1003)
        "force_function": "bossen_heckbert",
        "improve": True,           # in-loop add/delete improvement
        "improve_every": 10,       # OM2D imp cadence
        "qual_tol": 0.01,          # OM2D qual_tol stagnation gate
        "exit_quality": 0.30,      # OM2D EXIT_QUALITY termination
        # plateau cutoff is an OWNER-REQUESTED option with no OM2D
        # counterpart; disabled by default for parity (OM2D only
        # exits on quality>0.30 or itmax)
        "max_stalls": None,
        "cleanup": "default",
        "cpp": True,               # OM2D proj parity (m_map 'trans' analog)
        "heal_fixed_edges_every": 2,  # OM2D delImp cadence
        "rewind_threshold": 0.10,  # abort improvement on >10% loss
    }
    opts.update(kwargs)
    _parse_kwargs(kwargs)

    fd, bbox = _unpack_domain(domain, opts)
    fh, min_edge_length = _unpack_sizing(edge_length, opts)

    _check_bbox(bbox)
    bbox = np.array(bbox).reshape(-1, 2)

    assert np.all(np.asarray(min_edge_length) > 0), (
        "`min_edge_length` must be > 0"
    )

    assert opts["max_iter"] > 0, "`max_iter` must be > 0"
    max_iter = opts["max_iter"]

    np.random.seed(opts["seed"])

    L0mult = 1 + 0.4 / 2 ** (_DIM - 1)
    if opts["pfix"] is not None and len(opts["pfix"]) > 0:
        # meshgen.m:762-763: Fscale = 1.1 when fixed points exist
        L0mult = 1.1
    delta_t = opts["pseudo_dt"]
    # meshgen.m:652: geps = 1e-12*min(h0)/Re — effectively
    # zero; keep only strictly-interior centroids
    geps = 1e-12 * np.amin(min_edge_length)
    deps = np.sqrt(np.finfo(np.double).eps)  # * np.amin(min_edge_length)

    pfix, _nfix = _unpack_pfix(_DIM, opts)

    # OM2D meshes in the m_map projected plane (meshgen proj=
    # 'trans' = Transverse Mercator centred on the domain, with an
    # m_ll2xy/m_xy2ll sandwich). Exact equivalent via pyproj tmerc.
    # Gated off near the equator / for abstract unit domains where
    # the projection is a no-op anyway.
    _unproject = None
    if (
        opts.get("cpp", True)
        and not opts["stereo"]
        and abs(bbox[1]).max() <= 90.0
        and abs(bbox[0]).max() <= 360.0
    ):
        _lat0 = 0.5 * (bbox[1][0] + bbox[1][1])
        _c0 = float(np.cos(np.deg2rad(_lat0)))
        if 0.05 < _c0 <= 0.999:
            from pyproj import Transformer

            _lon0 = 0.5 * (bbox[0][0] + bbox[0][1])
            _tr = Transformer.from_crs(
                "EPSG:4326",
                f"+proj=tmerc +lon_0={_lon0} +lat_0={_lat0} "
                "+ellps=WGS84 +units=m",
                always_xy=True,
            )
            # degree->metre factor consistent with finalize's
            # _deg_factor, so fd/fh keep their length scale and
            # min_edge_length/geps need no rescaling
            _M = 111e3

            def _to_proj(q):
                x, y = _tr.transform(q[:, 0], q[:, 1])
                return np.column_stack([x, y]) / _M

            _wrap360 = bbox[0][1] > 180.0

            def _from_proj(q):
                x, y = _tr.transform(
                    q[:, 0] * _M, q[:, 1] * _M,
                    direction="INVERSE",
                )
                if _wrap360:
                    # pyproj normalises lon to [-180,180); a 0-360
                    # domain (Example_8 dateline boubox) needs its
                    # own frame back or fd rejects the eastern half
                    x = np.where(x < bbox[0][0] - 1e-9, x + 360.0, x)
                return np.column_stack([x, y])

            _unproject = _from_proj
            logger.info(
                "tmerc frame: lon_0=%.4f lat_0=%.2f (m_map "
                "'Transverse Mercator' analog)", _lon0, _lat0,
            )
            _fd_r, _fh_r = fd, fh

            def fd(q, *a, _f=_fd_r, **k):
                return _f(_from_proj(np.asarray(q, float)), *a, **k)

            def fh(q, _f=_fh_r):
                return _f(_from_proj(np.asarray(q, float)))

            _corners = np.array(
                [
                    [bbox[0][0], bbox[1][0]],
                    [bbox[0][0], bbox[1][1]],
                    [bbox[0][1], bbox[1][0]],
                    [bbox[0][1], bbox[1][1]],
                    [_lon0, bbox[1][0]],
                    [_lon0, bbox[1][1]],
                ]
            )
            _pc = _to_proj(_corners)
            bbox = np.array(
                [
                    [_pc[:, 0].min(), _pc[:, 0].max()],
                    [_pc[:, 1].min(), _pc[:, 1].max()],
                ]
            )
            if len(pfix):
                pfix = _to_proj(np.asarray(pfix, float))
            if opts["points"] is not None:
                opts["points"] = _to_proj(
                    np.asarray(opts["points"], dtype=float)
                )

    # egfix: (M, 2) indices into pfix. Constrained edges are FORCED
    # into every retriangulation via the CGAL constrained Delaunay
    # binding — the OceanMesh2D high-fidelity capability (MATLAB
    # delaunayTriangulation constraints).
    egfix = opts.get("egfix", None)
    eg_segs = None
    if egfix is not None and len(egfix) > 0:
        egfix = np.asarray(egfix, dtype=int)
        if egfix.max() >= len(pfix):
            raise ValueError("egfix indices must reference pfix rows")
        eg_segs = np.hstack(
            [pfix[egfix[:, 0]], pfix[egfix[:, 1]]]
        ).ravel().tolist()
        logger.info(f"Constraining {len(egfix)} fixed edges (egfix)")
    lock_boundary = opts["lock_boundary"]

    if opts["points"] is None:
        p = _generate_initial_points(
            min_edge_length,
            geps,
            bbox,
            fh,
            fd,
            pfix,
            opts["stereo"],
        )
        # meshgen.m:740-747: also seed the domain-outline ring
        # points (rejection-thinned like the lattice) so frame-edge
        # coverage is stable
        _ring = getattr(domain, "boubox_ring", None)
        if _ring is not None and len(_ring):
            ring = np.asarray(_ring, dtype=float)
            try:
                ring = _to_proj(ring)
            except NameError:
                pass
            ring = ring[fd(ring) < geps]
            if len(ring):
                r0 = fh(ring)
                h0_ = float(np.amin(min_edge_length))
                keep = np.random.rand(len(ring)) < h0_**2 / r0**2
                p = np.vstack((p, ring[keep]))
                logger.info(
                    f"seeded {int(keep.sum())} domain-outline points"
                )
    else:
        # opts["points"] are projected in the tmerc block above
        # (line ~1000); do NOT project twice
        p = np.asarray(opts["points"], dtype=float)

    N = p.shape[0]

    assert N > 0, "No vertices to mesh with!"

    logger.info(
        f"Commencing mesh generation with {N} vertices will perform {max_iter} iterations."
    )
    qual_hist = []
    stall_count = 0
    heal_consecutive = 0
    p_before_improve = None
    qual_before_improve = 0.0
    pold = None
    ttol = 0.1  # meshgen.m:653 (movement-gated retriangulation)
    h0_gate = float(np.amin(min_edge_length))
    t = None
    for count in range(max_iter):
        start = time.time()

        # movement-gated retriangulation (meshgen.m:791-798): only
        # rebuild the topology when some point moved > ttol*h0;
        # each rebuild first dedups points and drops vertices that
        # ended up unused/outside (fixmesh([pfix; p]))
        if pold is None or t is None:
            move = np.inf
        else:
            n_common = min(len(p), len(pold))
            move = float(
                np.max(
                    np.sqrt(
                        ((p[:n_common] - pold[:n_common]) ** 2).sum(1)
                    )
                )
            ) / h0_gate
        if move > ttol:
            if t is not None:
                # fixmesh([pfix; p]) (meshgen.m:793-795 +
                # utilities/fixmesh.m:7-16): prepend pfix, then
                # merge ALL near-coincident points on the ptol grid
                # (ptol = 1024*eps relative to the mesh extent).
                # Without this, improvement splits / projection
                # pile-ups leave duplicate points => zero-area
                # slivers that never heal (min quality ~0).
                if len(pfix) > 0:
                    p = np.vstack((pfix, p))
                ptol = 1024 * np.finfo(float).eps * float(
                    np.max(np.ptp(p, axis=0))
                )
                _, ix = np.unique(
                    np.round(p / ptol) * ptol, axis=0,
                    return_index=True,
                )
                # keep first occurrences in original order so pfix
                # rows survive the merge
                p = p[np.sort(ix)]
            pold = p.copy()

            # (Re)-triangulation by the Delaunay algorithm
            if eg_segs is not None:
                dt = CDT()
                dt.insert(p.ravel().tolist())
                dt.insert_constraints(eg_segs)
            else:
                dt = DT()
                dt.insert(p.ravel().tolist())

            # Get the current topology of the triangulation
            p, t = _get_topology(dt)

            # Remove points outside the domain, then prune vertices
            # no interior triangle references (fixmesh semantics)
            t = _remove_triangles_outside(p, t, fd, geps)
            p, t, _ = fix_mesh(p, t, dim=_DIM, delete_unused=True)

            # restore the OM2D contract p(1:nfix,:) == pfix
            # (meshgen.m keeps pfix as the permanent first rows;
            # CGAL does not preserve insertion order, and the old
            # per-iteration closest-node snapping made the fixed
            # set drift — heal_fixed_edges then over-fired and its
            # `continue` starved the improvement cycle)
            if len(pfix) > 0:
                from scipy.spatial import cKDTree

                d_, ix_ = cKDTree(p).query(pfix)
                perm = np.concatenate(
                    [ix_, np.setdiff1d(np.arange(len(p)), ix_)]
                )
                inv = np.empty(len(p), dtype=int)
                inv[perm] = np.arange(len(p))
                p = p[perm]
                t = inv[t]
                p[:len(pfix)] = pfix

        fixed_indices = []
        if lock_boundary:
            _bedges, _ = _external_topology(p, t)
            fixed_indices = np.unique(
                np.asarray(_bedges, dtype=int).reshape(-1)
            ).tolist()

        if len(pfix) > 0:
            p[:len(pfix)] = pfix
            fixed_indices.extend(range(len(pfix)))

        # --- OM2D meshgen.build parity block (port plan P1) ---------
        q = _al_quality(p, t)
        qual_hist.append(
            (float(np.mean(q)),
             float(np.mean(q) - 3.0 * np.std(q)),
             float(np.min(q)))
        )
        logger.info(
            "iter %3d: NP=%d qual mean=%.4f p3sig=%.4f min=%.4f",
            count + 1, len(p), *qual_hist[-1],
        )
        _cwin = os.environ.get("OM_TRACE_CORNER")
        if _cwin:
            _x0, _x1, _y0, _y1 = map(float, _cwin.split(","))
            _pl = _unproject(p) if _unproject is not None else p
            _in = ((_pl[:, 0] > _x0) & (_pl[:, 0] < _x1)
                   & (_pl[:, 1] > _y0) & (_pl[:, 1] < _y1))
            logger.info("[corner] it=%d n=%d", count + 1,
                        int(_in.sum()))
        if (os.environ.get("OM_TRACE_SCALE") == "1"
                and (count + 1) % 10 == 0):
            _be, _ = _external_topology(p, t)
            _bi = np.unique(np.asarray(_be, dtype=int).reshape(-1))
            _bd = np.abs(fd(p[_bi]))
            logger.info(
                "[traj] it=%d standoff mean=%.3e p90=%.3e nb=%d",
                count + 1, float(_bd.mean()),
                float(np.percentile(_bd, 90)), len(_bi),
            )

        if (eg_segs is not None
                and opts["heal_fixed_edges_every"] > 0
                and (count + 1) % opts["heal_fixed_edges_every"] == 0):
            # heal_fixed_edges (meshgen.m:1153-1168): thin (<0.25)
            # triangles CONTAINING a constrained edge lose their
            # free (>nfix) vertices. pfix rows are stable now, so
            # egfix pairs index p directly.
            nfix_ = len(pfix)
            eg_set = {(int(a), int(b)) if a < b else (int(b), int(a))
                      for a, b in egfix}
            thin = np.where(q < 0.25)[0]
            kill = set()
            for e in thin:
                tri = sorted(int(v) for v in t[e])
                prs = [(tri[0], tri[1]), (tri[0], tri[2]),
                       (tri[1], tri[2])]
                if any(pr in eg_set for pr in prs):
                    kill.update(v for v in tri if v >= nfix_)
            if kill and heal_consecutive < 4:
                # delIT escape valve (meshgen.m:867-878): after 4
                # consecutive heals give up and let the iteration
                # (and the improvement cycle) proceed — without it
                # the every-2nd-iteration `continue` starves
                # improvement forever on crowded constrained meshes
                heal_consecutive += 1
                logger.info(
                    f"heal_fixed_edges: removing {len(kill)} vertices"
                )
                keep = np.setdiff1d(np.arange(len(p)),
                                    np.fromiter(kill, int))
                p = p[keep]
                pold = None
                continue
            heal_consecutive = 0

        at_checkpoint = (opts["improve_every"] > 0
                         and (count + 1) % opts["improve_every"] == 0)
        if at_checkpoint and qual_hist[-1][2] > opts["exit_quality"]:
            p, t, _ = fix_mesh(p, t, dim=_DIM, delete_unused=True)
            p, t = _maybe_om2d_clean(p, t, opts, pfix, _unproject)
            logger.info(
                "Termination: minimum quality %.3f > %.2f",
                qual_hist[-1][2], opts["exit_quality"],
            )
            return p, t

        # OM2D rewind of a failed improvement (meshgen.m:818-830):
        # at mod(it, imp+1) restore the pre-improvement points when
        # mean quality dropped > 0.10 or node count dropped > 10%
        if (p_before_improve is not None
                and (count + 1) % (opts["improve_every"] + 1) == 0):
            # meshgen.m:813-822: the quality test is the ONE-STEP
            # mean drop qual(it)-qual(it-1) < -0.10
            mean_drop = (qual_hist[-2][0] - qual_hist[-1][0]
                         if len(qual_hist) >= 2 else 0.0)
            count_drop = (len(p_before_improve) - len(p)) / max(
                len(p_before_improve), 1
            )
            _pb = p_before_improve
            p_before_improve = None
            if mean_drop > 0.10 or count_drop > 0.10:
                logger.info(
                    "improvement rewound (mean drop %.3f, count "
                    "drop %.1f%%)", mean_drop, 100 * count_drop,
                )
                p = _pb
                pold = None
                # meshgen.m:819-821: `it=it+1; continue` — the .m
                # SKIPS the rest of this iteration so the next pass
                # retriangulates BEFORE any force step. Falling
                # through computed forces on the restored (longer)
                # point array with the OLD post-improvement
                # triangulation's indices — vertex mix-up that
                # shredded the mesh (Example_6: NP 25.7k -> 18.3k,
                # qual 0.877 -> 0.633 right after a rewind)
                continue

        if opts["improve"] and at_checkpoint:
            # meshgen.m:883,952-953: gate on the 3-sigma-LOW metric
            # vs qual(max(1, it-imp)) — active from the FIRST
            # checkpoint — and only act while quality is IMPROVING
            prev_l3 = qual_hist[
                max(0, len(qual_hist) - 1 - opts["improve_every"])
            ][1]
            qual_diff = qual_hist[-1][1] - prev_l3
            if abs(qual_diff) < (
                    opts["improve_every"] * opts["qual_tol"]):
                stall_count += 1
                _ms = opts.get("max_stalls")
                if _ms is not None and stall_count >= int(_ms):
                    # owner-requested plateau cutoff (opt-in; no
                    # OM2D counterpart)
                    p, t, _ = fix_mesh(p, t, dim=_DIM,
                                       delete_unused=True)
                    p, t = _maybe_om2d_clean(p, t, opts, pfix, _unproject)
                    logger.info(
                        "Termination: quality plateau after %d "
                        "improvement cycles (mean %.4f, min %.4f) "
                        "at iter %d — cutting off.",
                        stall_count, qual_hist[-1][0],
                        qual_hist[-1][2], count + 1,
                    )
                    return p, t
                if qual_diff > 0:
                    p_before_improve = p.copy()
                    qual_before_improve = qual_hist[-1][0]
                    p = _improve_points(
                        p, t, fh, fd, geps, pfix, lock_boundary,
                        opts["rewind_threshold"],
                        stereo=opts["stereo"],
                    )
                    pold = None
                    continue
            else:
                stall_count = 0
        # -------------------------------------------------------------

        # Number of iterations reached, stop.
        if count == (max_iter - 1):
            p, t, _ = fix_mesh(p, t, dim=_DIM, delete_unused=True)
            p, t = _maybe_om2d_clean(p, t, opts, pfix, _unproject)
            logger.info("Termination reached...maximum number of iterations.")
            return p, t

        # Compute the forces on the bars
        Ftot = _compute_forces(p, t, fh, min_edge_length, L0mult, opts)

        # Force = 0 at fixed points
        if fixed_indices:
            fixed_indices = np.unique(np.asarray(fixed_indices, dtype=int))
            Ftot[fixed_indices] = 0

        # Update positions
        p += delta_t * Ftot

        # Bring outside points back to the boundary
        p = _project_points_back(p, fd, deps)

        # Show the user some progress so they know something is happening
        maxdp = delta_t * np.sqrt((Ftot**2).sum(1)).max()

        logger.info(
            "Iteration #%d, max movement %.3f, vertices %s, elements %s",
            count + 1,
            float(maxdp),
            f"{len(p):,}",
            f"{len(t):,}",
        )

        end = time.time()
        logger.info("Elapsed wall-clock time %.3f seconds", end - start)

    # Loop exhausted via a `continue` path (healing/improvement on
    # the final iteration): finalize from the last point set.
    if eg_segs is not None:
        dt = CDT()
        dt.insert(p.ravel().tolist())
        dt.insert_constraints(eg_segs)
    else:
        dt = DT()
        dt.insert(p.ravel().tolist())
    p, t = _get_topology(dt)
    t = _remove_triangles_outside(p, t, fd, geps)
    p, t, _ = fix_mesh(p, t, dim=_DIM, delete_unused=True)
    p, t = _maybe_om2d_clean(p, t, opts, pfix, _unproject)
    logger.info("Termination reached...maximum number of iterations.")
    return p, t


def _unpack_sizing(edge_length, opts):
    if isinstance(edge_length, Grid):
        fh = edge_length.eval
        min_edge_length = edge_length.hmin
        # Defensive: if hmin is invalid, recompute from grid values
        if (
            min_edge_length is None
            or not np.isfinite(min_edge_length)
            or min_edge_length <= 0
        ):
            vals = edge_length.values
            if np.ma.isMaskedArray(vals):
                vals = np.ma.filled(vals, np.nan)
            vals = np.asarray(vals)
            pos = vals[np.isfinite(vals) & (vals > 0)]
            if pos.size > 0:
                min_edge_length = float(np.nanmin(pos))
                edge_length.hmin = min_edge_length
                logger.warning(
                    f"Edge length grid had invalid hmin; recomputed fallback min_edge_length={min_edge_length:.3f}"
                )
            else:
                raise ValueError(
                    "Edge length grid contains no positive values to determine a minimum edge length."
                )
    elif callable(edge_length):
        fh = edge_length
        min_edge_length = opts["min_edge_length"]
    else:
        raise ValueError(
            "`edge_length` must either be a function or a `edge_length` object"
        )
    return fh, min_edge_length


def _unpack_domain(domain, opts):
    if isinstance(domain, Domain):
        bbox = domain.bbox
        fd = domain.eval
    elif callable(domain):
        bbox = opts["bbox"]
        fd = domain
    else:
        raise ValueError(
            "`domain` must be a function or a :class:`signed_distance_function object"
        )
    return fd, bbox


def _get_bars(t):
    """Describe each bar by a unique pair of nodes"""
    bars = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    return unique_edges(bars)


# Persson-Strang
def _al_quality(p, t):
    """Area-length element quality, equilateral = 1 (OM2D
    gettrimeshquan's qm): 4*sqrt(3)*A / (l1^2 + l2^2 + l3^2). The
    in-loop OM2D thresholds (EXIT_QUALITY=0.30, heal 0.25) are
    defined on THIS scale (fork's simp_qual uses another scale)."""
    tri = p[t]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 1]
    e3 = tri[:, 0] - tri[:, 2]
    area = 0.5 * np.abs(e1[:, 0] * (-e3[:, 1]) - e1[:, 1] * (-e3[:, 0]))
    den = (e1**2).sum(1) + (e2**2).sum(1) + (e3**2).sum(1)
    den[den == 0] = np.finfo(float).eps
    return 4.0 * np.sqrt(3.0) * area / den


def _improve_points(p, t, fh, fd, geps, pfix, lock_boundary,
                    rewind_threshold, L0mult=1.2, stereo=False):
    """Exact port of the OM2D improvement block
    (meshgen.m:948-1000):
    - nn: nodes touching <= 4 elements, excluding boundary nodes
      and pfix (get_small_connectivity, meshgen.m:1142-1151)
    - nn1: BOTH endpoints of every bar with LN < 0.5, excluding
      ONLY pfix (meshgen.m:963 — boundary nodes are NOT protected)
    - splits: one MIDPOINT per bar with floor(LN) >= 2
      (meshgen.m:971-987, jj=2:2)
    - LN = L / L0 with L0 = hbars * Fscale * median(L)/
      median(hbars) — the FORCE-STEP normalisation
      (meshgen.m:944), not raw fh
    - no candidate filtering and no internal rewind: the loop-level
    mod(it, imp+1) rewind (meshgen.m:818-830) is the only guard."""
    n0 = len(p)
    protected = set()
    protected.update(range(len(pfix)))
    # boundary_edges already hold vertex INDICES — recovering them
    # via per-point _closest_node scans was O(B*N) (67+ min at
    # NP=4.8M on Example_4) and wrong under exact-duplicate points
    _bedges, _ = _external_topology(p, t)
    bnd_protected = set(protected)
    bnd_protected.update(
        np.unique(np.asarray(_bedges, dtype=int).reshape(-1)).tolist()
    )

    conn = np.bincount(t.ravel(), minlength=len(p))
    low = {int(v) for v in np.where(conn <= 4)[0]
           if conn[v] > 0 and int(v) not in bnd_protected}

    bars = _get_bars(t)
    barvec = p[bars[:, 0]] - p[bars[:, 1]]
    L = np.sqrt((barvec**2).sum(1))
    mid = p[bars].sum(1) / 2
    if stereo:
        # points live in the stereographic plane: evaluate fh in
        # lat/lon and convert to plane units, EXACTLY as the force
        # step does — evaluating fh on raw stereo coordinates made
        # LN uniform-in-plane and the split rule inflated the
        # global mesh 2.7M -> 6M vertices (Example_7)
        _lon, _lat = to_lat_lon(mid[:, 0], mid[:, 1])
        hbars = np.asarray(
            fh(np.column_stack([_lon, _lat])), dtype=float
        ) * _stereo_distortion_dist(_lat)
    else:
        hbars = np.asarray(fh(mid), dtype=float)
    valid = np.isfinite(hbars) & (hbars > 0)
    if not valid.all():
        hbars[~valid] = np.nanmedian(hbars[valid])
    L0 = hbars * L0mult * (np.nanmedian(L) / np.nanmedian(hbars))
    LN = L / L0
    if os.environ.get("OM_TRACE_IMPROVE") == "1":
        _sc = float(np.nanmedian(L) / np.nanmedian(hbars))
        _del = LN < 0.5
        _spl = np.floor(LN) >= 2
        logger.info(
            "[improve-trace] scale=%.3f  del n=%d L_m p50=%.0f "
            "hbar_m p50=%.0f  spl n=%d L_m p50=%.0f hbar_m p50=%.0f",
            _sc, int(_del.sum()),
            float(np.nanmedian(L[_del]) * 111e3) if _del.any() else -1,
            float(np.nanmedian(hbars[_del]) * 111e3) if _del.any() else -1,
            int(_spl.sum()),
            float(np.nanmedian(L[_spl]) * 111e3) if _spl.any() else -1,
            float(np.nanmedian(hbars[_spl]) * 111e3) if _spl.any() else -1,
        )

    n_low = len(low)
    for a, b in bars[LN < 0.5]:
        for v in (int(a), int(b)):
            if v not in protected:
                low.add(v)
    n_short = len(low) - n_low

    new_pts = []
    for (a, b) in bars[np.floor(LN) >= 2]:
        new_pts.append(0.5 * (p[int(a)] + p[int(b)]))

    keep = np.setdiff1d(np.arange(n0), np.fromiter(low, int)
                        if low else np.array([], dtype=int))
    p_new = p[keep]
    if new_pts:
        p_new = np.vstack([p_new, np.asarray(new_pts)])
    logger.info(
        f"improvement: -{n_low} small-connectivity, "
        f"-{n_short} too-close, +{len(new_pts)} splits"
    )
    return p_new


def _compute_forces(p, t, fh, min_edge_length, L0mult, opts):
    """Compute the forces on each edge based on the sizing function"""
    N = p.shape[0]
    bars = _get_bars(t)
    barvec = p[bars[:, 0]] - p[bars[:, 1]]  # List of bar vectors
    L = np.sqrt((barvec**2).sum(1))  # L = Bar lengths
    L[L == 0] = np.finfo(float).eps
    if opts["stereo"]:
        # For global+regional multiscale meshes, this branch handles the global stereo case.
        # Regional sizing functions have been wrapped or transformed earlier so fh(p2)
        # evaluates correctly on lat/lon even though points are maintained in stereo space.
        p1 = p[bars].sum(1) / 2
        x, y = to_lat_lon(p1[:, 0], p1[:, 1])
        p2 = np.asarray([x, y]).T
        hbars = fh(p2) * _stereo_distortion_dist(y)
    else:
        hbars = fh(p[bars].sum(1) / 2)
    # Guard against non-finite or non-positive sizing values that can poison forces
    hbars = np.asarray(hbars, dtype=float)
    valid = np.isfinite(hbars) & (hbars > 0)
    if not np.any(valid):
        raise ValueError(
            "Sizing function returned no positive finite values inside domain."
        )
    if not np.all(valid):
        repl = np.nanmedian(hbars[valid])
        hbars = np.where(valid, hbars, repl)
    L0 = hbars * L0mult * (np.nanmedian(L) / np.nanmedian(hbars))
    if os.environ.get("OM_TRACE_SCALE") == "1":
        logger.info(
            "[traj] medL=%.6e medH=%.6e ratio=%.6f nbars=%d",
            float(np.nanmedian(L)), float(np.nanmedian(hbars)),
            float(np.nanmedian(L) / np.nanmedian(hbars)), len(L),
        )
    if opts.get("force_function", "bossen_heckbert") == "bossen_heckbert":
        # meshgen.m:1001-1005 verbatim: LN = L./L0;
        # F = (1-LN.^4).*exp(-LN.^4)./LN; F(isinf(F))=0;
        # Fvec = F*[1,1].*barvec  (barvec RAW, not normalized)
        LN = L / L0
        with np.errstate(divide="ignore", invalid="ignore"):
            F = (1.0 - LN**4) * np.exp(-(LN**4)) / LN
        F[~np.isfinite(F)] = 0.0
        Fvec = F[:, None] * barvec
    else:
        F = L0 - L
        F[F < 0] = 0  # Bar forces (scalars)
        Fvec = (
            F[:, None] / L[:, None].dot(np.ones((1, 2))) * barvec
        )  # Bar forces (x,y components)
    Ftot = _dense(
        bars[:, [0] * 2 + [1] * 2],
        np.repeat([list(range(2)) * 2], len(F), axis=0),
        np.hstack((Fvec, -Fvec)),
        shape=(N, 2),
    )
    return Ftot


# Bossen-Heckbert
# def _compute_forces(p, t, fh, min_edge_length, L0mult):
#    """Compute the forces on each edge based on the sizing function"""
#    N = p.shape[0]
#    bars = _get_bars(t)
#    barvec = p[bars[:, 0]] - p[bars[:, 1]]  # List of bar vectors
#    L = np.sqrt((barvec ** 2).sum(1))  # L = Bar lengths
#    L[L == 0] = np.finfo(float).eps
#    hbars = fh(p[bars].sum(1) / 2)
#    L0 = hbars * L0mult * (np.nanmedian(L) / np.nanmedian(hbars))
#    LN = L / L0
#    F = (1 - LN ** 4) * np.exp(-(LN ** 4)) / LN
#    Fvec = (
#        F[:, None] / LN[:, None].dot(np.ones((1, 2))) * barvec
#    )  # Bar forces (x,y components)
#    Ftot = _dense(
#        bars[:, [0] * 2 + [1] * 2],
#        np.repeat([list(range(2)) * 2], len(F), axis=0),
#        np.hstack((Fvec, -Fvec)),
#        shape=(N, 2),
#    )
#    return Ftot


def _dense(Ix, J, S, shape=None, dtype=None):
    """
    Similar to MATLAB's SPARSE(I, J, S, ...), but instead returning a
    dense array.
    """

    # Advanced usage: allow J and S to be scalars.
    if np.isscalar(J):
        x = J
        J = np.empty(Ix.shape, dtype=int)
        J.fill(x)
    if np.isscalar(S):
        x = S
        S = np.empty(Ix.shape)
        S.fill(x)

    # Turn these into 1-d arrays for processing.
    S = S.flat
    II = Ix.flat
    J = J.flat
    return spsparse.coo_matrix((S, (II, J)), shape, dtype).toarray()



def _maybe_om2d_clean(p, t, opts, pfix, unproject=None):
    """OM2D meshgen.build parity: msh.clean('default') at every
    build exit (meshgen.m:1063-1068) unless cleanup='none'. Like
    msh.clean's m_proj sandwich, the clean runs in the projected
    (tmerc) frame and the output is unprojected to lon/lat."""
    if opts.get("cleanup", "default") == "default":
        from .clean import om2d_default_clean

        keep = pfix if pfix is not None and len(pfix) else None
        p, t = om2d_default_clean(p, t, pfix=keep)
    if unproject is not None:
        p = unproject(p)
    return p, t


def _remove_triangles_outside(p, t, fd, geps):
    """Remove vertices outside the domain"""
    pmid = p[t].sum(1) / 3  # Compute centroids
    return t[fd(pmid) < -geps]  # Keep interior triangles


def _project_points_back(p, fd, deps):
    """Project points outsidt the domain back within"""
    d = fd(p)
    ix = d > 0  # Find points outside (d>0)
    if ix.any():

        def _deps_vec(i):
            a = [0] * 2
            a[i] = deps
            return a

        try:
            dgrads = [
                (fd(p[ix] + _deps_vec(i)) - d[ix]) / deps for i in range(2)
            ]  # old method
        except ValueError:  # an error is thrown if all points in fd are outside
            # bbox domain, so instead calulate all fd and then
            # take the solely ones outside domain
            dgrads = [(fd(p + _deps_vec(i)) - d) / deps for i in range(2)]
            dgrads = list(np.array(dgrads)[:, ix])
        dgrad2 = sum(dgrad**2 for dgrad in dgrads)
        dgrad2 = np.where(dgrad2 < deps, deps, dgrad2)
        p[ix] -= (d[ix] * np.vstack(dgrads) / dgrad2).T  # Project
    return p


def _stereo_distortion(lat):
    # we use here Stereographic projection of the sphere
    # from the north pole onto the plane
    # https://en.wikipedia.org/wiki/Stereographic_projection
    lat0 = 90
    ll = lat + lat0
    lrad = ll / 180 * np.pi
    res = 2 / (1 + np.sin(lrad))
    return res


def _stereo_distortion_dist(lat):
    lrad = np.radians(lat)
    # Calculate the scale factor for the stereographic projection
    res = 2 / (1 + np.sin(lrad)) / 180 * np.pi
    return res


def _generate_initial_points(min_edge_length, geps, bbox, fh, fd, pfix, stereo=False):
    """Create initial distribution in bounding box (equilateral triangles)"""
    if stereo:
        bbox = np.array([[-180, 180], [-89, 89]])
    if stereo:
        p = np.mgrid[
            tuple(
                slice(min, max + min_edge_length, min_edge_length)
                for min, max in bbox
            )
        ].astype(float)
    else:
        # OM2D equilateral seeding (meshgen.m:680-712): rows at
        # min_edge spacing, columns at 2/sqrt(3)*min_edge, odd rows
        # offset by half a column — 13% fewer seeds than a square
        # lattice and the classic DistMesh starting layout
        dxs = 2.0 / np.sqrt(3.0) * min_edge_length
        ys = np.arange(
            bbox[1][0], bbox[1][1] + min_edge_length, min_edge_length
        )
        rows = []
        for i, y in enumerate(ys):
            x0 = bbox[0][0] + (0.5 * dxs if i % 2 else 0.0)
            xs = np.arange(x0, bbox[0][1] + dxs, dxs)
            rows.append(
                np.column_stack([xs, np.full(len(xs), y)])
            )
        p = np.vstack(rows)
    if stereo:
        # For global meshes (including mixed global+regional) we generate points in lat/lon,
        # then project to stereo. The sizing function fh has already been wrapped (if needed)
        # to internally transform coordinates back to lat/lon for regional grids.
        # for global meshes in stereographic projections,
        # we need to reproject the points from lon/lat to stereo projection
        # then, we need to rectify their coordinates to lat/lon for the sizing function
        p0 = p.reshape(2, -1).T
        x, y = to_stereo(p0[:, 0], p0[:, 1])
        p = np.asarray([x, y]).T
        _ll = to_lat_lon(p[:, 0], p[:, 1])
        r0 = (fh(np.column_stack([_ll[0], _ll[1]]))
              * _stereo_distortion(p0[:, 1]))
    else:
        r0 = fh(p)
    # meshgen.m:668 anchors the rejection at h0 itself
    # (max_r0 = 1/h0_l^2): accept with prob (h0/fh)^2. Anchoring at
    # the lattice-sampled minimum over-seeds by (min_sampled/h0)^2
    # (2.8x on the JBAY weir case, where the h0-sized cells are a
    # tiny sliver of the domain).
    r0m = float(np.amin(min_edge_length))
    p = p[np.random.rand(p.shape[0]) < r0m**2 / r0**2]
    p = p[fd(p) < geps]  # Keep only d<0 points
    return np.vstack(
        (
            pfix,
            p,
        )
    )


def _dist(p1, p2):
    """Euclidean distance between two sets of points"""
    return np.sqrt(((p1 - p2) ** 2).sum(1))


def _unpack_pfix(dim, opts):
    """Unpack fixed points"""
    pfix = np.empty((0, dim))
    nfix = 0
    if opts["pfix"] is not None:
        pfix = np.array(opts["pfix"], dtype="d")
        nfix = len(pfix)
        logger.info(f"Constraining {nfix} fixed points..")
    return pfix, nfix


def _get_topology(dt):
    """Get points and entities from :clas:`CGAL:DelaunayTriangulation2/3` object"""
    return dt.get_finite_vertices(), dt.get_finite_cells()


def _closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum("ij,ij->i", deltas, deltas)
    return np.argmin(dist_2)
