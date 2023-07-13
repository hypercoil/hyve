# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unified plotter
~~~~~~~~~~~~~~~
Unified plotting function for surface, volume, and network data.
"""
import dataclasses
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

from .surf import (
    CortexTriSurface,
)
from .util import robust_clim


@dataclasses.dataclass
class HemisphereParameters:
    left: Mapping[str, Any]
    right: Mapping[str, Any]

    def get(self, hemi, param):
        return self.__getattribute__(hemi)[param]


def _get_hemisphere_parameters(
    *,
    surf_scalars_cmap: Any,
    surf_scalars_clim: Any,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    left = {}
    right = {}

    def assign_tuple(arg, name):
        left[name], right[name] = arg

    def assign_scalar(arg, name):
        left[name] = right[name] = arg

    def conditional_assign(condition, arg, name):
        if condition(arg):
            assign_tuple(arg, name)
        else:
            assign_scalar(arg, name)

    conditional_assign(
        lambda x: len(x) == 2,
        surf_scalars_cmap,
        'surf_scalars_cmap',
    )
    conditional_assign(
        lambda x: len(x) == 2 and isinstance(x[0], (tuple, list)),
        surf_scalars_clim,
        'surf_scalars_clim',
    )
    return HemisphereParameters(left, right)


def _get_color(color, cmap, clim):
    if (
        isinstance(color, str)
        or isinstance(color, tuple)
        or isinstance(color, list)
    ):
        return color
    else:
        try:
            cmap = get_cmap(cmap)
        except ValueError:
            cmap = cmap
        vmin, vmax = clim
        norm = Normalize(vmin=vmin, vmax=vmax)
        return cmap(norm(color))


def _map_to_attr(values, attr, attr_range):
    if attr == 'index':
        attr = np.array(values.index)
    else:
        attr = values[attr]
    if attr_range is None:
        return attr
    max_val = attr.max()
    min_val = attr.min()
    return (
        attr_range[0]
        + (attr_range[1] - attr_range[0])
        * (attr - min_val)
        / (max_val - min_val)
    )


def _map_to_radius(
    values: Sequence,
    radius: Union[float, str],
    radius_range: Tuple[float, float],
) -> Sequence:
    if isinstance(radius, float):
        return (radius,) * len(values)
    else:
        return _map_to_attr(values, radius, radius_range)


def _map_to_color(
    values: Sequence,
    color: Union[str, Sequence],
    clim: Optional[Tuple[float, float]] = None,
) -> Sequence:
    if color in values.columns or color == 'index':
        return _map_to_attr(values, color, clim)
    else:
        return (color,) * len(values)


def _map_to_opacity(
    values: Sequence,
    alpha: Union[float, str],
) -> Sequence:
    if isinstance(alpha, float):
        return (alpha,) * len(values)
    # TODO: this will fail if you want to use the index as the opacity.
    #       There's no legitimate reason you would want to do this
    #       so it's a very low priority fix.
    opa_min = max(0, values[alpha].min())
    opa_max = min(1, values[alpha].max())
    return _map_to_attr(values, alpha, (opa_min, opa_max))


def unified_plotter(
    *,
    surf: Optional['CortexTriSurface'] = None,
    surf_projection: str = 'pial',
    surf_alpha: float = 1.0,
    surf_scalars: Optional[str] = None,
    surf_scalars_boundary_color: str = 'black',
    surf_scalars_boundary_width: int = 0,
    surf_scalars_cmap: Any = (None, None),
    surf_scalars_clim: Any = 'robust',
    surf_scalars_below_color: str = 'black',
    vol_coor: Optional[np.ndarray] = None,
    vol_scalars: Optional[np.ndarray] = None,
    vol_scalars_point_size: Optional[float] = None,
    vol_voxdim: Optional[Sequence[float]] = None,
    vol_scalars_cmap: Optional[str] = None,
    vol_scalars_clim: Optional[tuple] = None,
    vol_scalars_alpha: float = 0.99,
    node_values: Optional[pd.DataFrame] = None,
    node_coor: Optional[np.ndarray] = None,
    node_parcel_scalars: Optional[str] = None,
    node_color: Optional[str] = 'black',
    node_radius: Union[float, str] = 3.0,
    node_radius_range: Tuple[float, float] = (2, 10),
    node_cmap: Any = 'viridis',
    node_clim: Tuple[float, float] = (0, 1),
    node_alpha: Union[float, str] = 1.0,
    node_lh: Optional[np.ndarray] = None,
    edge_values: Optional[pd.DataFrame] = None,
    edge_color: Optional[str] = 'edge_sgn',
    edge_radius: Union[float, str] = 'edge_val',
    edge_radius_range: Tuple[float, float] = (0.1, 1.8),
    edge_cmap: Any = 'RdYlBu',
    edge_clim: Tuple[float, float] = (0, 1),
    edge_alpha: Union[float, str] = 1.0,
    hemisphere: Optional[Literal['left', 'right']] = None,
    hemisphere_slack: Optional[Union[float, Literal['default']]] = 'default',
    off_screen: bool = True,
    copy_actors: bool = False,
    theme: Optional[Any] = None,
    views: Sequence = (),
    return_plotter: bool = False,
    return_screenshot: bool = True,
    return_html: bool = False,
) -> Optional[pv.Plotter]:
    """
    Plot a surface, volume, and/or graph in a single figure.

    This is the main plotting function for this package. It uses PyVista
    as the backend, and returns a PyVista plotter object or a specified
    output format.

    It is not recommended to use this function directly. Instead, define a
    functional pipeline using the other plotting functions in this package;
    the specified pipeline transforms ``unified_plotter`` into a more
    user-friendly interface that enables reconfiguring the acceptable input
    and output formats. For example, a pipeline can reconfigure the input
    formats to accept standard neuroimaging data types, and reconfigure the
    output formats to return a set of static images corresponding to different
    views or camera angles, or an interactive HTML visualization.

    Parameters
    ----------
    surf : cortex.CortexTriSurface (default: None)
        A surface to plot. If not specified, no surface will be plotted.
    surf_projection : str (default: 'pial')
        The projection of the surface to plot. The projection must be
        available in the surface's ``projections`` attribute. For typical
        surfaces, available projections might include ``'pial'``,
        ``'inflated'``, ``veryinflated``, ``'white'``, and ``'sphere'``.
    surf_alpha : float (default: 1.0)
        The opacity of the surface.
    surf_scalars : str (default: None)
        The name of the scalars to plot on the surface. The scalars must be
        available in the surface's ``point_data`` attribute. If not specified,
        no scalars will be plotted.
    surf_scalars_boundary_color : str (default: 'black')
        The color of the boundary between the surface and the background. Note
        that this boundary is only visible if ``surf_scalars_boundary_width``
        is greater than 0.
    surf_scalars_boundary_width : int (default: 0)
        The width of the boundary between the surface and the background. If
        set to 0, no boundary will be plotted.
    surf_scalars_cmap : str or tuple (default: (None, None))
        The colormap to use for the surface scalars. If a tuple is specified,
        the first element is the colormap to use for the left hemisphere, and
        the second element is the colormap to use for the right hemisphere.
        If a single colormap is specified, it will be used for both
        hemispheres.
    surf_scalars_clim : str or tuple (default: 'robust')
        The colormap limits to use for the surface scalars. If a tuple is
        specified, the first element is the colormap limits to use for the
        left hemisphere, and the second element is the colormap limits to use
        for the right hemisphere. If a single value is specified, it will be
        used for both hemispheres. If set to ``'robust'``, the colormap limits
        will be set to the 5th and 95th percentiles of the data.
    surf_scalars_below_color : str (default: 'black')
        The color to use for values below the colormap limits.
    vol_coor : np.ndarray (default: None)
        The coordinates of the volumetric data to plot. If not specified, no
        volumetric data will be plotted.
    vol_scalars : np.ndarray (default: None)
        The volumetric data to plot. If not specified, no volumetric data will
        be plotted.
    vol_scalars_point_size : float (default: None)
        The size of the points to plot for the volumetric data. If not
        specified, the size of the points will be automatically determined
        based on the size of the volumetric data.
    vol_voxdim : tuple (default: None)
        The dimensions of the voxels in the volumetric data.
    vol_scalars_cmap : str (default: 'viridis')
        The colormap to use for the volumetric data.
    vol_scalars_clim : tuple (default: None)
        The colormap limits to use for the volumetric data.
    vol_scalars_alpha : float (default: 1.0)
        The opacity of the volumetric data.
    node_values : pd.DataFrame (default: None)
        A table containing node-valued variables. Columns in the table can be
        used to specify attributes of plotted nodes, such as their color,
        radius, and opacity.
    node_coor : np.ndarray (default: None)
        The coordinates of the nodes to plot. If not specified, no nodes will
        be plotted. Node coordinates can also be computed from a parcellation
        by specifying ``node_parcel_scalars``.
    node_parcel_scalars : str (default: None)
        If provided, node coordinates will be computed as the centroids of
        parcels in the specified parcellation. The parcellation must be
        available in the ``point_data`` attribute of the surface. If not
        specified, node coordinates must be provided in ``node_coor`` or
        nodes will not be plotted.
    node_color : str or colour specification (default: 'black')
        The color of the nodes. If ``node_values`` is specified, this argument
        can be used to specify a column in the table to use for the node
        colors.
    node_radius : float or str (default: 3.0)
        The radius of the nodes. If ``node_values`` is specified, this
        argument can be used to specify a column in the table to use for the
        node radii.
    node_radius_range : tuple (default: (2, 10))
        The range of node radii to use. The values in ``node_radius`` will be
        linearly scaled to this range.
    node_cmap : str or matplotlib colormap (default: 'viridis')
        The colormap to use for the nodes.
    node_clim : tuple (default: (0, 1))
        The range of values to map into the dynamic range of the colormap.
    node_alpha : float or str (default: 1.0)
        The opacity of the nodes. If ``node_values`` is specified, this
        argument can be used to specify a column in the table to use for the
        node opacities.
    node_lh : np.ndarray (default: None)
        Boolean-valued array indicating which nodes belong to the left
        hemisphere.
    edge_values : pd.DataFrame (default: None)
        A table containing edge-valued variables. The table must have a
        MultiIndex with two levels, where the first level contains the
        starting node of each edge, and the second level contains the ending
        node of each edge. Additional columns can be used to specify
        attributes of plotted edges, such as their color, radius, and
        opacity.
    edge_color : str or colour specification (default: 'edge_sgn')
        The color of the edges. If ``edge_values`` is specified, this argument
        can be used to specify a column in the table to use for the edge
        colors. By default, edges are colored according to the value of the
        ``edge_sgn`` column in ``edge_values``, which is 1 for positive edges
        and -1 for negative edges when the edges are digested by the
        ``filter_adjacency_data`` function using the default settings.
    edge_radius : float or str (default: 'edge_val')
        The radius of the edges. If ``edge_values`` is specified, this
        argument can be used to specify a column in the table to use for the
        edge radii. By default, edges are sized according to the value of the
        ``edge_val`` column in ``edge_values``, which is the absolute value of
        the edge weight when the edges are digested by the
        ``filter_adjacency_data`` function using the default settings.
    edge_radius_range : tuple (default: (0.1, 1.8))
        The range of edge radii to use. The values in ``edge_radius`` will be
        linearly scaled to this range.
    edge_cmap : str or matplotlib colormap (default: 'RdYlBu')
        The colormap to use for the edges.
    edge_clim : tuple (default: None)
        The range of values to map into the dynamic range of the colormap.
    edge_alpha : float or str (default: 1.0)
        The opacity of the edges. If ``edge_values`` is specified, this
        argument can be used to specify a column in the table to use for the
        edge opacities.
    hemisphere : str (default: None)
        The hemisphere to plot. If not specified, both hemispheres will be
        plotted.
    hemisphere_slack : float, None, or ``'default'`` (default: ``'default'``)
        The amount of slack to add between the hemispheres when plotting both
        hemispheres. This argument is ignored if ``hemisphere`` is not
        specified. The slack is specified in units of hemisphere width. Thus,
        a slack of 1.0 means that the hemispheres will be plotted without any
        extra space or overlap between them. When the slack is greater than
        1.0, the hemispheres will be plotted with extra space between them.
        When the slack is less than 1.0, the hemispheres will be plotted with
        some overlap between them. If the slack is set to ``'default'``, the
        slack will be set to 1.1 for projections that have overlapping
        hemispheres and None for projections that do not have overlapping
        hemispheres.
    off_screen : bool (default: True)
        Whether to render the plot off-screen. If ``False``, a window will
        appear containing an interactive plot.
    copy_actors : bool (default: True)
        Whether to copy the actors before returning them. If ``False``, the
        actors will be modified in-place.
    theme : PyVista plotter theme (default: None)
        The PyVista plotter theme to use. If not specified, the default
        DocumentTheme will be used.
    """

    # Helper functions for graph plots
    def process_edge_values():
        start_node, end_node = tuple(zip(*edge_values.index))
        start_node, end_node = np.array(start_node), np.array(end_node)
        start = node_coor[start_node]
        end = node_coor[end_node]
        centre = (start + end) / 2
        direction = end - start
        length = np.linalg.norm(direction, axis=-1)
        direction = direction / length.reshape(-1, 1)
        return centre, direction, length

    # TODO: cortex_theme doesn't work here for some reason. If the background
    #       is transparent, all of the points are also made transparent. So
    #       we're sticking with a white background for now.
    theme = theme or pv.themes.DocumentTheme()

    hemispheres = (
        (hemisphere,) if hemisphere is not None else ('left', 'right')
    )
    hemi_params = _get_hemisphere_parameters(
        surf_scalars_cmap=surf_scalars_cmap,
        surf_scalars_clim=surf_scalars_clim,
    )
    if node_parcel_scalars is not None:
        node_coor = surf.parcel_centres_of_mass(
            node_parcel_scalars,
            surf_projection,
        )

    p = pv.Plotter(off_screen=off_screen, theme=theme)

    if hemisphere_slack == 'default':
        proj_require_slack = {'inflated', 'veryinflated', 'sphere'}
        if surf_projection in proj_require_slack:
            hemisphere_slack = 1.1
        else:
            hemisphere_slack = None
    if len(hemispheres) == 2 and hemisphere_slack is not None:
        if surf is not None:
            surf.left.project(surf_projection)
            surf.right.project(surf_projection)
            hw_left = (surf.left.bounds[1] - surf.left.bounds[0]) / 2
            hw_right = (surf.right.bounds[1] - surf.right.bounds[0]) / 2
            hemi_gap = surf.right.center[0] - surf.left.center[0]
        elif node_coor is not None and node_lh is not None:
            hw_left = (
                node_coor[node_lh, 0].max() - node_coor[node_lh, 0].min()
            ) / 2
            hw_right = (
                node_coor[~node_lh, 0].max() - node_coor[~node_lh, 0].min()
            ) / 2
            hemi_gap = (
                node_coor[~node_lh, 0].max() + node_coor[~node_lh, 0].min()
            ) / 2 - (
                node_coor[node_lh, 0].max() + node_coor[node_lh, 0].min()
            ) / 2
        elif vol_coor is not None:
            left_mask = vol_coor[:, 0] < 0
            hw_left = (
                vol_coor[left_mask, 0].max() - vol_coor[left_mask, 0].min()
            ) / 2
            hw_right = (
                vol_coor[~left_mask, 0].max() - vol_coor[~left_mask, 0].min()
            ) / 2
            hemi_gap = (
                vol_coor[~left_mask, 0].max() + vol_coor[~left_mask, 0].min()
            ) / 2 - (
                vol_coor[left_mask, 0].max() + vol_coor[left_mask, 0].min()
            ) / 2
        else:
            hw_left = hw_right = hemi_gap = 0
        min_gap = hw_left + hw_right
        target_gap = min_gap * hemisphere_slack
        displacement = (target_gap - hemi_gap) / 2
        if surf is not None:
            left = surf.left.translate((-displacement, 0, 0))
            right = surf.right.translate((displacement, 0, 0))
            surf = CortexTriSurface(left=left, right=right, mask=surf.mask)
        if node_coor is not None and node_lh is not None:
            # We need to make a copy of coordinate arrays because we might be
            # making multiple calls to this function, and we don't want to
            # keep displacing coordinates.
            node_coor = node_coor.copy()
            node_coor[node_lh, 0] -= displacement
            node_coor[~node_lh, 0] += displacement
        if vol_coor is not None:
            left_mask = vol_coor[:, 0] < 0
            vol_coor = vol_coor.copy()
            vol_coor[left_mask, 0] -= displacement
            vol_coor[~left_mask, 0] += displacement
    elif surf is not None:
        for hemisphere in hemispheres:
            surf.__getattribute__(hemisphere).project(surf_projection)

    if surf is not None:
        for hemisphere in hemispheres:
            hemi_surf = surf.__getattribute__(hemisphere)
            # hemi_surf.project(surf_projection)
            hemi_clim = hemi_params.get(hemisphere, 'surf_scalars_clim')
            hemi_cmap = hemi_params.get(hemisphere, 'surf_scalars_cmap')
            hemi_color = None if hemi_cmap else 'white'
            if hemi_clim == 'robust' and surf_scalars is not None:
                hemi_clim = robust_clim(hemi_surf, surf_scalars)
            # TODO: copying the mesh seems like it could create memory issues.
            #       A better solution would be delayed execution.
            p.add_mesh(
                hemi_surf,
                opacity=surf_alpha,
                show_edges=False,
                scalars=surf_scalars,
                cmap=hemi_cmap,
                clim=hemi_clim,
                color=hemi_color,
                below_color=surf_scalars_below_color,
                copy_mesh=copy_actors,
            )
            if surf_scalars_boundary_width > 0 and surf_scalars is not None:
                p.add_mesh(
                    hemi_surf.contour(
                        isosurfaces=range(
                            int(max(hemi_surf.point_data[surf_scalars]))
                        )
                    ),
                    color=surf_scalars_boundary_color,
                    line_width=surf_scalars_boundary_width,
                )

    if vol_scalars is not None:
        assert (
            vol_coor is not None
        ), 'Volumetric scalars provided with unspecified coordinates'
        vol_scalars_point_size = vol_scalars_point_size or min(vol_voxdim[:3])
        p.add_points(
            vol_coor,
            render_points_as_spheres=False,
            style='points_gaussian',
            emissive=False,
            scalars=vol_scalars,
            opacity=vol_scalars_alpha,
            point_size=vol_scalars_point_size,
            ambient=1.0,
            cmap=vol_scalars_cmap,
            clim=vol_scalars_clim,
        )

    if node_coor is not None:
        for c, col, rad, opa in zip(
            node_coor,
            _map_to_color(node_values, node_color, None),
            _map_to_radius(node_values, node_radius, node_radius_range),
            _map_to_opacity(node_values, node_alpha),
        ):
            node = pv.Icosphere(
                radius=rad,
                center=c,
            )
            p.add_mesh(
                node,
                color=_get_color(color=col, cmap=node_cmap, clim=node_clim),
                opacity=opa,
            )
    if edge_values is not None:
        for c, d, ht, col, rad, opa in zip(
            *process_edge_values(),
            _map_to_color(edge_values, edge_color, None),
            _map_to_radius(edge_values, edge_radius, edge_radius_range),
            _map_to_opacity(edge_values, edge_alpha),
        ):
            edge = pv.Cylinder(
                center=c,
                direction=d,
                height=ht,
                radius=rad,
            )
            p.add_mesh(
                edge,
                color=_get_color(color=col, cmap=edge_cmap, clim=edge_clim),
                opacity=opa,
            )

    return p
