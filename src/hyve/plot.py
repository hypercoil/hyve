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
from matplotlib import cm, colors

from .const import Tensor
from .surf import (
    CortexTriSurface,
)
from .util import (
    PointData,
    PointDataCollection,
    premultiply_alpha,
    robust_clim,
    source_over,
    unmultiply_alpha,
)

DEFAULT_CMAP = 'viridis'
DEFAULT_COLOR = 'white'
BLEND_MODES = {
    'source_over': source_over,
}

LAYER_CLIM_DEFAULT_VALUE = None
LAYER_CMAP_NEGATIVE_DEFAULT_VALUE = None
LAYER_CLIM_NEGATIVE_DEFAULT_VALUE = None
LAYER_COLOR_DEFAULT_VALUE = None
LAYER_ALPHA_DEFAULT_VALUE = 1.0
LAYER_BELOW_COLOR_DEFAULT_VALUE = (0.0, 0.0, 0.0, 0.0)
LAYER_BLEND_MODE_DEFAULT_VALUE = 'source_over'

SURF_SCALARS_DEFAULT_VALUE = None
SURF_SCALARS_CMAP_DEFAULT_VALUE = (None, None)
SURF_SCALARS_CLIM_DEFAULT_VALUE = 'robust'
SURF_SCALARS_BELOW_COLOR_DEFAULT_VALUE = None
SURF_SCALARS_LAYERS_DEFAULT_VALUE = None

POINTS_SCALARS_DEFAULT_VALUE = None
POINTS_SCALARS_CMAP_DEFAULT_VALUE = None
POINTS_SCALARS_CLIM_DEFAULT_VALUE = None
POINTS_SCALARS_BELOW_COLOR_DEFAULT_VALUE = (0.0, 0.0, 0.0, 0.0)
POINTS_SCALARS_LAYERS_DEFAULT_VALUE = None


@dataclasses.dataclass(frozen=True)
class Layer:
    """Container for metadata to construct a single layer of a plot."""
    name: str
    cmap: Any = DEFAULT_CMAP
    clim: Optional[Tuple[float, float]] = LAYER_CLIM_DEFAULT_VALUE
    cmap_negative: Optional[Any] = LAYER_CMAP_NEGATIVE_DEFAULT_VALUE
    clim_negative: Optional[Tuple[float, float]] = (
        LAYER_CLIM_NEGATIVE_DEFAULT_VALUE
    )
    color: Optional[Any] = LAYER_COLOR_DEFAULT_VALUE
    alpha: float = LAYER_ALPHA_DEFAULT_VALUE
    below_color: Optional[Any] = LAYER_BELOW_COLOR_DEFAULT_VALUE
    blend_mode: Literal['source_over'] = LAYER_BLEND_MODE_DEFAULT_VALUE


@dataclasses.dataclass
class HemisphereParameters:
    """Addressable container for hemisphere-specific parameters."""
    left: Mapping[str, Any]
    right: Mapping[str, Any]

    def get(self, hemi, param):
        return self.__getattribute__(hemi)[param]


def _get_hemisphere_parameters(
    *,
    surf_scalars_cmap: Any,
    surf_scalars_clim: Any,
    surf_scalars_layers: Any,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    left = {}
    right = {}

    def assign_tuple(arg, name):
        left[name], right[name] = arg

    def assign_scalar(arg, name):
        left[name] = right[name] = arg

    def conditional_assign(condition, arg, name):
        if arg is not None and condition(arg):
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
    conditional_assign(
        lambda x: len(x) == 2 and isinstance(x[0], (tuple, list)),
        surf_scalars_layers,
        'surf_scalars_layers',
    )
    return HemisphereParameters(left, right)


def _cfg_hemispheres(
    hemisphere: Optional[Literal['left', 'right']] = None,
    surf_scalars: Optional[Union[str, Sequence[str]]] = None,
    surf: Optional[CortexTriSurface] = None,
):
    hemispheres = (
        (hemisphere,) if hemisphere is not None else ('left', 'right')
    )
    # TODO: Later on, we're going to add plot layering, which will allow us to
    #       plot multiple scalars at once by blending colours. One of these
    #       scalars can be the "key", which will be used to automatically
    #       remove any hemispheres that the scalar isn't present in. For now,
    #       we just use the only scalar that's present.
    key_scalars = surf_scalars
    if surf is not None and surf_scalars is not None:
        hemispheres = tuple(
            hemi for hemi in hemispheres
            if key_scalars is None or (
                key_scalars in surf.__getattribute__(hemi).point_data
                or key_scalars in surf.__getattribute__(hemi).cell_data
            )
        )
    if len(hemispheres) == 1:
        hemispheres_str = hemispheres[0]
    else:
        hemispheres_str = 'both'
    return hemispheres, hemispheres_str


def _get_color(color, cmap, clim):
    if (
        isinstance(color, str)
        or isinstance(color, tuple)
        or isinstance(color, list)
    ):
        return color
    else:
        try:
            cmap = cm.get_cmap(cmap)
        except ValueError:
            cmap = cmap
        vmin, vmax = clim
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
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


def _null_op(**params):
    return params


def _null_postprocessor(plotter):
    return plotter


def _null_auxwriter(metadata):
    return metadata


def _rgba_impl(
    scalars: Tensor,
    clim: Optional[Tuple[float, float]] = None,
    cmap: Any = 'viridis',
    below_color: Optional[str] = None,
):
    if clim == 'robust':
        clim = robust_clim(scalars)
    if clim is not None:
        vmin, vmax = clim
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        vmin, vmax, norm = None, None, None
    rgba = cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(scalars)
    if below_color is not None and vmin is not None:
        rgba[scalars < vmin] = colors.to_rgba(below_color)
    elif vmin is not None:
        # Set alpha to 0 for sub-threshold values
        rgba[scalars < vmin, 3] = 0
    return rgba


def scalars_to_rgba(
    scalars: Optional[Tensor] = None,
    clim: Optional[Tuple[float, float]] = None,
    clim_negative: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = None,
    cmap_negative: Optional[str] = None,
    color: Optional[str] = None,
    alpha: Optional[float] = None,
    below_color: Optional[str] = None,
) -> Tensor:
    """
    Convert scalar values to RGBA colors.

    Converting all scalars to RGBA colors enables us to plot multiple
    scalar values on the same surface by leveraging blend operations from
    PIL.

    Parameters
    ----------
    scalars : Tensor
        Scalar values to convert to RGBA colors.
    clim : tuple of float, optional
        Color limits. If ``clim_neg`` is also specified, this is the color
        limits for positive values.
    clim_neg : tuple of float, optional
        Color limits for negative values.
    cmap : str, optional
        Name of colormap to use for positive values.
    cmap_neg : str, optional
        Name of colormap to use for negative values.
    alpha : float, optional
        Opacity value to use for all scalar values, or opacity multiplier
        for the colormap(s).
    color : str, optional
        Color to use for all scalar values.
    """
    if color is not None:
        rgba = np.tile(colors.to_rgba(color), (len(scalars), 1))
        if alpha is not None:
            rgba[:, 3] = alpha
        return rgba

    if cmap_negative is not None:
        if clim_negative is None:
            clim_negative = clim
        scalars_negative = -scalars.copy()
        neg_idx = scalars_negative > 0
        scalars_negative[scalars_negative < 0] = 0
        scalars[neg_idx] = 0
        rgba_neg = _rgba_impl(
            scalars_negative,
            clim_negative,
            cmap_negative,
            below_color,
        )

    rgba = _rgba_impl(scalars, clim, cmap, below_color)
    if cmap_negative is not None:
        rgba[neg_idx] = rgba_neg[neg_idx]
    if alpha is not None:
        rgba[:, 3] *= alpha
    return rgba


def layer_rgba(
    surf: pv.PolyData,
    layer: Layer,
    data_domain: Literal['point_data', 'cell_data'],
) -> Tensor:
    """
    Convert a layer to RGBA colors.
    """
    try:
        scalar_array = surf.__getattribute__(
            data_domain
        )[layer.name]
    except KeyError:
        raise ValueError(
            'All layers must be defined over the same data '
            'domain. The base layer is defined over '
            f'{data_domain}, but the layer {layer.name} was not '
            f'found in {data_domain}. In particular, ensure that '
            'that, if you have mapped any layer to faces, all '
            'other layers are also mapped to faces.'
        )
    cmap = layer.cmap or DEFAULT_CMAP
    return scalars_to_rgba(
        scalars=scalar_array,
        cmap=cmap,
        clim=layer.clim,
        cmap_negative=layer.cmap_negative,
        clim_negative=layer.clim_negative,
        alpha=layer.alpha,
        below_color=layer.below_color,
    )


def compose_layers(
    surf: pv.PolyData,
    layers: Sequence[Layer],
) -> Tuple[Tensor, str]:
    """
    Compose layers into a single RGB(A) array.
    """
    dst = layers[0]
    data_domain = 'cell_data'
    if dst.name is not None:
        cmap = dst.cmap or DEFAULT_CMAP
        color = None
        try:
            scalar_array = surf.cell_data[dst.name]
        except KeyError:
            scalar_array = surf.point_data[dst.name]
            data_domain = 'point_data'
    else:
        try:
            surf.cell_data[layers[1].name]
            scalar_array = np.empty(surf.n_cells)
        except (KeyError, IndexError):
            data_domain = 'point_data'
            scalar_array = np.empty(surf.n_points)
        cmap = dst.cmap
        color = None if cmap else DEFAULT_COLOR
    dst = scalars_to_rgba(
        scalars=scalar_array,
        cmap=cmap,
        clim=dst.clim,
        cmap_negative=dst.cmap_negative,
        clim_negative=dst.clim_negative,
        color=color,
        alpha=dst.alpha,
        below_color=dst.below_color,
    )
    dst = premultiply_alpha(dst)

    for layer in layers[1:]:
        src = layer_rgba(surf, layer, data_domain)
        src = premultiply_alpha(src)
        blend_layers = BLEND_MODES[layer.blend_mode]
        dst = blend_layers(src, dst)
    dst = unmultiply_alpha(dst)
    return dst, data_domain


def add_composed_rgba(
    surf: pv.PolyData,
    layers: Sequence[Layer],
    surf_alpha: float,
) -> Tuple[pv.PolyData, str]:
    rgba, data_domain = compose_layers(surf, layers)
    # We shouldn't have to do this, but for some reason exporting to
    # HTML adds transparency even if we set alpha to 1 when we use
    # explicit RGBA to colour the mesh. Dropping the alpha channel
    # fixes this.
    # TODO: This will produce unexpected results for custom colormaps
    # that specify alpha.
    if surf_alpha == 1:
        rgba = rgba[:, :3]

    name = '-'.join([str(layer.name) for layer in layers])
    name = f'{name}_{data_domain}_rgba'
    if data_domain == 'cell_data':
        surf.cell_data[name] = rgba
    elif data_domain == 'point_data':
        surf.point_data[name] = rgba
    return surf, name


def add_points_scalars(
    plotter: pv.Plotter,
    points: PointDataCollection,
    layers: Sequence[Layer],
) -> pv.Plotter:
    # We could implement blend modes for points, but it's not clear
    # that it would be worth the tradeoff of potentially having to
    # compute the union of all coors in the dataset at every blend
    # step. Easy to implement with scipy.sparse, but not sure how it
    # would scale computationally. So instead, we're literally just
    # layering the points on top of each other. VTK might be smart
    # enough to automatically apply a reasonable blend mode even in
    # this regime.
    for layer in layers:
        dataset = points.get_dataset(layer.name)
        scalar_array = dataset.points.point_data[layer.name]
        rgba = scalars_to_rgba(
            scalars=scalar_array,
            cmap=layer.cmap,
            clim=layer.clim,
            cmap_negative=layer.cmap_negative,
            clim_negative=layer.clim_negative,
            color=layer.color,
            alpha=layer.alpha,
            below_color=layer.below_color,
        )
        plotter.add_points(
            points=dataset.points.points,
            render_points_as_spheres=False,
            style='points_gaussian',
            emissive=False,
            scalars=rgba,
            opacity=layer.alpha,
            point_size=dataset.point_size,
            ambient=1.0,
            rgb=True,
        )
    return plotter


def unified_plotter(
    *,
    surf: Optional['CortexTriSurface'] = None,
    surf_projection: str = 'pial',
    surf_alpha: float = 1.0,
    surf_scalars: Optional[str] = SURF_SCALARS_DEFAULT_VALUE,
    surf_scalars_boundary_color: str = 'black',
    surf_scalars_boundary_width: int = 0,
    surf_scalars_cmap: Any = SURF_SCALARS_CMAP_DEFAULT_VALUE,
    surf_scalars_clim: Any = SURF_SCALARS_CLIM_DEFAULT_VALUE,
    surf_scalars_below_color: str = SURF_SCALARS_BELOW_COLOR_DEFAULT_VALUE,
    surf_scalars_layers: Union[
        Optional[Sequence[Layer]],
        Tuple[Optional[Sequence[Layer]]]
    ] = SURF_SCALARS_LAYERS_DEFAULT_VALUE,
    points: Optional[PointDataCollection] = None,
    points_scalars: Optional[str] = POINTS_SCALARS_DEFAULT_VALUE,
    points_alpha: float = 1.0,
    points_scalars_cmap: Any = POINTS_SCALARS_CMAP_DEFAULT_VALUE,
    points_scalars_clim: Optional[Tuple] = POINTS_SCALARS_CLIM_DEFAULT_VALUE,
    points_scalars_below_color: str = POINTS_SCALARS_BELOW_COLOR_DEFAULT_VALUE,
    points_scalars_layers: Union[
        Optional[Sequence[Layer]],
        Tuple[Optional[Sequence[Layer]]]
    ] = POINTS_SCALARS_LAYERS_DEFAULT_VALUE,
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
    node_cmap: Any = DEFAULT_CMAP,
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
    postprocessors: Optional[Sequence[callable]] = None,
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
    surf : cortex.CortexTriSurface (default: ``None``)
        A surface to plot. If not specified, no surface will be plotted.
    surf_projection : str (default: ``'pial'``)
        The projection of the surface to plot. The projection must be
        available in the surface's ``projections`` attribute. For typical
        surfaces, available projections might include ``'pial'``,
        ``'inflated'``, ``veryinflated``, ``'white'``, and ``'sphere'``.
    surf_alpha : float (default: ``1.0``)
        The opacity of the surface.
    surf_scalars : str (default: ``None``)
        The name of the scalars to plot on the surface. The scalars must be
        available in the surface's ``point_data`` attribute. If not specified,
        no scalars will be plotted.
    surf_scalars_boundary_color : str (default: ``'black'``)
        The color of the boundary between the surface and the background. Note
        that this boundary is only visible if ``surf_scalars_boundary_width``
        is greater than 0.
    surf_scalars_boundary_width : int (default: ``0``)
        The width of the boundary between the surface and the background. If
        set to 0, no boundary will be plotted.
    surf_scalars_cmap : str or tuple (default: ``(None, None)``)
        The colormap to use for the surface scalars. If a tuple is specified,
        the first element is the colormap to use for the left hemisphere, and
        the second element is the colormap to use for the right hemisphere.
        If a single colormap is specified, it will be used for both
        hemispheres.
    surf_scalars_clim : str or tuple (default: ``'robust'``)
        The colormap limits to use for the surface scalars. If a tuple is
        specified, the first element is the colormap limits to use for the
        left hemisphere, and the second element is the colormap limits to use
        for the right hemisphere. If a single value is specified, it will be
        used for both hemispheres. If set to ``'robust'``, the colormap limits
        will be set to the 5th and 95th percentiles of the data.
        .. warning::
            If the colormap limits are set to ``'robust'``, the colormap
            limits will be calculated based on the data in the surface
            scalars, separately for each hemisphere. This means that the
            colormap limits may be different for each hemisphere, and the
            colors in the colormap may not be aligned between hemispheres.
    surf_scalars_below_color : str (default: ``'black'``)
        The color to use for values below the colormap limits.
    vol_coor : np.ndarray (default: ``None``)
        The coordinates of the volumetric data to plot. If not specified, no
        volumetric data will be plotted.
    vol_scalars : np.ndarray (default: ``None``)
        The volumetric data to plot. If not specified, no volumetric data will
        be plotted.
    vol_scalars_point_size : float (default: ``None``)
        The size of the points to plot for the volumetric data. If not
        specified, the size of the points will be automatically determined
        based on the size of the volumetric data.
    vol_voxdim : tuple (default: ``None``)
        The dimensions of the voxels in the volumetric data.
    vol_scalars_cmap : str (default: ``'viridis'``)
        The colormap to use for the volumetric data.
    vol_scalars_clim : tuple (default: ``None``)
        The colormap limits to use for the volumetric data.
    vol_scalars_alpha : float (default: ``1.0``)
        The opacity of the volumetric data.
    node_values : pd.DataFrame (default: ``None``)
        A table containing node-valued variables. Columns in the table can be
        used to specify attributes of plotted nodes, such as their color,
        radius, and opacity.
    node_coor : np.ndarray (default: ``None``)
        The coordinates of the nodes to plot. If not specified, no nodes will
        be plotted. Node coordinates can also be computed from a parcellation
        by specifying ``node_parcel_scalars``.
    node_parcel_scalars : str (default: ``None``)
        If provided, node coordinates will be computed as the centroids of
        parcels in the specified parcellation. The parcellation must be
        available in the ``point_data`` attribute of the surface. If not
        specified, node coordinates must be provided in ``node_coor`` or
        nodes will not be plotted.
    node_color : str or colour specification (default: ``'black'``)
        The color of the nodes. If ``node_values`` is specified, this argument
        can be used to specify a column in the table to use for the node
        colors.
    node_radius : float or str (default: ``3.0``)
        The radius of the nodes. If ``node_values`` is specified, this
        argument can be used to specify a column in the table to use for the
        node radii.
    node_radius_range : tuple (default: ``(2, 10)``)
        The range of node radii to use. The values in ``node_radius`` will be
        linearly scaled to this range.
    node_cmap : str or matplotlib colormap (default: ``'viridis'``)
        The colormap to use for the nodes.
    node_clim : tuple (default: ``(0, 1)``)
        The range of values to map into the dynamic range of the colormap.
    node_alpha : float or str (default: ``1.0``)
        The opacity of the nodes. If ``node_values`` is specified, this
        argument can be used to specify a column in the table to use for the
        node opacities.
    node_lh : np.ndarray (default: ``None``)
        Boolean-valued array indicating which nodes belong to the left
        hemisphere.
    edge_values : pd.DataFrame (default: ``None``)
        A table containing edge-valued variables. The table must have a
        MultiIndex with two levels, where the first level contains the
        starting node of each edge, and the second level contains the ending
        node of each edge. Additional columns can be used to specify
        attributes of plotted edges, such as their color, radius, and
        opacity.
    edge_color : str or colour specification (default: ``'edge_sgn'``)
        The color of the edges. If ``edge_values`` is specified, this argument
        can be used to specify a column in the table to use for the edge
        colors. By default, edges are colored according to the value of the
        ``'edge_sgn'`` column in ``edge_values``, which is 1 for positive
        edges and -1 for negative edges when the edges are digested by the
        ``filter_adjacency_data`` function using the default settings.
    edge_radius : float or str (default: ``'edge_val'``)
        The radius of the edges. If ``edge_values`` is specified, this
        argument can be used to specify a column in the table to use for the
        edge radii. By default, edges are sized according to the value of the
        ``'edge_val'`` column in ``edge_values``, which is the absolute value
        of the edge weight when the edges are digested by the
        ``filter_adjacency_data`` function using the default settings.
    edge_radius_range : tuple (default: ``(0.1, 1.8)``)
        The range of edge radii to use. The values in ``edge_radius`` will be
        linearly scaled to this range.
    edge_cmap : str or matplotlib colormap (default: ``'RdYlBu'``)
        The colormap to use for the edges.
    edge_clim : tuple (default: ``None``)
        The range of values to map into the dynamic range of the colormap.
    edge_alpha : float or str (default: ``1.0``)
        The opacity of the edges. If ``edge_values`` is specified, this
        argument can be used to specify a column in the table to use for the
        edge opacities.
    hemisphere : str (default: ``None``)
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
    off_screen : bool (default: ``True``)
        Whether to render the plot off-screen. If ``False``, a window will
        appear containing an interactive plot.
    copy_actors : bool (default: ``True``)
        Whether to copy the actors before returning them. If ``False``, the
        actors will be modified in-place.
    theme : PyVista plotter theme (default: ``None``)
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

    # Sometimes by construction surf_scalars is an empty tuple or list, which
    # causes problems later on. So we convert it to None if it's empty.
    if surf_scalars is not None:
        surf_scalars = None if len(surf_scalars) == 0 else surf_scalars
    hemispheres, hemisphere_str = _cfg_hemispheres(
        hemisphere=hemisphere,
        surf_scalars=surf_scalars,
        surf=surf,
    )
    hemi_params = _get_hemisphere_parameters(
        surf_scalars_cmap=surf_scalars_cmap,
        surf_scalars_clim=surf_scalars_clim,
        surf_scalars_layers=surf_scalars_layers,
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
            hemi_layers = hemi_params.get(hemisphere, 'surf_scalars_layers')
            if hemi_layers is None:
                hemi_layers = []

            base_layer = Layer(
                name=surf_scalars,
                cmap=hemi_cmap,
                clim=hemi_clim,
                cmap_negative=None,
                color=None,
                alpha=surf_alpha,
                below_color=surf_scalars_below_color,
            )
            hemi_layers = [base_layer] + list(hemi_layers)
            hemi_surf, hemi_scalars = add_composed_rgba(
                surf=hemi_surf,
                layers=hemi_layers,
                surf_alpha=surf_alpha,
            )
            # TODO: copying the mesh seems like it could create memory issues.
            #       A better solution would be delayed execution.
            p.add_mesh(
                hemi_surf,
                scalars=hemi_scalars,
                rgb=True,
                show_edges=False,
                copy_mesh=copy_actors,
            )
            # p.add_mesh(
            #     hemi_surf,
            #     scalars=surf_scalars,
            #     show_edges=False,
            #     cmap=hemi_cmap,
            #     clim=hemi_clim,
            #     opacity=surf_alpha,
            #     color=hemi_color,
            #     below_color=surf_scalars_below_color,
            #     copy_mesh=copy_actors,
            # )
            if (
                surf_scalars_boundary_width > 0
                and surf_scalars is not None
                and surf_scalars not in hemi_surf.cell_data
            ):
                p.add_mesh(
                    hemi_surf.contour(
                        isosurfaces=range(
                            int(max(hemi_surf.point_data[surf_scalars]))
                        ),
                        scalars=surf_scalars,
                    ),
                    color=surf_scalars_boundary_color,
                    line_width=surf_scalars_boundary_width,
                )

    if points is not None:
        if points_scalars is not None:
            base_layer = Layer(
                name=points_scalars,
                cmap=points_scalars_cmap,
                clim=points_scalars_clim,
                cmap_negative=None,
                color=None,
                alpha=points_alpha,
                below_color=points_scalars_below_color,
            )
            points_scalars_layers = [base_layer] + list(points_scalars_layers)
        p = add_points_scalars(
            plotter=p,
            points=points,
            layers=points_scalars_layers,
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

    if postprocessors is None or len(postprocessors) == 0:
        postprocessors = [_null_postprocessor]
    postprocessors = [
        w if w is not None else _null_postprocessor for w in postprocessors
    ]
    for i, w in enumerate(postprocessors):
        try:
            postprocessors[i] = w.bind(**locals())
        except AttributeError:
            pass
    out = tuple(w(plotter=p) for w in postprocessors)
    return out


def plotted_entities(
    *,
    entity_writers: Optional[Sequence[callable]] = None,
    plot_index: Optional[int] = None,
    **params,
) -> Mapping[str, Sequence[str]]:
    metadata = {}
    surface = params.get('surf', None)
    node = params.get('node_values', None)
    edge = params.get('edge_values', None)
    _, hemisphere_str = _cfg_hemispheres(
        hemisphere=params.get('hemisphere', None),
        surf_scalars=params.get('surf_scalars', None),
        surf=params.get('surf', None),
    )
    metadata['hemisphere'] = [hemisphere_str]
    if surface is not None:
        metadata['scalars'] = [params.get('surf_scalars', None)]
        metadata['projection'] = [params.get('surf_projection', None)]
        layers = params.get('surf_scalars_layers', None)
        if layers is not None:
            if len(layers) == 2 and isinstance(layers[0], (list, tuple)):
                layers = layers[0] if hemisphere_str != 'right' else layers[1]
            layers = '+'.join(layer.name for layer in layers)
            metadata['scalars'] = (
                [f'{metadata["scalars"][0]}+{layers}']
                if metadata['scalars'][0] is not None
                else [layers]
            )
    if node is not None:
        metadata['parcellation'] = [params.get('node_parcel_scalars', None)]
        metadata['nodecolor'] = [params.get('node_color', None)]
        metadata['noderadius'] = [params.get('node_radius', None)]
        metadata['nodealpha'] = [params.get('node_alpha', None)]
    if edge is not None:
        metadata['edgecolor'] = [params.get('edge_color', None)]
        metadata['edgeradius'] = [params.get('edge_radius', None)]
        metadata['edgealpha'] = [params.get('edge_alpha', None)]
    if metadata['hemisphere'][0] is None:
        metadata['hemisphere'] = ['both']
    metadata['plot_index'] = [plot_index]
    metadata = {k: v for k, v in metadata.items() if isinstance(v[0], str)}
    if entity_writers is None:
        entity_writers = [_null_auxwriter]
    entity_writers = [
        w if w is not None else _null_auxwriter for w in entity_writers
    ]
    for i, w in enumerate(entity_writers):
        try:
            entity_writers[i] = w.bind(**{**params, **metadata})
        except AttributeError:
            pass
    metadata = tuple(w(metadata=metadata) for w in entity_writers)
    return metadata
