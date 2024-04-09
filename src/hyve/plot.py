# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unified plotter and plotting primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unified plotting function for surface, volume, and network data.
"""
import dataclasses
import inspect
import io
from functools import WRAPPER_ASSIGNMENTS, wraps
from typing import (
    Any,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
import pyvista as pv
from conveyant import Primitive, emulate_assignment

from .const import (
    DEFAULT_WINDOW_SIZE,
    SCALAR_BAR_DEFAULT_LOC,
    SCALAR_BAR_DEFAULT_SIZE,
    SCALAR_BAR_DEFAULT_SPACING,
)
from .elements import ScalarBarBuilder, _uniquify_names
from .geom.prim import (
    GeomPrimitive,
    plot_surf_p,
    plot_points_p,
    plot_network_p,
)
from .geom.transforms import (
    GeomTransform,
    hemisphere_select,
    hemisphere_slack,
)


def _default_key_scalars(params: Mapping[str, Any]):
    key_scalars = params.get('surf_scalars', None)
    if key_scalars is None:
        key_scalars = params.get('surf_scalars_layers', None)
        if key_scalars is not None:
            try:
                key_scalars = key_scalars[0].name
            except AttributeError:
                # Two separate lists of layers, one for each hemisphere
                key_scalars = key_scalars[0][0].name
    if not key_scalars:
        key_scalars = None
    return key_scalars


def _null_op(**params):
    return params


def _null_sbprocessor(
    plotter: pv.Plotter,
    builders: Sequence[ScalarBarBuilder],
) -> Tuple[pv.Plotter, Sequence[ScalarBarBuilder]]:
    return plotter, builders


def _null_postprocessor(plotter):
    return plotter


def _null_auxwriter(metadata):
    return metadata


def overlay_scalar_bars(
    plotter: pv.Plotter,
    builders: Sequence[ScalarBarBuilder],
    loc: Union[
        Tuple[float, float],
        Mapping[str, Tuple[float, float]],
    ] = SCALAR_BAR_DEFAULT_LOC,
    size: Union[
        Tuple[float, float],
        Mapping[str, Tuple[float, float]],
    ] = SCALAR_BAR_DEFAULT_SIZE,
    default_spacing: float = SCALAR_BAR_DEFAULT_SPACING,
    require_unique_names: bool = True,
) -> Tuple[pv.Plotter, None]:
    # tuple, but we want to be tolerant if the user provides a list or
    # something
    if len(builders) == 0:
        return plotter, None
    if require_unique_names:
        builders = _uniquify_names(builders)
    if loc is not None and not isinstance(loc, Mapping):
        loc = {'__start__': loc}
    if loc is None or '__start__' not in loc:
        # This is gonna break when people want to plot multiple scalar
        # bars with different orientations. We'll cross that bridge when
        # we get there.
        if builders[0].orientation == 'v':
            loc = {'__start__': (0.02, 0.1)}
        elif builders[0].orientation == 'h':
            loc = {'__start__': (0.1, 0.02)}
    if size is not None and not isinstance(size, Mapping):
        size = {'__default__': size}
    if size is None or '__default__' not in size:
        # This is gonna break when people want to plot multiple scalar
        # bars with different orientations. We'll cross that bridge when
        # we get there.
        if builders[0].orientation == 'v':
            size = {'__default__': (0.05, 0.8)}
        elif builders[0].orientation == 'h':
            size = {'__default__': (0.8, 0.05)}

    offset = 0
    for builder in builders:
        bloc = loc.get(builder.name, loc['__start__'])
        bsize = size.get(builder.name, size['__default__'])
        if bloc == loc['__start__']:
            if builder.orientation == 'v':
                bloc = (bloc[0] + offset, bloc[1])
                offset += (bsize[0] + default_spacing)
            elif builder.orientation == 'h':
                bloc = (bloc[0], bloc[1] + offset)
                offset += (bsize[1] + default_spacing)
        if plotter.window_size is not None:
            aspect_ratio = builder.width / builder.length
            if builder.orientation == 'v':
                length = round(bsize[1] * plotter.window_size[1])
            elif builder.orientation == 'h':
                length = round(bsize[0] * plotter.window_size[0])
            width = round(length * aspect_ratio)
            builder = dataclasses.replace(
                builder,
                width=width,
                length=length,
            )
        fig = builder(backend='matplotlib')
        # TODO: This is really a terrible hack to get the sizes to always be
        #       consistent. I'm not sure why the sizes are inconsistent in the
        #       first place. But this is all the more reason to drop
        #       matplotlib and switch to building an SVG programmatically.
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', transparent=True)
        buffer.seek(0)
        f, ax = plt.subplots(figsize=fig.get_size_inches())
        ax.imshow(plt.imread(buffer))
        ax.axis('off')
        f.subplots_adjust(0, 0, 1, 1)
        buffer.close()

        scalar_bar = pv.ChartMPL(f, size=bsize, loc=bloc)
        scalar_bar.background_color = (0, 0, 0, 0)
        scalar_bar.border_color = (0, 0, 0, 0)

        plotter.add_chart(scalar_bar)
    return plotter, None


def bind_primitives(
    prims: Sequence[GeomPrimitive],
    transforms: Optional[Sequence[GeomTransform]] = None,
) -> callable:
    def _bind_primitives_inner(
        plot_func: Optional[callable] = None,
        meta_func: Optional[callable] = None,
    ) -> callable:
        plot_prim = [
            Primitive(
                prim.plot,
                name=prim.name,
                output=('plotter', 'scalar_bar_builders'),
                forward_unused=False,
            )
            for prim in prims
        ]
        meta_prim = [
            Primitive(
                prim.meta,
                name=prim.name,
                output=None,
                forward_unused=False,
            )
            for prim in prims
        ]

        if transforms is None:
            topo_signatures = []
            topo_params = []
        else:
            topo_signatures = [
                inspect.signature(topo.fit) for topo in transforms
            ]
            topo_params = [
                p
                for topo_signature in topo_signatures
                for p in topo_signature.parameters.values()
                if p.kind == p.KEYWORD_ONLY
            ]

        if plot_func is None:
            transformed_plot_func = None
        else:
            @wraps(
                plot_func,
                assigned=WRAPPER_ASSIGNMENTS + ('__kwdefaults__',),
            )
            def transformed_plot_func(**params):
                plot_primitives = params.pop('plot_primitives', [])
                plot_primitives = list(plot_primitives) + plot_prim

                topo_transforms = params.pop('topo_transforms', [])
                if transforms is not None:
                    topo_transforms = list(topo_transforms) + list(transforms)
                for i, topo in enumerate(topo_transforms):
                    fit, transform = topo.fit, topo.transform
                    for prim in prims:
                        _fit, _transform = prim.xfms.get(
                            topo.name, (None, None)
                        )
                        if _fit is not None:
                            fit = _fit(fit)
                        if _transform is not None:
                            transform = _transform(transform)
                    topo_transforms[i] = GeomTransform(
                        topo.name, fit, transform
                    )

                return plot_func(
                    plot_primitives=plot_primitives,
                    topo_transforms=topo_transforms,
                    **params,
                )

            prim_signatures = [inspect.signature(prim.plot) for prim in prims]
            func_signature = inspect.signature(plot_func)
            prim_params = [
                p
                for prim_signature in prim_signatures
                for p in prim_signature.parameters.values()
                if p.kind == p.KEYWORD_ONLY
            ]
            func_params = [
                p for p in func_signature.parameters.values()
                if p.kind != p.VAR_KEYWORD and p.kind != p.VAR_POSITIONAL
            ]

            transformed_signature = func_signature.replace(
                parameters=func_params + prim_params + topo_params
            )
            transformed_plot_func.__signature__ = transformed_signature
            # transformed_plot_func.__kwdefaults__ = {
            #     **(transformed_plot_func.__kwdefaults__ or {}),
            #     **{
            #         p.name: p.default
            #         for p in prim_params
            #         if p.default is not p.empty
            #     },
            # }

        if meta_func is None:
            transformed_meta_func = None
        else:
            @wraps(
                meta_func,
                assigned=WRAPPER_ASSIGNMENTS + ('__kwdefaults__',),
            )
            def transformed_meta_func(**params):
                meta_primitives = params.pop('meta_primitives', [])
                meta_primitives = list(meta_primitives) + meta_prim
                meta_transforms = params.pop('meta_transforms', [])
                new_meta_transforms = transforms or []
                new_meta_transforms = [e for e in transforms if e.meta is not None]
                meta_transforms = list(meta_transforms) + list(new_meta_transforms)
                return meta_func(
                    meta_primitives=meta_primitives,
                    meta_transforms=meta_transforms,
                    **params
                )

            prim_signatures = [inspect.signature(prim.meta) for prim in prims]
            func_signature = inspect.signature(meta_func)
            prim_params = [
                p
                for prim_signature in prim_signatures
                for p in prim_signature.parameters.values()
                if p.kind == p.KEYWORD_ONLY
            ]
            func_params = [
                p for p in func_signature.parameters.values()
                if p.kind != p.VAR_KEYWORD and p.kind != p.VAR_POSITIONAL
            ]

            transformed_signature = func_signature.replace(
                parameters=func_params + prim_params + topo_params
            )
            transformed_meta_func.__signature__ = transformed_signature

        return (
            emulate_assignment()(transformed_plot_func),
            emulate_assignment(strict=False)(transformed_meta_func),
        )
    return _bind_primitives_inner


# TODO: There are some nasty anti-patterns in this function:
#       - using locals() and indefinite kwargs
#       - arguments and variables that are apparently unused / not accessed
#         but are in fact packaged into the params dict
#       - delayed binding of parameters
#       I'm not sure this can be improved without substantially reducing the
#       flexibility of the function. This is supposed to be low-level code
#       that is used to build higher-level functions, so a potential approach
#       could be having the operations that build the higher-level functions
#       be responsible for sanitising the inputs to this function and building
#       an informative documentation string.
def base_plotter(
    *,
    plot_primitives: Sequence[Primitive] = (),
    topo_transforms: Optional[Sequence[GeomTransform]] = (),
    key_scalars: Optional[str] = '__default__',
    plotter: Optional[pv.Plotter] = None,
    off_screen: bool = True,
    copy_actors: bool = False,
    theme: Optional[Any] = None,
    window_size: Tuple[int, int] = DEFAULT_WINDOW_SIZE,
    return_builders: bool = False,
    sbprocessor: Optional[callable] = None,
    empty_builders: bool = False,
    postprocessors: Optional[Sequence[callable]] = None,
    **params,
) -> Tuple[Any]:

    # TODO: cortex_theme doesn't work here for some reason. If the background
    #       is transparent, all of the points are also made transparent. So
    #       we're sticking with a white background for now.
    theme = theme or pv.themes.DocumentTheme()

    if plotter is None:
        plotter = pv.Plotter(
            window_size=window_size,
            off_screen=off_screen,
            theme=theme,
        )
        close_plotter = True # for potential use by postprocessors at binding
    else:
        plotter.clear()
        plotter.enable_lightkit()
        plotter.window_size = window_size or (1920, 1080)
        close_plotter = False # for potential use by postprocessors at binding

    if key_scalars == '__default__':
        key_scalars = _default_key_scalars(params)

    params = {**locals(), **params}
    for transform in topo_transforms or ():
        fit_params = transform.fit(**params)
        result = transform.transform(fit_params=fit_params, **params)
        params = {**params, **result}

    scalar_bar_builders = ()
    params = {**locals(), **params}
    for prim in plot_primitives or ():
        result = prim(**params)
        params = {**params, **result}

    if sbprocessor is None:
        sbprocessor = overlay_scalar_bars
    if empty_builders:
        plotter, scalar_bar = sbprocessor(
            plotter=params['plotter'],
            builders=(),
        )
    else:
        plotter, scalar_bar = sbprocessor(
            plotter=params['plotter'],
            builders=params['scalar_bar_builders'],
        )
    builders = {'scalar_bar': scalar_bar}

    if postprocessors is None or len(postprocessors) == 0:
        postprocessors = [_null_postprocessor]
    postprocessors = [
        w if w is not None else _null_postprocessor for w in postprocessors
    ]
    for i, w in enumerate(postprocessors):
        try:
            postprocessors[i] = w.bind(**params)
        except AttributeError:
            pass
    out = tuple(w(plotter=plotter) for w in postprocessors)

    if return_builders:
        return out, builders
    return out


def base_plotmeta(
    *,
    meta_primitives: Sequence[Primitive] = (),
    meta_transforms: Optional[Sequence[GeomTransform]] = (),
    key_scalars: Optional[str] = '__default__',
    entity_writers: Optional[Sequence[callable]] = None,
    **params,
) -> Mapping[str, Sequence[str]]:
    metadata = {}

    if key_scalars == '__default__':
        key_scalars = _default_key_scalars(params)
    for transform in meta_transforms or ():
        xfm_metadata = transform.meta(**params)
        metadata = {**metadata, **xfm_metadata}

    for prim in meta_primitives or ():
        metadata = prim(metadata=metadata, **params)

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


unified_plotter, plotted_entities = bind_primitives(
    prims=(
        plot_surf_p,
        plot_points_p,
        plot_network_p,
    ),
    transforms=(
        hemisphere_select,
        hemisphere_slack,
    ),
)(base_plotter, base_plotmeta)
unified_plotter.__doc__ = """
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
    surf_scalars_layers : list of Layer (default: ``None``)
        A list of layers to plot on the surface. Each layer is defined by a
        ``Layer`` object, which specifies the name of the layer, the colormap
        to use, the colormap limits, the color, the opacity, and the blend
        mode. If not specified, no layers will be plotted.
    points : PointDataCollection (default: ``None``)
        A collection of points to plot. If not specified, no points will be
        plotted.
    points_scalars : str (default: ``None``)
        The name of the scalars to plot on the points. The scalars must be
        available in the points' ``point_data`` attribute. If not specified,
        no scalars will be plotted.
    points_alpha : float (default: ``1.0``)
        The opacity of the points.
    points_scalars_cmap : str (default: ``None``)
        The colormap to use for the points scalars.
    points_scalars_clim : tuple (default: ``None``)
        The colormap limits to use for the points scalars.
    points_scalars_below_color : str (default: ``'black'``)
        The color to use for values below the colormap limits.
    points_scalars_layers : list of Layer (default: ``None``)
        A list of layers to plot on the points. Each layer is defined by a
        ``Layer`` object, which specifies the name of the layer, the colormap
        to use, the colormap limits, the color, the opacity, and the blend
        mode. If not specified, no layers will be plotted.
    networks : NetworkDataCollection (default: ``None``)
        A collection of networks to plot. If not specified, no networks will
        be plotted. Each network in the collection must include a ``'coor'``
        attribute, which specifies the coordinates of the nodes in the
        network. The coordinates must be specified as a ``(N, 3)`` array,
        where ``N`` is the number of nodes in the network. The collection may
        also optionally include a ``'nodes'`` attribute, The node attributes
        must be specified as a ``pandas.DataFrame`` with ``N`` rows, where
        ``N`` is the number of nodes in the network. The collection may also
        optionally include an ``'edges'`` attribute, which specifies the
        edges in the network. The edge attributes must be specified as a
        ``pandas.DataFrame`` with ``M`` rows, where ``M`` is the number of
        edges in the network. Finally, the collection may also optionally
        include a ``lh_mask`` attribute, which is a boolean-valued array
        indicating which nodes belong to the left hemisphere.
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
    num_edge_radius_bins : int (default: ``10``)
        The number of bins to use when binning the edges by radius. Because
        edges are intractable to render when there are many of them, this
        argument can be used to bin the edges by radius and render each bin
        separately. This can significantly improve performance when there are
        many edges but will result in a loss of detail.
    network_layers: list of NodeLayer (default: ``None``)
        A list of layers to plot on the networks. Each layer is defined by a
        ``NodeLayer`` object, which specifies the name of the layer, together
        with various parameters for the nodes and edges in the layer. If not
        specified, a single layer will be created for the first network in
        ``networks``.
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
