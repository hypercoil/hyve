# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unified plotter
~~~~~~~~~~~~~~~
Unified plotting function for surface, volume, and network data.
"""
import dataclasses
import io
from collections.abc import Mapping as MappingABC
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib import cm, colors, patheffects
from matplotlib.figure import Figure
from PIL import Image

from .const import (
    DEFAULT_CMAP,
    DEFAULT_COLOR,
    EDGE_ALPHA_DEFAULT_VALUE,
    EDGE_CLIM_DEFAULT_VALUE,
    EDGE_CMAP_DEFAULT_VALUE,
    EDGE_COLOR_DEFAULT_VALUE,
    EDGE_RADIUS_DEFAULT_VALUE,
    EDGE_RLIM_DEFAULT_VALUE,
    LAYER_ALPHA_DEFAULT_VALUE,
    LAYER_BELOW_COLOR_DEFAULT_VALUE,
    LAYER_BLEND_MODE_DEFAULT_VALUE,
    LAYER_CLIM_DEFAULT_VALUE,
    LAYER_CLIM_NEGATIVE_DEFAULT_VALUE,
    LAYER_CMAP_NEGATIVE_DEFAULT_VALUE,
    LAYER_COLOR_DEFAULT_VALUE,
    NETWORK_LAYER_BELOW_COLOR_DEFAULT_VALUE,
    NODE_ALPHA_DEFAULT_VALUE,
    NODE_CLIM_DEFAULT_VALUE,
    NODE_CMAP_DEFAULT_VALUE,
    NODE_COLOR_DEFAULT_VALUE,
    NODE_RADIUS_DEFAULT_VALUE,
    NODE_RLIM_DEFAULT_VALUE,
    POINTS_SCALARS_BELOW_COLOR_DEFAULT_VALUE,
    POINTS_SCALARS_CLIM_DEFAULT_VALUE,
    POINTS_SCALARS_CMAP_DEFAULT_VALUE,
    POINTS_SCALARS_DEFAULT_VALUE,
    POINTS_SCALARS_LAYERS_DEFAULT_VALUE,
    SCALAR_BAR_DEFAULT_BELOW_COLOR,
    SCALAR_BAR_DEFAULT_FONT,
    SCALAR_BAR_DEFAULT_FONT_COLOR,
    SCALAR_BAR_DEFAULT_FONT_OUTLINE_COLOR,
    SCALAR_BAR_DEFAULT_FONT_OUTLINE_WIDTH,
    SCALAR_BAR_DEFAULT_LENGTH,
    SCALAR_BAR_DEFAULT_LIM_FONTSIZE_MULTIPLIER,
    SCALAR_BAR_DEFAULT_LOC,
    SCALAR_BAR_DEFAULT_NAME,
    SCALAR_BAR_DEFAULT_NAME_FONTSIZE_MULTIPLIER,
    SCALAR_BAR_DEFAULT_NUM_SIG_FIGS,
    SCALAR_BAR_DEFAULT_ORIENTATION,
    SCALAR_BAR_DEFAULT_SIZE,
    SCALAR_BAR_DEFAULT_SPACING,
    SCALAR_BAR_DEFAULT_WIDTH,
    SURF_SCALARS_BELOW_COLOR_DEFAULT_VALUE,
    SURF_SCALARS_CLIM_DEFAULT_VALUE,
    SURF_SCALARS_CMAP_DEFAULT_VALUE,
    SURF_SCALARS_DEFAULT_VALUE,
    SURF_SCALARS_LAYERS_DEFAULT_VALUE,
    TYPICAL_DPI,
    Tensor,
)
from .surf import (
    CortexTriSurface,
)
from .util import (
    NetworkDataCollection,
    PointDataCollection,
    premultiply_alpha,
    robust_clim,
    source_over,
    unmultiply_alpha,
)

BLEND_MODES = {
    'source_over': source_over,
}


@dataclasses.dataclass(frozen=True)
class _LayerBase:
    """Base class for layers."""
    name: Optional[str]
    long_name: Optional[str] = None
    cmap: Optional[Any] = DEFAULT_CMAP
    clim: Optional[Tuple[float, float]] = LAYER_CLIM_DEFAULT_VALUE
    cmap_negative: Optional[Any] = LAYER_CMAP_NEGATIVE_DEFAULT_VALUE
    clim_negative: Optional[Tuple[float, float]] = (
        LAYER_CLIM_NEGATIVE_DEFAULT_VALUE
    )
    color: Optional[Any] = LAYER_COLOR_DEFAULT_VALUE
    alpha: float = LAYER_ALPHA_DEFAULT_VALUE
    below_color: Optional[Any] = LAYER_BELOW_COLOR_DEFAULT_VALUE
    hide_subthreshold: bool = False
    style: Optional[Mapping[str, Any]] = None
    scalar_bar_style: Optional[Mapping[str, Any]] = dataclasses.field(
        default_factory=dict,
    )
    blend_mode: Literal['source_over'] = LAYER_BLEND_MODE_DEFAULT_VALUE


# Right now, it's the same as the base class, but we might want to add
# additional parameters later.
@dataclasses.dataclass(frozen=True)
class Layer(_LayerBase):
    """Container for metadata to construct a single layer of a plot."""
    name: str
    hide_subthreshold: bool = True


@dataclasses.dataclass(frozen=True)
class EdgeLayer(_LayerBase):
    """Container for metadata to construct a single edge layer of a plot."""
    name: str
    clim: Optional[Tuple[float, float]] = EDGE_CLIM_DEFAULT_VALUE
    color: str = EDGE_COLOR_DEFAULT_VALUE
    radius: Union[float, str] = EDGE_RADIUS_DEFAULT_VALUE
    radius_range: Tuple[float, float] = EDGE_RLIM_DEFAULT_VALUE
    alpha: float = EDGE_ALPHA_DEFAULT_VALUE
    below_color: Optional[Any] = NETWORK_LAYER_BELOW_COLOR_DEFAULT_VALUE


@dataclasses.dataclass(frozen=True)
class NodeLayer(_LayerBase):
    """Container for metadata to construct a single node layer of a plot."""
    name: str
    clim: Optional[Tuple[float, float]] = NODE_CLIM_DEFAULT_VALUE
    color: str = NODE_COLOR_DEFAULT_VALUE
    radius: Union[float, str] = NODE_RADIUS_DEFAULT_VALUE
    radius_range: Tuple[float, float] = NODE_RLIM_DEFAULT_VALUE
    alpha: float = NODE_ALPHA_DEFAULT_VALUE
    below_color: Optional[Any] = NETWORK_LAYER_BELOW_COLOR_DEFAULT_VALUE
    edge_layers: Sequence[EdgeLayer] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class ScalarBarBuilder(MappingABC):
    """Addressable container for scalar bar parameters."""
    mapper: Optional[cm.ScalarMappable]
    name: Optional[str] = SCALAR_BAR_DEFAULT_NAME
    below_color: Optional[str] = SCALAR_BAR_DEFAULT_BELOW_COLOR
    length: int = SCALAR_BAR_DEFAULT_LENGTH
    width: int = SCALAR_BAR_DEFAULT_WIDTH
    orientation: Literal['h', 'v'] = SCALAR_BAR_DEFAULT_ORIENTATION
    num_sig_figs: int = SCALAR_BAR_DEFAULT_NUM_SIG_FIGS
    font: str = SCALAR_BAR_DEFAULT_FONT
    name_fontsize_multiplier: float = (
        SCALAR_BAR_DEFAULT_NAME_FONTSIZE_MULTIPLIER
    )
    lim_fontsize_multiplier: float = (
        SCALAR_BAR_DEFAULT_LIM_FONTSIZE_MULTIPLIER
    )
    font_color: Any = SCALAR_BAR_DEFAULT_FONT_COLOR
    font_outline_color: Any = (
        SCALAR_BAR_DEFAULT_FONT_OUTLINE_COLOR
    )
    font_outline_width: float = (
        SCALAR_BAR_DEFAULT_FONT_OUTLINE_WIDTH
    )

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __iter__(self):
        return iter(dataclasses.asdict(self))

    def __len__(self):
        return len(dataclasses.asdict(self))


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


def _normalise_to_range(values, valid_range):
    if valid_range is None:
        return values
    vmin = values.min()
    vmax = values.max()
    return (
        valid_range[0]
        + (valid_range[1] - valid_range[0])
        * (values - vmin)
        / (vmax - vmin)
    )


def _null_op(**params):
    return params


def _null_postprocessor(plotter):
    return plotter


def _null_auxwriter(metadata):
    return metadata


def build_scalar_bar(
    *,
    mapper: cm.ScalarMappable,
    name: Optional[str] = SCALAR_BAR_DEFAULT_NAME,
    below_color: Optional[str] = SCALAR_BAR_DEFAULT_BELOW_COLOR,
    length: int = SCALAR_BAR_DEFAULT_LENGTH,
    width: int = SCALAR_BAR_DEFAULT_WIDTH,
    orientation: Literal['h', 'v'] = SCALAR_BAR_DEFAULT_ORIENTATION,
    num_sig_figs: int = SCALAR_BAR_DEFAULT_NUM_SIG_FIGS,
    font: str = SCALAR_BAR_DEFAULT_FONT,
    name_fontsize_multiplier: float = (
        SCALAR_BAR_DEFAULT_NAME_FONTSIZE_MULTIPLIER
    ),
    lim_fontsize_multiplier: float = (
        SCALAR_BAR_DEFAULT_LIM_FONTSIZE_MULTIPLIER
    ),
    font_color: Any = SCALAR_BAR_DEFAULT_FONT_COLOR,
    font_outline_color: Any = (
        SCALAR_BAR_DEFAULT_FONT_OUTLINE_COLOR
    ),
    font_outline_width: float = (
        SCALAR_BAR_DEFAULT_FONT_OUTLINE_WIDTH
    ),
) -> Figure:
    name = name.upper() # TODO: change this! work into style
    vmin, vmax = mapper.get_clim()
    static_length = num_sig_figs * width // 2
    dynamic_length = length - 2 * static_length
    dynamic = mapper.to_rgba(np.linspace(vmin, vmax, dynamic_length))
    above = np.tile(mapper.to_rgba(vmax), (static_length, 1))
    if below_color is not None:
        if len(below_color) == 4 and below_color[-1] == 0:
            # Not ideal, but looks better than transparent and too many
            # color bars actually end in black
            below_color = '#444444'
        below = np.tile(colors.to_rgba(below_color), (static_length, 1))
    else:
        below = np.tile(mapper.to_rgba(vmin), (static_length, 1))
    rgba = np.stack(width * [np.concatenate([below, dynamic, above])])

    match orientation:
        case 'h':
            figsize = (length / TYPICAL_DPI, width / TYPICAL_DPI)
            vmin_params = {
                'xy': (0, 0),
                'xytext': (0.02, 0.5),
                'rotation': 0,
                'ha': 'left',
                'va': 'center',
            }
            vmax_params = {
                'xy': (1, 0),
                'xytext': (0.98, 0.5),
                'rotation': 0,
                'ha': 'right',
                'va': 'center',
            }
            name_params = {
                'xy': (0.5, 0),
                'xytext': (0.5, 0.5),
                'rotation': 0,
                'ha': 'center',
                'va': 'center',
            }
        case 'v':
            figsize = (width / TYPICAL_DPI, length / TYPICAL_DPI)
            rgba = rgba.swapaxes(0, 1)[::-1]
            vmin_params = {
                'xy': (0, 0),
                'xytext': (0.5, 0.02),
                'rotation': 90,
                'ha': 'center',
                'va': 'bottom',
            }
            vmax_params = {
                'xy': (0, 1),
                'xytext': (0.5, 0.98),
                'rotation': 90,
                'ha': 'center',
                'va': 'top',
            }
            name_params = {
                'xy': (0, .5),
                'xytext': (0.5, 0.5),
                'rotation': 90,
                'ha': 'center',
                'va': 'center',
            }
        case _:
            raise ValueError(f'Invalid orientation: {orientation}')

    f, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgba)
    for vlim, vlim_params in zip(
        (vmin, vmax),
        (vmin_params, vmax_params),
    ):
        ax.annotate(
            f'{vlim:.{num_sig_figs}g}',
            xycoords='axes fraction',
            fontsize=width * lim_fontsize_multiplier,
            fontfamily=font,
            color=font_color,
            path_effects=[
                patheffects.withStroke(
                    linewidth=font_outline_width,
                    foreground=font_outline_color,
                )
            ],
            **vlim_params,
        )
    if name is not None:
        ax.annotate(
            name,
            xycoords='axes fraction',
            fontsize=width * name_fontsize_multiplier,
            fontfamily=font,
            color=font_color,
            path_effects=[
                patheffects.withStroke(
                    linewidth=font_outline_width,
                    foreground=font_outline_color,
                )
            ],
            **name_params,
        )
    ax.axis('off')
    f.subplots_adjust(0, 0, 1, 1)
    return f


def collect_scalar_bars(
    plotter: pv.Plotter,
    builders: Sequence[ScalarBarBuilder],
    spacing: float = SCALAR_BAR_DEFAULT_SPACING,
    max_dimension: Optional[Tuple[int, int]] = (1920, 1080),
    require_unique_names: bool = True,
) -> Tuple[pv.Plotter, Tensor]:
    # Algorithm from https://stackoverflow.com/a/28268965
    if max_dimension is None:
        max_dimension = plotter.window_size
    if require_unique_names:
        unique_names = set()
        retained_builders = []
        for builder in builders:
            if builder.name not in unique_names:
                unique_names.add(builder.name)
                retained_builders.append(builder)
        builders = retained_builders
    count = len(builders)
    max_width, max_height = max_dimension
    if spacing < 1:
        spacing = spacing * min(max_width, max_height)
    spacing = int(spacing)
    width = max([
        builder.length
        if builder.orientation == 'h'
        else builder.width
        for builder in builders
    ])
    height = max([
        builder.width
        if builder.orientation == 'h'
        else builder.length
        for builder in builders
    ])
    candidates = np.concatenate((
        (max_width - np.arange(count) * spacing) /
        (np.arange(1, count + 1) * width),
        (max_height - np.arange(count) * spacing) /
        (np.arange(1, count + 1) * height),
    ))
    candidates = np.sort(candidates[candidates > 0])
    # Vectorising is probably faster than a binary search, so that's what
    # we're doing here
    n_col = np.floor((max_width + spacing) / (width * candidates + spacing))
    n_row = np.floor((max_height + spacing) / (height * candidates + spacing))
    feasible = (n_row * n_col) >= count
    layout_idx = np.max(np.where(feasible))
    layout = np.array([n_row[layout_idx], n_col[layout_idx]], dtype=int)
    scalar = candidates[layout_idx]
    width = int(scalar * width)
    height = int(scalar * height)
    # End algorithm from https://stackoverflow.com/a/28268965

    images = []
    buffers = []
    for builder in builders:
        builder_length = height if builder.orientation == 'v' else width
        builder_width = width if builder.orientation == 'v' else height
        builder = dataclasses.replace(
            builder,
            length=builder_length,
            width=builder_width,
        )
        fig = build_scalar_bar(**builder)
        # From https://stackoverflow.com/a/8598881
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', transparent=True)
        buffer.seek(0)
        images += [Image.open(buffer)]
        buffers += [buffer]

    #tight = {0: 'row', 1: 'col'}[np.argmin(layout)]
    argtight = np.argmin(layout)
    argslack = 1 - argtight
    counttight = layout[argtight]
    countslack = np.ceil(count / counttight).astype(int)
    layout = [None, None]
    layout[argtight] = counttight
    layout[argslack] = countslack

    # Things are about to get confusing because we're switching back and forth
    # between array convention and Cartesian convention. Sorry.
    canvas = Image.new('RGBA', (max_width, max_height), (0, 0, 0, 1))
    active_canvas_size = (
        layout[1] * width + (layout[1] - 1) * spacing,
        layout[0] * height + (layout[0] - 1) * spacing,
    )
    offset = [0, 0]
    offset[1 - argslack] = (
        canvas.size[1 - argslack] - active_canvas_size[1 - argslack]
    ) // 2
    for i, image in enumerate(images):
        # Unfortunately matplotlib invariably adds some padding, so an aspect
        # preserving resize is still necessary (Lanczos interpolation)
        image = image.resize((width, height), Image.ANTIALIAS)
        canvas.paste(image, tuple(int(o) for o in offset))
        # Filling this in row-major order
        if i % layout[1] == layout[1] - 1:
            offset[0] = 0
            offset[1] += height + spacing
        else:
            offset[0] += width + spacing
    canvas.show()

    for buffer in buffers:
        buffer.close()
    return plotter, np.array(canvas)


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
) -> Tuple[pv.Plotter, None]:
    # tuple, but we want to be tolerant if the user provides a list or
    # something
    if len(builders) == 0:
        return plotter, None
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
        fig = build_scalar_bar(**builder)
        scalar_bar = pv.ChartMPL(fig, size=bsize, loc=bloc)
        scalar_bar.background_color = (0, 0, 0, 0)
        scalar_bar.border_color = (0, 0, 0, 0)

        plotter.add_chart(scalar_bar)
    return plotter, None


def _rgba_impl(
    scalars: Tensor,
    clim: Optional[Tuple[float, float]] = None,
    cmap: Any = 'viridis',
    below_color: Optional[str] = None,
    hide_subthreshold: bool = False,
    scalar_bar_builder: Optional[ScalarBarBuilder] = None,
) -> Tuple[Tensor, Optional[ScalarBarBuilder]]:
    if clim == 'robust':
        clim = robust_clim(scalars)
    if clim is not None:
        vmin, vmax = clim
    else:
        vmin, vmax = scalars.min(), scalars.max()
        hide_subthreshold = False
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba = mapper.to_rgba(scalars)
    if below_color is not None and vmin is not None:
        rgba[scalars < vmin] = colors.to_rgba(below_color)
    elif vmin is not None and hide_subthreshold:
        # Set alpha to 0 for sub-threshold values
        rgba[scalars < vmin, 3] = 0
    if scalar_bar_builder is not None:
        scalar_bar_builder = ScalarBarBuilder(**{
            **scalar_bar_builder,
            **{
                'mapper': mapper,
                'below_color': below_color,
            }
        })
    return rgba, scalar_bar_builder


def scalars_to_rgba(
    scalars: Optional[Tensor] = None,
    clim: Optional[Tuple[float, float]] = None,
    clim_negative: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = None,
    cmap_negative: Optional[str] = None,
    color: Optional[str] = None,
    alpha: Optional[float] = None,
    below_color: Optional[str] = None,
    hide_subthreshold: bool = False,
    scalar_bar_builder: Optional[ScalarBarBuilder] = None,
) -> Tuple[Tensor, Sequence[Optional[ScalarBarBuilder]]]:
    """
    Convert scalar values to RGBA colors.

    Converting all scalars to RGBA colors enables us to plot multiple
    scalar values on the same surface by leveraging blend operations.

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
    below_color : str, optional
        Color to use for values below ``clim``. If ``clim_neg`` is also
        specified, this is the color to use for small absolute values
        between ``-clim_neg[0]`` and ``clim[0]``.
    hide_subthreshold : bool, optional
        If ``True``, set the alpha value to 0 for values below ``clim``.
    scalar_bar_builder : ScalarBarBuilder, optional
        Template for building scalar bars. If not specified, no scalar bar
        will be built.
    """
    if color is not None:
        rgba = np.tile(colors.to_rgba(color), (len(scalars), 1))
        if alpha is not None:
            rgba[:, 3] = alpha
        return rgba, (None,)

    scalar_bar_builder_negative = None
    if cmap_negative is not None:
        if clim_negative is None:
            clim_negative = clim
        scalars_negative = -scalars.copy()
        neg_idx = scalars_negative > 0
        scalars_negative[scalars_negative < 0] = 0
        scalars[neg_idx] = 0
        if scalar_bar_builder is not None:
            scalar_bar_builder_negative = ScalarBarBuilder(**{
                **scalar_bar_builder,
                **{'name': f'{scalar_bar_builder.name} (—)'}
            })
            scalar_bar_builder = ScalarBarBuilder(**{
                **scalar_bar_builder,
                **{'name': f'{scalar_bar_builder.name} (+)'}
            })
        rgba_neg, scalar_bar_builder_negative = _rgba_impl(
            scalars=scalars_negative,
            clim=clim_negative,
            cmap=cmap_negative,
            below_color=below_color,
            hide_subthreshold=hide_subthreshold,
            scalar_bar_builder=scalar_bar_builder_negative,
        )

    rgba, scalar_bar_builder = _rgba_impl(
        scalars=scalars,
        clim=clim,
        cmap=cmap,
        below_color=below_color,
        hide_subthreshold=hide_subthreshold,
        scalar_bar_builder=scalar_bar_builder,
    )
    if cmap_negative is not None:
        rgba[neg_idx] = rgba_neg[neg_idx]
    if alpha is not None:
        rgba[:, 3] *= alpha
    return rgba, (scalar_bar_builder, scalar_bar_builder_negative)


def layer_rgba(
    layer: Layer,
    scalar_array: Tensor,
) -> Tuple[Tensor, Sequence[ScalarBarBuilder]]:
    cmap = layer.cmap or DEFAULT_CMAP
    if layer.scalar_bar_style is not None:
        scalar_bar_builder = ScalarBarBuilder(mapper=None, name=layer.name)
    else:
        scalar_bar_builder = None
    rgba, scalar_bar_builders = scalars_to_rgba(
        scalars=scalar_array,
        cmap=cmap,
        clim=layer.clim,
        cmap_negative=layer.cmap_negative,
        clim_negative=layer.clim_negative,
        color=layer.color,
        alpha=layer.alpha,
        below_color=layer.below_color,
        hide_subthreshold=layer.hide_subthreshold,
        scalar_bar_builder=scalar_bar_builder,
    )
    if layer.scalar_bar_style is not None:
        # We should be able to override anything in the scalar bar builder
        # with the layer's scalar bar style, including the mapper and name
        # if we want to.
        scalar_bar_builders = tuple(
            ScalarBarBuilder(**{
                **scalar_bar_builder,
                **layer.scalar_bar_style,
            })
            for scalar_bar_builder in scalar_bar_builders
            if scalar_bar_builder is not None
        )
    else:
        scalar_bar_builders = ()

    return rgba, scalar_bar_builders


def surf_layer_rgba(
    surf: pv.PolyData,
    layer: Layer,
    data_domain: Literal['point_data', 'cell_data'],
) -> Tuple[Tensor, Sequence[ScalarBarBuilder]]:
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
    rgba, scalar_bar_builders = layer_rgba(layer, scalar_array)

    return rgba, scalar_bar_builders


def compose_layers(
    surf: pv.PolyData,
    layers: Sequence[Layer],
) -> Tuple[Tensor, str, Sequence[ScalarBarBuilder]]:
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
    dst = Layer(
        name=dst.name,
        cmap=cmap,
        clim=dst.clim,
        cmap_negative=dst.cmap_negative,
        clim_negative=dst.clim_negative,
        color=color,
        alpha=dst.alpha,
        below_color=dst.below_color,
        hide_subthreshold=dst.hide_subthreshold,
        scalar_bar_style=dst.scalar_bar_style,
    )
    dst, scalar_bar_builders = layer_rgba(dst, scalar_array)
    dst = premultiply_alpha(dst)

    for layer in layers[1:]:
        src, new_builders = surf_layer_rgba(surf, layer, data_domain)
        src = premultiply_alpha(src)
        blend_layers = BLEND_MODES[layer.blend_mode]
        dst = blend_layers(src, dst)
        scalar_bar_builders = scalar_bar_builders + new_builders
    dst = unmultiply_alpha(dst)
    return dst, data_domain, scalar_bar_builders


def add_composed_rgba(
    surf: pv.PolyData,
    layers: Sequence[Layer],
    surf_alpha: float,
) -> Tuple[pv.PolyData, str, Sequence[ScalarBarBuilder]]:
    rgba, data_domain, scalar_bar_builders = compose_layers(surf, layers)
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
    return surf, name, scalar_bar_builders


def add_points_scalars(
    plotter: pv.Plotter,
    points: PointDataCollection,
    layers: Sequence[Layer],
    copy_actors: bool = False,
) -> Tuple[pv.Plotter, Sequence[ScalarBarBuilder]]:
    # We could implement blend modes for points, but it's not clear
    # that it would be worth the tradeoff of potentially having to
    # compute the union of all coors in the dataset at every blend
    # step. Easy to implement with scipy.sparse, but not sure how it
    # would scale computationally. So instead, we're literally just
    # layering the points on top of each other. VTK might be smart
    # enough to automatically apply a reasonable blend mode even in
    # this regime.
    # TODO: Check out pyvista.StructuredGrid. Could be the right
    #       data structure for this.
    for layer in layers:
        dataset = points.get_dataset(layer.name)
        scalar_array = dataset.points.point_data[layer.name]
        rgba, scalar_bar_builders = layer_rgba(layer, scalar_array)
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
            copy_mesh=copy_actors,
        )
    return plotter, scalar_bar_builders


def build_edges_mesh(
    edge_values: pd.DataFrame,
    node_coor: np.ndarray,
    layer: EdgeLayer,
    radius: float,
) -> Tuple[pv.PolyData, Sequence[ScalarBarBuilder]]:
    edge_values = edge_values.reset_index()
    # DataFrame indices begin at 1 after we filter them, but we need them to
    # begin at 0 for indexing into node_coor.
    target = edge_values.dst.values - 1
    source = edge_values.src.values - 1
    midpoints = (node_coor[target] + node_coor[source]) / 2
    orientations = node_coor[target] - node_coor[source]
    norm = np.linalg.norm(orientations, axis=-1)

    edges = pv.PolyData(midpoints)
    if layer.color not in edge_values.columns:
        scalars = None
        color = layer.color
    else:
        scalars = edge_values[layer.color].values
        color = None
    if not isinstance(layer.alpha, str):
        alpha = layer.alpha
    else:
        alpha = None
    flayer = dataclasses.replace(layer, alpha=alpha, color=color)
    rgba, scalar_bar_builders = layer_rgba(flayer, scalars)
    if isinstance(layer.alpha, str):
        rgba[:, 3] = edge_values[layer.alpha].values
    elif alpha == 1:
        # We shouldn't have to do this, but for some reason either VTK or
        # PyVista is adding transparency even if we set alpha to 1 when we use
        # explicit RGBA to colour the mesh. Dropping the alpha channel
        # fixes this.
        rgba = rgba[:, :3]

    # TODO: This is a hack to get the glyphs to scale correctly.
    # The magic scalar is used to scale the glyphs to the correct radius.
    # Where does it come from? I have no idea. It's just what works. And
    # that, only roughly. No guarantees that it will work on new data.
    geom = pv.Cylinder(resolution=20, radius=0.01 * radius)

    edges.point_data[f'{layer.name}_norm'] = norm
    edges.point_data[f'{layer.name}_vecs'] = orientations
    edges.point_data[f'{layer.name}_rgba'] = rgba
    glyph = edges.glyph(
        scale=f'{layer.name}_norm',
        orient=f'{layer.name}_vecs',
        geom=geom,
        factor=1,
    )
    return glyph, scalar_bar_builders


def build_edges_meshes(
    edge_values: pd.DataFrame,
    node_coor: np.ndarray,
    layer: EdgeLayer,
    num_radius_bins: int = 10,
) -> Tuple[pv.PolyData, Sequence[ScalarBarBuilder]]:
    if not isinstance(layer.radius, str):
        radius_str = 'edge_radius'
        edge_radius = np.full(len(edge_values), layer.radius)
    else:
        radius_str = layer.radius
        edge_radius = edge_values[radius_str].values
        if layer.radius_range is not None:
            edge_radius = _normalise_to_range(
                edge_radius,
                layer.radius_range,
            )

    num_radius_bins = min(num_radius_bins, len(edge_radius))
    bins = np.quantile(
        edge_radius,
        np.linspace(0, 1, num_radius_bins + 1),
    )[1:]
    # bins = np.linspace(
    #     edge_radius.min(),
    #     edge_radius.max(),
    #     num_radius_bins + 1,
    # )[1:]
    asgt = np.digitize(edge_radius, bins, right=True)
    #assert num_radius_bins == len(np.unique(bins)), (
    assert num_radius_bins == len(bins), (
        'Binning failed to produce the correct number of bins. '
        'This is likely a bug. Please report it at '
        'https://github.com/hypercoil/hyve/issues.'
    )

    edges = pv.PolyData()
    for i in range(num_radius_bins):
        idx = asgt == i
        selected = edge_values[idx]
        if len(selected) == 0:
            continue
        # TODO: We're replacing the builders at every call. We definitely
        #       don't want to get multiple builders, but is this really the
        #       right way to do it? Double check to make sure it makes sense.
        mesh, scalar_bar_builders = build_edges_mesh(
            selected,
            node_coor,
            layer,
            bins[i],
        )
        edges = edges.merge(mesh)
    return edges, scalar_bar_builders


def build_nodes_mesh(
    node_values: pd.DataFrame,
    node_coor: np.ndarray,
    layer: NodeLayer,
) -> Tuple[pv.PolyData, Sequence[ScalarBarBuilder]]:
    node_values = node_values.reset_index()
    nodes = pv.PolyData(node_coor)
    if not isinstance(layer.radius, str):
        radius_str = 'node_radius'
        node_radius = np.full(len(node_coor), layer.radius)
    else:
        radius_str = layer.radius
        node_radius = node_values[radius_str].values
        if layer.radius_range is not None:
            node_radius = _normalise_to_range(
                node_radius,
                layer.radius_range,
            )

    if layer.color not in node_values.columns:
        scalars = None
        color = layer.color
    else:
        scalars = node_values[layer.color].values
        color = None
    if not isinstance(layer.alpha, str):
        alpha = layer.alpha
    else:
        alpha = None
    flayer = dataclasses.replace(layer, alpha=alpha, color=color)
    rgba, scalar_bar_builders = layer_rgba(flayer, scalars)
    if isinstance(layer.alpha, str):
        rgba[:, 3] = node_values[layer.alpha].values
    elif alpha == 1:
        # We shouldn't have to do this, but for some reason either VTK or
        # PyVista is adding transparency even if we set alpha to 1 when we use
        # explicit RGBA to colour the mesh. Dropping the alpha channel
        # fixes this.
        rgba = rgba[:, :3]

    nodes.point_data[radius_str] = node_radius
    nodes.point_data[f'{layer.name}_rgba'] = rgba
    glyph = nodes.glyph(
        scale=radius_str,
        orient=False,
        geom=pv.Icosphere(nsub=3),
    )
    return glyph, scalar_bar_builders


def add_network(
    plotter: pv.Plotter,
    networks: NetworkDataCollection,
    layers: Sequence[NodeLayer],
    num_edge_radius_bins: int = 10,
    copy_actors: bool = False,
) -> Tuple[pv.Plotter, Sequence[ScalarBarBuilder]]:
    # TODO: See if we're better off merging the nodes and edges into a
    #       single mesh, or if there's any reason to keep them separate.
    scalar_bar_builders = ()
    for layer in layers:
        network = networks.get_dataset(layer.name)
        node_coor = network.coor
        node_values = network.nodes
        edge_values = network.edges
        glyph, new_builder = build_nodes_mesh(node_values, node_coor, layer)
        plotter.add_mesh(
            glyph,
            scalars=f'{layer.name}_rgba',
            rgb=True,
            # shouldn't do anything here, but just in case
            copy_mesh=copy_actors,
        )

        layer_builders = new_builder
        for edge_layer in layer.edge_layers:
            glyphs, new_builder = build_edges_meshes(
                edge_values,
                node_coor,
                edge_layer,
                num_edge_radius_bins,
            )
            plotter.add_mesh(
                glyphs,
                scalars=f'{edge_layer.name}_rgba',
                rgb=True,
                copy_mesh=copy_actors,
            )
            if new_builder[0].name == layer_builders[0].name:
                new_builder = (dataclasses.replace(
                    new_builder[0],
                    name=f'{layer.name} (edges)',
                ),)
            layer_builders = layer_builders + new_builder
        scalar_bar_builders = scalar_bar_builders + layer_builders

    return plotter, scalar_bar_builders


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
    points_scalars_below_color: str = (
        POINTS_SCALARS_BELOW_COLOR_DEFAULT_VALUE
    ),
    points_scalars_layers: Optional[Sequence[Layer]] = (
        POINTS_SCALARS_LAYERS_DEFAULT_VALUE
    ),
    networks: Optional[NetworkDataCollection] = None,
    node_color: Optional[str] = NODE_COLOR_DEFAULT_VALUE,
    node_radius: Union[float, str] = NODE_RADIUS_DEFAULT_VALUE,
    node_radius_range: Tuple[float, float] = NODE_RLIM_DEFAULT_VALUE,
    node_cmap: Any = NODE_CMAP_DEFAULT_VALUE,
    node_clim: Tuple[float, float] = NODE_CLIM_DEFAULT_VALUE,
    node_alpha: Union[float, str] = NODE_ALPHA_DEFAULT_VALUE,
    edge_color: Optional[str] = EDGE_COLOR_DEFAULT_VALUE,
    edge_radius: Union[float, str] = EDGE_RADIUS_DEFAULT_VALUE,
    edge_radius_range: Tuple[float, float] = EDGE_RLIM_DEFAULT_VALUE,
    edge_cmap: Any = EDGE_CMAP_DEFAULT_VALUE,
    edge_clim: Tuple[float, float] = EDGE_CLIM_DEFAULT_VALUE,
    edge_alpha: Union[float, str] = 1.0,
    num_edge_radius_bins: int = 10,
    network_layers: Optional[Sequence[NodeLayer]] = None,
    hemisphere: Optional[Literal['left', 'right']] = None,
    hemisphere_slack: Optional[Union[float, Literal['default']]] = 'default',
    off_screen: bool = True,
    copy_actors: bool = False,
    theme: Optional[Any] = None,
    sbprocessor: Optional[callable] = None,
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

    p = pv.Plotter(off_screen=off_screen, theme=theme)

    # TODO: We can see that the conditionals below suggest a more general
    #       approach to plotting. We should refactor this code to make the
    #       unified plotter accept plotting primitives (e.g., surfaces,
    #       volumes, graphs) and then apply the appropriate plotting
    #       operations to them. This would make it easier to add new
    #       primitives in the future.
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
        elif networks is not None or points is not None:
            if networks is not None:
                ref_coor = np.concatenate([n.coor for n in networks])
                if any([n.lh_mask is None for n in networks]):
                    left_mask = ref_coor[:, 0] < 0
                else:
                    left_mask = np.concatenate([n.lh_mask for n in networks])
            elif points is not None:
                ref_coor = np.concatenate([p.points.points for p in points])
                left_mask = ref_coor[:, 0] < 0
            hw_left = (
                ref_coor[left_mask, 0].max()
                - ref_coor[left_mask, 0].min()
            ) / 2
            hw_right = (
                ref_coor[~left_mask, 0].max()
                - ref_coor[~left_mask, 0].min()
            ) / 2
            hemi_gap = (
                ref_coor[~left_mask, 0].max()
                + ref_coor[~left_mask, 0].min()
            ) / 2 - (
                ref_coor[left_mask, 0].max()
                + ref_coor[left_mask, 0].min()
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
        if networks is not None:
            if any([n.lh_mask is None for n in networks]):
                left_mask = ref_coor[:, 0] < 0
                def lh_condition(coor, _, __):
                    return coor[:, 0] < 0
                def rh_condition(coor, _, __):
                    return coor[:, 0] > 0
            else:
                left_mask = np.concatenate([n.lh_mask for n in networks])
                def lh_condition(_, __, lh_mask):
                    return lh_mask
                def rh_condition(_, __, lh_mask):
                    return ~lh_mask
            networks = networks.translate(
                (-displacement, 0, 0),
                condition=lh_condition,
            )
            networks = networks.translate(
                (displacement, 0, 0),
                condition=rh_condition,
            )
        if points is not None:
            left_points = points.translate(
                (-displacement, 0, 0),
                condition=lambda coor, _: coor[:, 0] < 0,
            )
            right_points = points.translate(
                (displacement, 0, 0),
                condition=lambda coor, _: coor[:, 0] > 0
            )
            points = PointDataCollection(
                lp + rp for lp, rp in zip(left_points, right_points)
            )
    elif surf is not None:
        for hemisphere in hemispheres:
            surf.__getattribute__(hemisphere).project(surf_projection)

    scalar_bar_builders = ()

    if surf is not None:
        for hemisphere in hemispheres:
            hemi_surf = surf.__getattribute__(hemisphere)
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
            hemi_surf, hemi_scalars, new_builders = add_composed_rgba(
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
            scalar_bar_builders = scalar_bar_builders + new_builders

    if points is not None:
        if points_scalars_layers is None:
            points_scalars_layers = []
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
        p, new_builders = add_points_scalars(
            plotter=p,
            points=points,
            layers=points_scalars_layers,
            copy_actors=copy_actors,
        )
        scalar_bar_builders = scalar_bar_builders + new_builders

    if networks is not None:
        if network_layers is None or len(network_layers) == 0:
            # No point in multiple datasets without overlays, so we'll use the
            # first network's specifications to build the base layer.
            base_network = networks[0]
            network_name = base_network.name
            if base_network.edges is not None:
                base_edge_layers = [EdgeLayer(
                    name=network_name,
                    cmap=edge_cmap,
                    clim=edge_clim,
                    color=edge_color,
                    radius=edge_radius,
                    radius_range=edge_radius_range,
                    alpha=edge_alpha,
                )]
            else:
                base_edge_layers = []
            if base_network.coor is not None:
                base_layer = NodeLayer(
                    name=network_name,
                    cmap=node_cmap,
                    clim=node_clim,
                    color=node_color,
                    radius=node_radius,
                    radius_range=node_radius_range,
                    alpha=node_alpha,
                    edge_layers=base_edge_layers,
                )
            network_layers = [base_layer]
        p, new_builders = add_network(
            plotter=p,
            networks=networks,
            layers=network_layers,
            num_edge_radius_bins=num_edge_radius_bins,
            copy_actors=copy_actors,
        )
        scalar_bar_builders = scalar_bar_builders + new_builders

    if sbprocessor is None:
        sbprocessor = overlay_scalar_bars
    p, scalar_bar = sbprocessor(plotter=p, builders=scalar_bar_builders)

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
