# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Geometric primitives
~~~~~~~~~~~~~~~~~~~~
Geometric primitives and geometric transforms.
"""
import dataclasses
from collections.abc import Mapping as MappingABC
from typing import (
    Any,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from matplotlib import cm, colors

from ..elements import ScalarBarBuilder
from ..const import (
    Tensor,
    DEFAULT_CMAP,
    LAYER_ALIM_DEFAULT_VALUE,
    LAYER_ALIM_NEGATIVE_DEFAULT_VALUE,
    LAYER_ALIM_PERCENTILE_DEFAULT_VALUE,
    LAYER_ALPHA_DEFAULT_VALUE,
    LAYER_ALPHA_NEGATIVE_DEFAULT_VALUE,
    LAYER_AMAP_DEFAULT_VALUE,
    LAYER_AMAP_NEGATIVE_DEFAULT_VALUE,
    LAYER_BELOW_COLOR_DEFAULT_VALUE,
    LAYER_BLEND_MODE_DEFAULT_VALUE,
    LAYER_CLIM_DEFAULT_VALUE,
    LAYER_CLIM_NEGATIVE_DEFAULT_VALUE,
    LAYER_CLIM_PERCENTILE_DEFAULT_VALUE,
    LAYER_COLOR_DEFAULT_VALUE,
    LAYER_COLOR_NEGATIVE_DEFAULT_VALUE,
    LAYER_CMAP_NEGATIVE_DEFAULT_VALUE,
    LAYER_NAN_OVERRIDE_DEFAULT_VALUE,
)
from ..util import (
    premultiply_alpha,
    scalar_percentile,
    source_over,
    unmultiply_alpha,
    LinearScalarMapper,
)

BLEND_MODES = {
    'source_over': source_over,
}


@dataclasses.dataclass(frozen=True)
class _LayerBase:
    """Base class for layers."""
    name: Optional[str]
    # long_name: Optional[str] = None
    color: Optional[Any] = LAYER_COLOR_DEFAULT_VALUE
    color_negative: Optional[Any] = LAYER_COLOR_NEGATIVE_DEFAULT_VALUE
    cmap: Optional[Any] = DEFAULT_CMAP
    cmap_negative: Optional[Any] = LAYER_CMAP_NEGATIVE_DEFAULT_VALUE
    clim: Optional[Tuple[float, float]] = LAYER_CLIM_DEFAULT_VALUE
    clim_negative: Optional[Tuple[float, float]] = (
        LAYER_CLIM_NEGATIVE_DEFAULT_VALUE
    )
    clim_percentile: bool = LAYER_CLIM_PERCENTILE_DEFAULT_VALUE
    below_color: Optional[Any] = LAYER_BELOW_COLOR_DEFAULT_VALUE
    alpha: Union[float, str] = LAYER_ALPHA_DEFAULT_VALUE
    alpha_negative: Optional[float] = LAYER_ALPHA_NEGATIVE_DEFAULT_VALUE
    alim: Optional[Tuple[float, float]] = LAYER_ALIM_DEFAULT_VALUE
    alim_negative: Optional[Tuple[float, float]] = (
        LAYER_ALIM_NEGATIVE_DEFAULT_VALUE
    )
    alim_percentile: bool = LAYER_ALIM_PERCENTILE_DEFAULT_VALUE
    amap: Optional[callable] = LAYER_AMAP_DEFAULT_VALUE
    amap_negative: Optional[callable] = LAYER_AMAP_NEGATIVE_DEFAULT_VALUE
    nan_override: Optional[Tuple[float, float, float, float]] = (
        LAYER_NAN_OVERRIDE_DEFAULT_VALUE
    )
    hide_subthreshold: bool = False
    # style: Optional[Mapping[str, Any]] = None
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
    # hide_subthreshold: bool = True


@dataclasses.dataclass
class SubgeometryParameters(MappingABC):
    """
    Addressable container for parameters specific to some subgeometry, most
    likely a cortical hemisphere.
    """
    params: Mapping[str, Mapping[str, Any]]

    def __init__(self, **params):
        self.params = params

    def __len__(self):
        return len(self.params)
    
    def __iter__(self):
        return iter(self.params)
    
    def __getitem__(self, key):
        return self.params[key]

    def get(self, geom: str, param: str):
        return self.params[geom][param]


def _property_vector(
    scalar_array: Tensor,
    lim: Optional[Tuple[float, float]] = None,
    percentile: bool = False,
    mapper: Optional[Union[callable, Tuple[float, float]]] = None,
) -> Tuple[Tensor, Optional[callable]]:
    if lim is not None:
        if percentile:
            lim = scalar_percentile(scalar_array, lim)
        vmin, vmax = lim
    else:
        vmin, vmax = np.nanmin(scalar_array), np.nanmax(scalar_array)
    scalar_array = np.clip(scalar_array, vmin, vmax)
    if mapper is not None:
        if isinstance(mapper, Sequence):
            mapper = LinearScalarMapper(norm=mapper)
        scalar_array = mapper(
            scalar_array,
            vmin=vmin,
            vmax=vmax,
        )
    return scalar_array, mapper


def _rgba_impl(
    scalars: Tensor,
    alpha_scalars: Optional[Tensor] = None,
    color: Optional[str] = None,
    clim: Optional[Tuple[float, float]] = None,
    clim_percentile: bool = False,
    cmap: Any = 'viridis',
    below_color: Optional[str] = None,
    alpha: Optional[float] = None,
    alim: Optional[Tuple[float, float]] = None,
    alim_percentile: bool = False,
    amap: Optional[callable] = None,
    nan_override: Optional[
        Tuple[float, float, float, float]
    ] = (0, 0, 0, 0),
    hide_subthreshold: bool = False,
    scalar_bar_builder: Optional[ScalarBarBuilder] = None,
) -> Tuple[Tensor, Optional[ScalarBarBuilder]]:
    if color is not None:
        rgba = np.tile(colors.to_rgba(color), (len(scalars), 1))
        vmin, vmax, mapper = None, None, None
    else:
        if clim_percentile:
            clim = scalar_percentile(scalars, clim)
        if clim is not None:
            vmin, vmax = clim
        else:
            vmin, vmax = np.nanmin(scalars), np.nanmax(scalars)
            hide_subthreshold = False
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        rgba = mapper.to_rgba(scalars)
    if alpha_scalars is not None:
        alpha_scalars, _ = _property_vector(
            alpha_scalars,
            lim=alim,
            percentile=alim_percentile,
            mapper=amap if amap is not None else (0, 1),
        )
        rgba[:, 3] = alpha_scalars
    elif alpha is not None:
        rgba[:, 3] *= alpha
    if nan_override is not None:
        rgba = np.where(
            np.isnan(scalars[..., None]),
            np.asarray(nan_override),
            rgba,
        )
    if vmin is not None:
        if below_color is not None:
            rgba[scalars < vmin] = colors.to_rgba(below_color)
        elif hide_subthreshold:
            # Set alpha to 0 for sub-threshold values
            rgba[scalars < vmin, 3] = 0
    if mapper is not None:
        scalar_bar_builder = ScalarBarBuilder(**{
            **scalar_bar_builder,
            **{
                'mapper': mapper,
                'below_color': below_color,
            },
        })
    else:
        scalar_bar_builder = None
    return rgba, scalar_bar_builder


def scalars_to_rgba(
    scalars: Optional[Tensor] = None,
    alpha_scalars: Optional[Tensor] = None,
    color: Optional[str] = LAYER_COLOR_DEFAULT_VALUE,
    color_negative: Optional[str] = LAYER_COLOR_NEGATIVE_DEFAULT_VALUE,
    clim: Optional[Tuple[float, float]] = LAYER_CLIM_DEFAULT_VALUE,
    clim_negative: Optional[Tuple[float, float]] = (
        LAYER_CLIM_NEGATIVE_DEFAULT_VALUE
    ),
    clim_percentile: bool = LAYER_CLIM_PERCENTILE_DEFAULT_VALUE,
    cmap: Optional[str] = DEFAULT_CMAP,
    cmap_negative: Optional[str] = LAYER_CMAP_NEGATIVE_DEFAULT_VALUE,
    below_color: Optional[str] = LAYER_BELOW_COLOR_DEFAULT_VALUE,
    alpha: Optional[float] = LAYER_ALPHA_DEFAULT_VALUE,
    alpha_negative: Optional[float] = LAYER_ALPHA_NEGATIVE_DEFAULT_VALUE,
    alim: Optional[Tuple[float, float]] = LAYER_ALIM_DEFAULT_VALUE,
    alim_negative: Optional[Tuple[float, float]] = (
        LAYER_ALIM_NEGATIVE_DEFAULT_VALUE
    ),
    alim_percentile: bool = LAYER_ALIM_PERCENTILE_DEFAULT_VALUE,
    amap: Optional[callable] = LAYER_AMAP_DEFAULT_VALUE,
    amap_negative: Optional[callable] = LAYER_AMAP_NEGATIVE_DEFAULT_VALUE,
    nan_override: Optional[
        Tuple[float, float, float, float]
    ] = LAYER_NAN_OVERRIDE_DEFAULT_VALUE,
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
    scalar_bar_builder_negative = None
    if cmap_negative is not None or color_negative is not None:
        if color_negative is None:
            color_negative = color
        if clim_negative is None:
            clim_negative = clim
        if alpha_negative is None:
            alpha_negative = alpha
        if amap_negative is None:
            amap_negative = amap
        if alim_negative is None:
            alim_negative = alim
        scalars_negative = -scalars.copy()
        scalars = scalars.copy()
        neg_idx = (scalars_negative > 0)
        scalars_negative[~neg_idx] = 0
        scalars[neg_idx] = 0
        if scalar_bar_builder is not None:
            scalar_bar_builder_negative = ScalarBarBuilder(**{
                **scalar_bar_builder,
                **{'name_suffix': ' (—)'}
            })
            scalar_bar_builder = ScalarBarBuilder(**{
                **scalar_bar_builder,
                **{'name_suffix': ' (+)'}
            })
        rgba_neg, scalar_bar_builder_negative = _rgba_impl(
            scalars=scalars_negative,
            alpha_scalars=alpha_scalars,
            color=color_negative,
            clim=clim_negative,
            clim_percentile=clim_percentile,
            cmap=cmap_negative,
            below_color=below_color,
            alpha=alpha_negative,
            alim=alim_negative,
            alim_percentile=alim_percentile,
            amap=amap_negative,
            nan_override=nan_override,
            hide_subthreshold=hide_subthreshold,
            scalar_bar_builder=scalar_bar_builder_negative,
        )

    rgba, scalar_bar_builder = _rgba_impl(
        scalars=scalars,
        alpha_scalars=alpha_scalars,
        color=color,
        clim=clim,
        clim_percentile=clim_percentile,
        cmap=cmap,
        below_color=below_color,
        alpha=alpha,
        alim=alim,
        alim_percentile=alim_percentile,
        amap=amap,
        nan_override=nan_override,
        hide_subthreshold=hide_subthreshold,
        scalar_bar_builder=scalar_bar_builder,
    )
    if cmap_negative is not None:
        rgba[neg_idx] = rgba_neg[neg_idx]
    return rgba, (scalar_bar_builder, scalar_bar_builder_negative)


def layer_rgba(
    layer: Layer,
    scalar_array: Tensor,
    alpha_scalar_array: Optional[Tensor] = None,
) -> Tuple[Tensor, Sequence[ScalarBarBuilder]]:
    cmap = layer.cmap or DEFAULT_CMAP
    if layer.scalar_bar_style is not None:
        scalar_bar_builder = ScalarBarBuilder(mapper=None, name=layer.name)
    else:
        scalar_bar_builder = None
    rgba, scalar_bar_builders = scalars_to_rgba(
        scalars=scalar_array,
        alpha_scalars=alpha_scalar_array,
        cmap=cmap,
        cmap_negative=layer.cmap_negative,
        clim=layer.clim,
        clim_negative=layer.clim_negative,
        clim_percentile=layer.clim_percentile,
        color=layer.color,
        color_negative=layer.color_negative,
        below_color=layer.below_color,
        alpha=layer.alpha,
        alpha_negative=layer.alpha_negative,
        amap=layer.amap,
        amap_negative=layer.amap_negative,
        alim=layer.alim,
        alim_negative=layer.alim_negative,
        alim_percentile=layer.alim_percentile,
        nan_override=layer.nan_override,
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


def compose_layers(
    geom: Any,
    layers: Sequence[Layer],
    **params,
) -> Tuple[Tensor, str, Sequence[ScalarBarBuilder]]:
    """
    Compose layers into a single RGB(A) array.
    """
    dst, tensors, params = geom.init_composition(layers, **params)
    dst, scalar_bar_builders = layer_rgba(dst, *tensors)
    dst = premultiply_alpha(dst)

    for layer in layers[1:]:
        src, new_builders = geom.layer_to_tensors(layer, **params)
        src = premultiply_alpha(src)
        blend_layers = BLEND_MODES[layer.blend_mode]
        dst = blend_layers(src, dst)
        scalar_bar_builders = scalar_bar_builders + new_builders
    dst = unmultiply_alpha(dst)
    return dst, scalar_bar_builders, params
