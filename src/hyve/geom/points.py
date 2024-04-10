# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Point clouds
~~~~~~~~~~~~
Point cloud geometry data containers and geometric primitives.
"""
import dataclasses
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import pyvista as pv

from .base import Layer, layer_rgba
from ..elements import ScalarBarBuilder
from ..const import (
    Tensor,
    LAYER_ALIM_DEFAULT_VALUE,
    LAYER_ALIM_NEGATIVE_DEFAULT_VALUE,
    LAYER_ALIM_PERCENTILE_DEFAULT_VALUE,
    LAYER_ALPHA_DEFAULT_VALUE,
    LAYER_AMAP_DEFAULT_VALUE,
    LAYER_AMAP_NEGATIVE_DEFAULT_VALUE,
    LAYER_CLIM_NEGATIVE_DEFAULT_VALUE,
    LAYER_CLIM_PERCENTILE_DEFAULT_VALUE,
    LAYER_CMAP_NEGATIVE_DEFAULT_VALUE,
    LAYER_COLOR_DEFAULT_VALUE,
    LAYER_NAN_OVERRIDE_DEFAULT_VALUE,
    POINTS_DEFAULT_STYLE,
    POINTS_SCALARS_BELOW_COLOR_DEFAULT_VALUE,
    POINTS_SCALARS_CLIM_DEFAULT_VALUE,
    POINTS_SCALARS_CMAP_DEFAULT_VALUE,
    POINTS_SCALARS_DEFAULT_VALUE,
    POINTS_SCALARS_LAYERS_DEFAULT_VALUE,
)


@dataclasses.dataclass
class PointData:
    points: pv.PointSet
    point_size: float = 1.0

    def __init__(
        self,
        points: pv.PointSet,
        point_size: float = 1.0,
        data: Optional[Mapping[str, Tensor]] = None,
    ):
        self.points = points
        self.point_size = point_size
        data = data or {}
        for key, value in data.items():
            self.points.point_data[key] = value

    # TODO: Let's use nitransforms for this
    def translate(
        self,
        translation: Sequence[float],
        condition: Optional[callable] = None,
    ) -> 'PointData':
        if not condition:
            points = pv.PointSet(self.points.points + np.array(translation))
            return self.__class__(
                points,
                self.point_size,
                data=self.points.point_data,
            )
        return self.select(condition).translate(translation)

    def select(self, condition: callable) -> 'PointData':
        mask = condition(self.points.points, self.points.point_data)
        return self.mask(mask)

    def mask(
        self,
        mask: Tensor,
        return_complement: bool = False
    ) -> 'PointData':
        if return_complement:
            mask = ~mask
        points = pv.PointSet(self.points.points[mask])
        for name, data in self.points.point_data.items():
            points.point_data[name] = data[mask]
        return self.__class__(points, self.point_size)

    def __add__(self, other: 'PointData') -> 'PointData':
        # Note that there is no coalescing of point data. We should probably
        # underpin this with ``scipy.sparse`` or something equivalent.
        # TODO: This is currently unsafe for point data with different keys.
        #       It's fine for our purposes since this is only accessed when we
        #       recombine point data that were split by a condition. However,
        #       we should probably raise a more appropriate error if nothing
        #       else.
        points = pv.PointSet(np.concatenate([
            self.points.points, other.points.points
        ]))
        for name, data in self.points.point_data.items():
            points.point_data[name] = np.concatenate([
                data, other.points.point_data[name]
            ])
        return self.__class__(points, self.point_size)


class PointDataCollection:
    def __init__(
        self,
        point_datasets: Optional[Sequence[PointData]] = None,
    ):
        self.point_datasets = list(point_datasets) or []

    def add_point_dataset(self, point_dataset: PointData):
        return self.__class__(self.point_datasets + [point_dataset])

    def get_dataset(
        self,
        key: str,
        return_all: bool = False,
        strict: bool = True
    ) -> PointData:
        matches = [
            (ds.points.point_data.get(key, None), i)
            for i, ds in enumerate(self.point_datasets)
        ]
        if all(scalars is None for scalars, _ in matches):
            raise KeyError(f'No point data with key {key}')
        indices = [index for scalars, index in matches if scalars is not None]
        if not return_all:
            if len(indices) > 1 and strict:
                raise KeyError(f'Multiple point data with key {key}')
            dataset = self[indices[0]]
            points = pv.PointSet(dataset.points.points)
            points.point_data[key] = dataset.points.point_data[key]
            point_size = dataset.point_size
        else:
            datasets = [self[i] for i in indices]
            points = pv.PointSet(np.concatenate([
                ds.points.points for ds in datasets
            ]))
            points.point_data[key] = np.concatenate([
                ds.points.point_data[key] for ds in datasets
            ])
            point_size = np.min([ds.point_size for ds in datasets])
        return PointData(
            points=points,
            point_size=point_size,
        )

    def translate(
        self,
        translation: Sequence[float],
        condition: Optional[callable] = None,
    ):
        return self.__class__([
            ds.translate(translation, condition=condition)
            for ds in self.point_datasets
        ])

    def present_in_hemisphere(self, hemi: str, key: str) -> bool:
        # Currently, we don't track hemisphere information in the point data
        # structure. If we begin to do so, we should update this method.
        try:
            self.get_dataset(key)
            return True
        except KeyError:
            return False

    def paint(
        self,
        plotter: pv.Plotter,
        layers: Sequence[Layer],
        points_alpha: Optional[float] = 1.0,
        style: Union[Mapping, Literal['__default__']] = '__default__',
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
        layers = [
            layer
            if layer.color is not None
            else dataclasses.replace(
                layer,
                color=layer.name,
            )
            for layer in layers
        ]
        if style == '__default__':
            base_style = POINTS_DEFAULT_STYLE
        else:
            base_style = {**POINTS_DEFAULT_STYLE, **style}
        scalar_bar_builders = ()
        for layer in layers:
            dataset = self.get_dataset(layer.name)
            color_array = dataset.points.point_data[layer.color]
            alpha_array = dataset.points.point_data.get('alpha', None)
            layer = dataclasses.replace(layer, color=None)
            rgba, new_builders = layer_rgba(
                layer, color_array, alpha_array
            )
            style = {**{'point_size': dataset.point_size}, **base_style}
            if (points_alpha is None):
                rgba = rgba[:, :3]
            plotter.add_points(
                points=dataset.points.points,
                scalars=rgba,
                opacity=points_alpha,
                rgb=True,
                **style,
                copy_mesh=copy_actors,
            )
            scalar_bar_builders += new_builders
        return plotter, scalar_bar_builders

    def __getitem__(self, key: int):
        return self.point_datasets[key]

    def __len__(self):
        return len(self.point_datasets)

    def __iter__(self):
        return iter(self.point_datasets)

    def __str__(self):
        return f'PointDataCollection({self.point_datasets})'

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return self.__class__(self.point_datasets + other.point_datasets)

    def __radd__(self, other):
        return self.__class__(other.point_datasets + self.point_datasets)


def plot_points_f(
    plotter: pv.Plotter,
    scalar_bar_builders: Sequence[ScalarBarBuilder],
    copy_actors: bool = False,
    *,
    points: Optional[PointDataCollection] = None,
    points_alpha: float = 1.0,
    points_style: Union[Mapping, Literal['__default__']] = '__default__',
    points_scalars: Optional[str] = POINTS_SCALARS_DEFAULT_VALUE,
    points_scalars_color: Optional[str] = LAYER_COLOR_DEFAULT_VALUE,
    points_scalars_cmap: Any = POINTS_SCALARS_CMAP_DEFAULT_VALUE,
    points_scalars_cmap_negative: Optional[
        Any
    ] = LAYER_CMAP_NEGATIVE_DEFAULT_VALUE,
    points_scalars_clim: Optional[
        Tuple[float, float]
    ] = POINTS_SCALARS_CLIM_DEFAULT_VALUE,
    points_scalars_clim_negative: Optional[
        Tuple[float, float]
    ] = LAYER_CLIM_NEGATIVE_DEFAULT_VALUE,
    points_scalars_clim_percentile: bool = (
        LAYER_CLIM_PERCENTILE_DEFAULT_VALUE
    ),
    points_scalars_alpha: float = LAYER_ALPHA_DEFAULT_VALUE,
    points_scalars_amap: Union[
        callable, Tuple[float, float]
    ] = LAYER_AMAP_DEFAULT_VALUE,
    points_scalars_amap_negative: Union[
        callable, Tuple[float, float]
    ] = LAYER_AMAP_NEGATIVE_DEFAULT_VALUE,
    points_scalars_alim: Optional[
        Tuple[float, float]
    ] = LAYER_ALIM_DEFAULT_VALUE,
    points_scalars_alim_negative: Optional[
        Tuple[float, float]
    ] = LAYER_ALIM_NEGATIVE_DEFAULT_VALUE,
    points_scalars_alim_percentile: bool = (
        LAYER_ALIM_PERCENTILE_DEFAULT_VALUE
    ),
    points_scalars_nan_override: Any = LAYER_NAN_OVERRIDE_DEFAULT_VALUE,
    points_scalars_below_color: str = (
        POINTS_SCALARS_BELOW_COLOR_DEFAULT_VALUE
    ),
    points_scalars_layers: Optional[Sequence[Layer]] = (
        POINTS_SCALARS_LAYERS_DEFAULT_VALUE
    ),
) -> Tuple[pv.Plotter, Sequence[ScalarBarBuilder]]:
    # Sometimes by construction points_scalars is an empty tuple or list,
    # which causes problems later on. So we convert it to None if it's empty.
    if points_scalars is not None:
        points_scalars = None if len(points_scalars) == 0 else points_scalars

    if points is not None:
        if points_scalars_layers is None:
            points_scalars_layers = []
        if points_scalars is not None:
            base_layer = Layer(
                name=points_scalars,
                color=points_scalars_color or points_scalars,
                cmap=points_scalars_cmap,
                cmap_negative=points_scalars_cmap_negative,
                clim=points_scalars_clim,
                clim_negative=points_scalars_clim_negative,
                clim_percentile=points_scalars_clim_percentile,
                alpha=points_scalars_alpha,
                amap=points_scalars_amap,
                amap_negative=points_scalars_amap_negative,
                alim=points_scalars_alim,
                alim_negative=points_scalars_alim_negative,
                alim_percentile=points_scalars_alim_percentile,
                below_color=points_scalars_below_color,
                nan_override=points_scalars_nan_override,
            )
            points_scalars_layers = [base_layer] + list(points_scalars_layers)
        plotter, new_builders = points.paint(
            plotter=plotter,
            layers=points_scalars_layers,
            points_alpha=points_alpha,
            style=points_style,
            copy_actors=copy_actors,
        )
        scalar_bar_builders = scalar_bar_builders + new_builders
    return plotter, scalar_bar_builders


def plot_points_aux_f(
    metadata: Mapping[str, Sequence[str]],
    *,
    points: Optional[PointDataCollection] = None,
    points_alpha: float = 1.0,
    points_style: Union[Mapping, Literal['__default__']] = '__default__',
    points_scalars: Optional[str] = POINTS_SCALARS_DEFAULT_VALUE,
    points_scalars_color: Optional[str] = LAYER_COLOR_DEFAULT_VALUE,
    points_scalars_cmap: Any = POINTS_SCALARS_CMAP_DEFAULT_VALUE,
    points_scalars_cmap_negative: Optional[
        Any
    ] = LAYER_CMAP_NEGATIVE_DEFAULT_VALUE,
    points_scalars_clim: Optional[
        Tuple[float, float]
    ] = POINTS_SCALARS_CLIM_DEFAULT_VALUE,
    points_scalars_clim_negative: Optional[
        Tuple[float, float]
    ] = LAYER_CLIM_NEGATIVE_DEFAULT_VALUE,
    points_scalars_clim_percentile: bool = (
        LAYER_CLIM_PERCENTILE_DEFAULT_VALUE
    ),
    points_scalars_alpha: float = LAYER_ALPHA_DEFAULT_VALUE,
    points_scalars_amap: Union[
        callable, Tuple[float, float]
    ] = LAYER_AMAP_DEFAULT_VALUE,
    points_scalars_amap_negative: Union[
        callable, Tuple[float, float]
    ] = LAYER_AMAP_NEGATIVE_DEFAULT_VALUE,
    points_scalars_alim: Optional[
        Tuple[float, float]
    ] = LAYER_ALIM_DEFAULT_VALUE,
    points_scalars_alim_negative: Optional[
        Tuple[float, float]
    ] = LAYER_ALIM_NEGATIVE_DEFAULT_VALUE,
    points_scalars_alim_percentile: bool = (
        LAYER_ALIM_PERCENTILE_DEFAULT_VALUE
    ),
    points_scalars_nan_override: Any = LAYER_NAN_OVERRIDE_DEFAULT_VALUE,
    points_scalars_below_color: str = (
        POINTS_SCALARS_BELOW_COLOR_DEFAULT_VALUE
    ),
    points_scalars_layers: Optional[Sequence[Layer]] = (
        POINTS_SCALARS_LAYERS_DEFAULT_VALUE
    ),
) -> Mapping[str, Sequence[str]]:
    metadata['pointsscalars'] = [points_scalars or None]
    if points_scalars_layers is not None:
        layers = '+'.join(layer.name for layer in points_scalars_layers)
        metadata['pointsscalars'] = (
            [f'{metadata["pointsscalars"][0]}+{layers}']
            if metadata['pointsscalars'][0] is not None
            else [layers]
        )
    return metadata
