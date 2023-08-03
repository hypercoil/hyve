# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Plot and report utilities
~~~~~~~~~~~~~~~~~~~~~~~~~
Utilities for plotting and reporting.
"""
import dataclasses
from math import floor
from typing import (
    Any,
    Dict,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import pyvista as pv
from PIL import Image
from pyvista.plotting.helpers import view_vectors

from .const import Tensor


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
        #       recombine point data that was split by a condition. However,
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


@dataclasses.dataclass(frozen=True)
class NetworkData:
    name: str
    coor: Tensor
    nodes: pd.DataFrame
    edges: Optional[pd.DataFrame] = None
    lh_mask: Optional[Tensor] = None

    def translate(
        self,
        translation: Sequence[float],
        condition: Optional[callable] = None,
    ) -> 'PointData':
        if not condition:
            coor = self.coor + np.array(translation)
        else:
            mask = condition(self.coor, self.nodes, self.lh_mask)
            coor = self.coor.copy()
            coor[mask] = coor[mask] + np.array(translation)
        return self.__class__(
            self.name,
            coor,
            self.nodes,
            self.edges,
            self.lh_mask,
        )


class NetworkDataCollection:
    def __init__(
        self,
        network_datasets: Optional[Sequence[NetworkData]] = None
    ):
        self.network_datasets = list(network_datasets) or []

    def add_network_dataset(self, network_dataset: NetworkData):
        return self.__class__(self.network_datasets + [network_dataset])

    def get_dataset(
        self,
        key: str,
        return_all: bool = False,
        strict: bool = True,
    ) -> Union[NetworkData, 'NetworkDataCollection']:
        indices = [
            i for i in range(len(self.network_datasets))
            if self.network_datasets[i].name == key
        ]
        if len(indices) == 0:
            raise KeyError(f'No node data with key {key}')
        if not return_all:
            if len(indices) > 1 and strict:
                raise KeyError(f'Multiple node data with key {key}')
            return self[indices[0]]
        else:
            return self.__class__([self[i] for i in indices])

    def get_node_dataset(
        self,
        key: str,
        return_all: bool = False,
        strict: bool = True,
    ) -> Union[NetworkData, Sequence[NetworkData], 'NetworkDataCollection']:
        return self.get_dataset(
            key,
            return_all=return_all,
            strict=strict,
            search='nodes',
        )

    def get_edge_dataset(
        self,
        key: str,
        return_all: bool = False,
        strict: bool = True,
    ) -> Union[NetworkData, Sequence[NetworkData], 'NetworkDataCollection']:
        return self.get_dataset(
            key,
            return_all=return_all,
            strict=strict,
            search='edges',
        )

    def translate(
        self,
        translation: Sequence[float],
        condition: Optional[callable] = None,
    ):
        return self.__class__([
            ds.translate(translation, condition=condition)
            for ds in self.network_datasets
        ])

    def __getitem__(self, key: int):
        return self.network_datasets[key]

    def __len__(self):
        return len(self.network_datasets)

    def __iter__(self):
        return iter(self.network_datasets)

    def __str__(self):
        return f'NetworkDataCollection({self.network_datasets})'

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return self.__class__(self.network_datasets + other.network_datasets)

    def __radd__(self, other):
        return self.__class__(other.network_datasets + self.network_datasets)


def cortex_theme() -> Any:
    """
    Return a theme for the pyvista plotter for use with the cortical surface
    plotter. In practice, we currently don't use this because PyVista doesn't
    always handle transparency the way that we would like, but we keep it
    here in case we want to use it in the future.
    """
    cortex_theme = pv.themes.DocumentTheme()
    cortex_theme.transparent_background = True
    return cortex_theme


def half_width(
    p: pv.Plotter,
    slack: float = 1.05,
) -> Tuple[float, float, float]:
    """
    Return the half-width of the plotter's bounds, multiplied by a slack
    factor.

    We use this to set the position of the camera when we're using
    ``cortex_cameras`` and we receive a string corresponding to an anatomical
    direction (e.g. "dorsal", "ventral", etc.) as the ``position`` argument.

    The slack factor is used to ensure that the camera is not placed exactly
    on the edge of the plotter bounds, which can cause clipping of the
    surface.
    """
    return (
        (p.bounds[1] - p.bounds[0]) / 2 * slack,
        (p.bounds[3] - p.bounds[2]) / 2 * slack,
        (p.bounds[5] - p.bounds[4]) / 2 * slack,
    )


def _relabel_parcels_hemi(
    data: np.ndarray,
    null_value: int = 0,
) -> np.ndarray:
    data = data.astype(np.int32)
    data[data == null_value] = -1
    _, data = np.unique(data, return_inverse=True)
    data[data == -1] = null_value - 1
    return data + 1


def relabel_parcels(
    left_data: np.ndarray,
    right_data: np.ndarray,
    null_value: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Relabel the parcels in the left and right hemisphere data arrays so that
    they are contiguous and start at 1.

    Parameters
    ----------
    left_data : np.ndarray
        Array of parcel values for the left hemisphere.
    right_data : np.ndarray
        Array of parcel values for the right hemisphere.
    null_value : int, optional
        The value to use for null (or background) values in the data arrays.
        By default, this is 0.

    Returns
    -------
    left_data : np.ndarray
        Array of parcel values for the left hemisphere, with contiguous parcel
        values starting at 1.
    right_data : np.ndarray
        Array of parcel values for the right hemisphere, with contiguous parcel
        values starting at the maximum value in left_data, plus 1.

    Notes
    -----
    What utter wickedness is this?! This function was written almost entirely
    by GitHub Copilot.
    """
    if left_data.squeeze().ndim == 2:
        return left_data, right_data
    # Relabel the parcels in the left hemisphere
    left_data = _relabel_parcels_hemi(left_data, null_value=null_value)
    offset = np.max(left_data)

    # Relabel the parcels in the right hemisphere
    right_data = _relabel_parcels_hemi(right_data, null_value=null_value)
    right_data[right_data != null_value] += offset

    return left_data, right_data


def auto_focus(
    vector: Sequence[float],
    plotter: pv.Plotter,
    slack: float = 1.05,
    focal_point: Optional[Sequence[float]] = None,
) -> Tuple[float, float, float]:
    vector = np.asarray(vector)
    if focal_point is None:
        focal_point = plotter.center
    hw = half_width(plotter, slack=slack)
    scalar = np.nanmin(hw / np.abs(vector))
    vector = vector * scalar + focal_point
    return vector, focal_point


def set_default_views(
    hemisphere: str,
) -> Sequence[str]:
    common = ('dorsal', 'ventral', 'anterior', 'posterior')
    if hemisphere == 'both':
        views = ('left', 'right') + common
    else:
        views = ('lateral', 'medial') + common
    return views


def cortex_view_dict() -> Dict[str, Tuple[Sequence[float], Sequence[float]]]:
    """
    Return a dict containing tuples of (vector, viewup) pairs for each
    hemisphere. The vector is the position of the camera, and the
    viewup is the direction of the "up" vector in the camera frame.
    """
    common = {
        'dorsal': ((0, 0, 1), (1, 0, 0)),
        'ventral': ((0, 0, -1), (-1, 0, 0)),
        'anterior': ((0, 1, 0), (0, 0, 1)),
        'posterior': ((0, -1, 0), (0, 0, 1)),
    }
    return {
        'left': {
            'lateral': ((-1, 0, 0), (0, 0, 1)),
            'medial': ((1, 0, 0), (0, 0, 1)),
            **common,
        },
        'right': {
            'lateral': ((1, 0, 0), (0, 0, 1)),
            'medial': ((-1, 0, 0), (0, 0, 1)),
            **common,
        },
        'both': {
            'left': ((-1, 0, 0), (0, 0, 1)),
            'right': ((1, 0, 0), (0, 0, 1)),
            **common,
        },
    }


def cortex_cameras(
    position: Union[str, Sequence[Tuple[float, float, float]]],
    plotter: pv.Plotter,
    negative: bool = False,
    hemisphere: Optional[Literal['left', 'right']] = None,
) -> Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]:
    """
    Return a tuple of (position, focal_point, view_up) for the camera. This
    function accepts a string corresponding to an anatomical direction (e.g.
    "dorsal", "ventral", etc.) as the ``position`` argument, and returns the
    corresponding camera position, focal point, and view up vector.
    """
    hemisphere = hemisphere or 'both'
    # if not isinstance(hemisphere, str):
    #     if len(hemisphere) == 1:
    #         hemisphere = hemisphere[0]
    #     else:
    #         hemisphere = 'both'
    if isinstance(position, str):
        try:
            # TODO: I'm a little bit concerned that ``view_vectors`` is not
            #       part of the public API. We should probably find a better
            #       way to do this.
            position = view_vectors(view=position, negative=negative)
        except ValueError as e:
            if isinstance(hemisphere, str):
                try:
                    vector, view_up = cortex_view_dict()[hemisphere][position]
                    vector, focal_point = auto_focus(vector, plotter)
                    return (vector, focal_point, view_up)
                except KeyError:
                    raise e
            else:
                raise e
    return position


def scale_image_preserve_aspect_ratio(
    img: Image.Image,
    target_size: Tuple[int, int],
) -> Image.Image:
    width, height = img.size
    target_width, target_height = target_size
    width_ratio = target_width / width
    height_ratio = target_height / height
    ratio = min(width_ratio, height_ratio)
    new_width = floor(width * ratio)
    new_height = floor(height * ratio)
    return img.resize((new_width, new_height))


def robust_clim(
    data: Tensor,
    percent: float = 5.0,
    bgval: Optional[float] = 0.0,
) -> Tuple[float, float]:
    if bgval is not None:
        data = data[~np.isclose(data, bgval)]
    return (
        np.nanpercentile(data, percent),
        np.nanpercentile(data, 100 - percent),
    )


def plot_to_display(
    p: pv.Plotter,
    cpos: Optional[Sequence[Sequence[float]]] = 'yz',
) -> None:
    p.show(cpos=cpos)


def format_position_as_string(
    position: Union[str, Sequence[Tuple[float, float, float]]],
    precision: int = 2,
    delimiter: str = 'x',
) -> str:
    def _fmt_field(field: float) -> str:
        return delimiter.join(
            str(round(v, precision))
            if v >= 0
            else f'neg{abs(round(v, precision))}'
            for v in field
        )

    if isinstance(position, str):
        return f'{position}'
    elif isinstance(position[0], float) or isinstance(position[0], int):
        return f'vector{_fmt_field(position)}'
    else:
        return (
            f'vector{_fmt_field(position[0])}AND'
            f'focus{_fmt_field(position[1])}AND'
            f'viewup{_fmt_field(position[2])}'
        )


def filter_node_data(
    val: np.ndarray,
    name: str = 'node',
    threshold: Optional[Union[float, int]] = 0.0,
    percent_threshold: bool = False,
    topk_threshold: bool = False,
    absolute: bool = True,
    node_selection: Optional[np.ndarray] = None,
    incident_edge_selection: Optional[np.ndarray] = None,
    removed_val: Optional[float] = None,
    surviving_val: Optional[float] = 1.0,
) -> pd.DataFrame:
    node_incl = np.ones_like(val, dtype=bool)

    sgn = np.sign(val)
    if absolute:
        val = np.abs(val)
    if node_selection is not None:
        node_incl[~node_selection] = 0
    if incident_edge_selection is not None:
        node_incl[~incident_edge_selection.any(axis=-1)] = 0
    if topk_threshold:
        indices = np.argpartition(-val, int(threshold))
        node_incl[indices[int(threshold) :]] = 0
    elif percent_threshold:
        node_incl[val < np.percentile(val[node_incl], 100 * threshold)] = 0
    elif threshold is not None:
        node_incl[val < threshold] = 0

    if removed_val is not None:
        val[~node_incl] = removed_val
        if surviving_val is not None:
            val[node_incl] = surviving_val
        index = np.arange(val.shape[0])
    else:
        val = val[node_incl]
        sgn = sgn[node_incl]
        index = np.where(node_incl)[0]

    return pd.DataFrame(
        {
            f'{name}_val': val,
            f'{name}_sgn': sgn,
        },
        index=pd.Index(index + 1, name=name),
    )


def filter_adjacency_data(
    adj: np.ndarray,
    name: str = 'edge',
    threshold: Union[float, int] = 0.0,
    percent_threshold: bool = False,
    topk_threshold_nodewise: bool = False,
    absolute: bool = True,
    incident_node_selection: Optional[np.ndarray] = None,
    connected_node_selection: Optional[np.ndarray] = None,
    edge_selection: Optional[np.ndarray] = None,
    removed_val: Optional[float] = None,
    surviving_val: Optional[float] = 1.0,
    emit_degree: Union[bool, Literal['abs', '+', '-']] = False,
    emit_incident_nodes: Union[bool, tuple] = False,
) -> pd.DataFrame:
    adj_incl = np.ones_like(adj, dtype=bool)

    sgn = np.sign(adj)
    if absolute:
        adj = np.abs(adj)
    if incident_node_selection is not None:
        adj_incl[~incident_node_selection, :] = 0
    if connected_node_selection is not None:
        adj_incl[~connected_node_selection, :] = 0
        adj_incl[:, ~connected_node_selection] = 0
    if edge_selection is not None:
        adj_incl[~edge_selection] = 0
    if topk_threshold_nodewise:
        indices = np.argpartition(-adj, int(threshold), axis=-1)
        indices = indices[..., int(threshold) :]
        adj_incl[
            np.arange(adj.shape[0], dtype=int).reshape(-1, 1), indices
        ] = 0
    elif percent_threshold:
        adj_incl[adj < np.percentile(adj[adj_incl], 100 * threshold)] = 0
    else:
        adj_incl[adj < threshold] = 0

    degree = None
    if emit_degree == 'abs':
        degree = np.abs(adj).sum(axis=0)
    elif emit_degree == '+':
        degree = np.maximum(adj, 0).sum(axis=0)
    elif emit_degree == '-':
        degree = -np.minimum(adj, 0).sum(axis=0)
    elif emit_degree:
        degree = adj.sum(axis=0)

    indices_incl = np.triu_indices(adj.shape[0], k=1)
    adj_incl = adj_incl | adj_incl.T

    incidents = None
    if emit_incident_nodes:
        incidents = adj_incl.any(axis=0)
        if isinstance(emit_incident_nodes, tuple):
            exc, inc = emit_incident_nodes
            incidents = np.where(incidents, inc, exc)

    if removed_val is not None:
        adj[~adj_incl] = removed_val
        if surviving_val is not None:
            adj[adj_incl] = surviving_val
    else:
        adj_incl = adj_incl[indices_incl]
        indices_incl = tuple(i[adj_incl] for i in indices_incl)
    adj = adj[indices_incl]
    sgn = sgn[indices_incl]

    indices_incl = [i + 1 for i in indices_incl]
    edge_values = pd.DataFrame(
        {
            f'{name}_val': adj,
            f'{name}_sgn': sgn,
        },
        index=pd.MultiIndex.from_arrays(indices_incl, names=['src', 'dst']),
    )

    if degree is not None:
        degree = pd.DataFrame(
            degree,
            index=range(1, degree.shape[0] + 1),
            columns=(f'{name}_degree',),
        )
        if incidents is None:
            return edge_values, degree
    if incidents is not None:
        incidents = pd.DataFrame(
            incidents,
            index=range(1, incidents.shape[0] + 1),
            columns=(f'{name}_incidents',),
        )
        if degree is None:
            return edge_values, incidents
        df = degree.join(incidents, how='outer')
        return edge_values, df
    return edge_values


def premultiply_alpha(
    input: Tensor,
) -> Tensor:
    """Premultiply alpha channel of RGBA image."""
    return np.concatenate(
        (input[..., :3] * input[..., 3:], input[..., 3:]),
        axis=-1,
    )


def unmultiply_alpha(
    input: Tensor,
) -> Tensor:
    """Unmultiply alpha channel of RGBA image."""
    out = np.concatenate(
        (input[..., :3] / input[..., 3:], input[..., 3:]),
        axis=-1,
    )
    return np.where(np.isnan(out), 0.0, out)


def source_over(
    src: Tensor,
    dst: Tensor,
) -> Tensor:
    """
    Alpha composite two RGBA images using the source-over blend mode.
    Assumes premultiplied alpha.
    """
    return src + dst * (1.0 - src[..., 3:])
