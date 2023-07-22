# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Plot and report utilities
~~~~~~~~~~~~~~~~~~~~~~~~~
Utilities for plotting and reporting.
"""
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyvista as pv
from pyvista.plotting.helpers import view_vectors

from .const import Tensor


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
    data[data == -1] = null_value
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
        index=pd.Index(index, name='node'),
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
