# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Primitive functional atoms
~~~~~~~~~~~~~~~~~~~~~~~~~~
Atomic functional primitives for building more complex functions.
"""
from io import StringIO
from math import ceil
from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import nibabel as nb
import numpy as np
import pandas as pd
import pyvista as pv
from conveyant import (
    Composition,
    Primitive,
    ichain,
)
from conveyant import (
    FunctionWrapper as F,
)
from conveyant.compositors import (
    _dict_to_seq,
    direct_compositor,
    reversed_args_compositor,
)
from conveyant.replicate import _flatten, _flatten_to_depth, replicate
from matplotlib.colors import ListedColormap
from PIL import Image

from .const import (
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
    SURF_SCALARS_BELOW_COLOR_DEFAULT_VALUE,
    SURF_SCALARS_CLIM_DEFAULT_VALUE,
    SURF_SCALARS_CMAP_DEFAULT_VALUE,
    SURF_SCALARS_DEFAULT_VALUE,
    SURF_SCALARS_LAYERS_DEFAULT_VALUE,
    Tensor,
)
from .layout import AnnotatedLayout, CellLayout, grid
from .plot import (
    EdgeLayer,
    Layer,
    NodeLayer,
    _get_hemisphere_parameters,
    _null_auxwriter,
    _null_op,
    plotted_entities,
    unified_plotter,
)
from .surf import CortexTriSurface
from .util import (
    NetworkData,
    NetworkDataCollection,
    PointData,
    PointDataCollection,
    auto_focus,
    cortex_cameras,
    filter_adjacency_data,
    filter_node_data,
    format_position_as_string,
    scale_image_preserve_aspect_ratio,
    set_default_views,
)


def surf_from_archive_f(
    template: str,
    load_mask: bool,
    projections: Sequence[str],
    archives: Mapping[str, Callable],
) -> Tuple[CortexTriSurface, Sequence[str]]:
    for archive, constructor in archives.items():
        try:
            surf = constructor(
                template=template,
                load_mask=load_mask,
                projections=projections,
            )
            return surf, projections
        except Exception:
            continue
    raise ValueError(
        f'Could not load {template} with projections {projections} '
        f'from any of {tuple(archives.keys())}.'
    )


def surf_scalars_from_cifti_f(
    surf: CortexTriSurface,
    scalars: str,
    cifti: nb.Cifti2Image,
    surf_scalars: Sequence[str] = (),
    is_masked: bool = True,
    apply_mask: bool = False,
    null_value: Optional[float] = 0.0,
    select: Optional[Sequence[int]] = None,
    exclude: Optional[Sequence[int]] = None,
    allow_multihemisphere: bool = True,
    coerce_to_scalar: bool = True,
    plot: bool = True,
) -> Tuple[CortexTriSurface, Sequence[str]]:
    scalar_names = surf.add_cifti_dataset(
        name=scalars,
        cifti=cifti,
        is_masked=is_masked,
        apply_mask=apply_mask,
        null_value=null_value,
        select=select,
        exclude=exclude,
        allow_multihemisphere=allow_multihemisphere,
        coerce_to_scalar=coerce_to_scalar,
    )
    if plot:
        surf_scalars = tuple(list(surf_scalars) + list(scalar_names))
    return surf, surf_scalars


def surf_scalars_from_gifti_f(
    surf: CortexTriSurface,
    scalars: str,
    left_gifti: Optional[nb.gifti.GiftiImage] = None,
    right_gifti: Optional[nb.gifti.GiftiImage] = None,
    surf_scalars: Sequence[str] = (),
    is_masked: bool = True,
    apply_mask: bool = False,
    null_value: Optional[float] = 0.0,
    select: Optional[Sequence[int]] = None,
    exclude: Optional[Sequence[int]] = None,
    allow_multihemisphere: bool = True,
    coerce_to_scalar: bool = True,
    plot: bool = True,
) -> Tuple[CortexTriSurface, Sequence[str]]:
    scalar_names = surf.add_gifti_dataset(
        name=scalars,
        left_gifti=left_gifti,
        right_gifti=right_gifti,
        is_masked=is_masked,
        apply_mask=apply_mask,
        null_value=null_value,
        select=select,
        exclude=exclude,
        allow_multihemisphere=allow_multihemisphere,
        coerce_to_scalar=coerce_to_scalar,
    )
    if plot:
        surf_scalars = tuple(list(surf_scalars) + list(scalar_names))
    return surf, surf_scalars


def points_scalars_from_nifti_f(
    scalars: str,
    nifti: nb.Nifti1Image,
    null_value: Optional[float] = 0.0,
    point_size: Optional[float] = None,
    points: Optional[PointDataCollection] = None,
    points_scalars: Sequence[str] = (),
    plot: bool = True,
) -> Tuple[PointDataCollection, str]:
    if not isinstance(nifti, nb.Nifti1Image):
        nifti = nb.load(nifti)
    vol = nifti.get_fdata()
    if null_value is None:
        null_value = np.nan
    loc = np.where(vol != null_value)

    vol_scalars = vol[loc]
    vol_coor = np.stack(loc)
    vol_coor = (nifti.affine @ np.concatenate(
        (vol_coor, np.ones((1, vol_coor.shape[-1])))
    ))[:3].T
    if point_size is None:
        vol_voxdim = nifti.header.get_zooms()
        point_size = np.min(vol_voxdim[:3])
    points_data = PointData(
        pv.PointSet(vol_coor),
        data={scalars: vol_scalars},
        point_size=point_size,
    )
    if points:
        points = points + PointDataCollection([points_data])
    else:
        points = PointDataCollection([points_data])
    if plot:
        points_scalars = tuple(list(points_scalars) + [scalars])
    return points, points_scalars


def points_scalars_from_array_f(
    scalars: str,
    coor: Tensor,
    values: Tensor,
    point_size: float = 1.0,
    points: Optional[PointDataCollection] = None,
    points_scalars: Sequence[str] = (),
    plot: bool = True,
) -> Tuple[PointDataCollection, str]:
    points_data = PointData(
        pv.PointSet(coor),
        data={scalars: values},
        point_size=point_size,
    )
    if points:
        points = points + PointDataCollection([points_data])
    else:
        points = PointDataCollection([points_data])
    if plot:
        points_scalars = tuple(list(points_scalars) + [scalars])
    return points, points_scalars


def surf_scalars_from_array_f(
    surf: CortexTriSurface,
    scalars: str,
    surf_scalars: Sequence[str] = (),
    array: Optional[Tensor] = None,
    left_array: Optional[Tensor] = None,
    right_array: Optional[Tensor] = None,
    left_slice: Optional[slice] = None,
    right_slice: Optional[slice] = None,
    default_slices: bool = True,
    is_masked: bool = False,
    apply_mask: bool = True,
    null_value: Optional[float] = 0.0,
    select: Optional[Sequence[int]] = None,
    exclude: Optional[Sequence[int]] = None,
    allow_multihemisphere: bool = True,
    coerce_to_scalar: bool = True,
    plot: bool = False,
) -> Tuple[CortexTriSurface, Sequence[str]]:
    scalar_names = surf.add_vertex_dataset(
        name=scalars,
        data=array,
        left_data=left_array,
        right_data=right_array,
        left_slice=left_slice,
        right_slice=right_slice,
        default_slices=default_slices,
        is_masked=is_masked,
        apply_mask=apply_mask,
        null_value=null_value,
        select=select,
        exclude=exclude,
        allow_multihemisphere=allow_multihemisphere,
        coerce_to_scalar=coerce_to_scalar,
    )
    if plot:
        surf_scalars = tuple(list(surf_scalars) + list(scalar_names))
    return surf, surf_scalars


def surf_scalars_from_nifti_f(
    surf: CortexTriSurface,
    scalars: str,
    nifti: nb.Nifti1Image,
    f_resample: callable,
    method: Literal['nearest', 'linear'] = 'linear',
    surf_scalars: Sequence[str] = (),
    null_value: Optional[float] = 0.0,
    threshold: Optional[float] = None,
    select: Optional[Sequence[int]] = None,
    exclude: Optional[Sequence[int]] = None,
    allow_multihemisphere: bool = True,
    coerce_to_scalar: bool = True,
    plot: bool = False,
) -> Mapping:
    left, right = f_resample(nifti, method=method, threshold=threshold)
    scalar_names = surf.add_gifti_dataset(
        name=scalars,
        left_gifti=left,
        right_gifti=right,
        is_masked=False,
        apply_mask=True,
        null_value=null_value,
        select=select,
        exclude=exclude,
        allow_multihemisphere=allow_multihemisphere,
        coerce_to_scalar=coerce_to_scalar,
    )
    if plot:
        surf_scalars = tuple(list(surf_scalars) + list(scalar_names))
    return surf, surf_scalars


def _cmap_impl_hemisphere(
    surf: CortexTriSurface,
    hemisphere: Literal['left', 'right'],
    parcellation: str,
    colours: Tensor,
    null_value: float,
) -> Tuple[Tensor, Tuple[float, float]]:
    """
    Helper function used when creating a colormap for a cortical hemisphere.
    """
    parcellation = surf.point_data[hemisphere][parcellation]
    start = int(np.min(parcellation[parcellation != null_value])) - 1
    stop = int(np.max(parcellation))
    cmap = ListedColormap(colours[start:stop, :3])
    clim = (start + 0.1, stop + 0.9)
    return cmap, clim


def make_cmap_f(
    surf: CortexTriSurface,
    cmap: str,
    parcellation: str,
    null_value: float = 0,
    return_left: bool = True,
    return_right: bool = True,
    return_both: bool = False,
) -> Union[
    Tuple[Tensor, Tuple[float, float]],
    Tuple[
        Tuple[Tensor, Tuple[float, float]],
        Tuple[Tensor, Tuple[float, float]],
    ],
    Tuple[
        Tuple[Tensor, Tuple[float, float]],
        Tuple[Tensor, Tuple[float, float]],
        Tuple[Tensor, Tuple[float, float]],
    ],
]:
    """
    Create a colormap for a parcellation dataset defined over a cortical
    surface.

    Parameters
    ----------
    surf : CortexTriSurface
        The surface over which the parcellation is defined. It should include
        both the parcellation dataset and a vertexwise colormap dataset.
    cmap : str
        The name of the vertex-wise colormap dataset to use.
    parcellation : str
        The name of the parcellation dataset to use.
    null_value : float (default: 0)
        The value to use for null values in the parcellation dataset.
    return_left : bool (default: True)
        Whether to return a colormap for the left hemisphere.
    return_right : bool (default: True)
        Whether to return a colormap for the right hemisphere.
    return_both : bool (default: False)
        Whether to return a colormap for both hemispheres.
    """
    colours = surf.parcellate_vertex_dataset(cmap, parcellation)
    colours = np.minimum(colours, 1)
    colours = np.maximum(colours, 0)

    ret = []
    if return_left or return_both:
        cmap_left, clim_left = _cmap_impl_hemisphere(
            surf,
            'left',
            parcellation,
            colours,
            null_value,
        )
        ret += [(cmap_left, clim_left)]
    if return_right or return_both:
        cmap_right, clim_right = _cmap_impl_hemisphere(
            surf,
            'right',
            parcellation,
            colours,
            null_value,
        )
        ret += [(cmap_right, clim_right)]
    if return_both:
        # TODO: Rewrite this to skip the unnecessary intermediate blocks
        #       above.
        cmin = min(clim_left[0], clim_right[0])
        cmax = max(clim_left[1], clim_right[1])
        cmap = ListedColormap(colours[:, :3])
        clim = (cmin, cmax)
        ret += [(cmap, clim)]
    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


def parcellate_colormap_f(
    surf: CortexTriSurface,
    cmap_name: str,
    parcellation_name: str,
    cmap: str,
    target: Union[str, Sequence[str]] = ('surf_scalars', 'node'),
):
    surf.add_cifti_dataset(
        name=f'cmap_{cmap_name}',
        cifti=cmap,
        is_masked=True,
        apply_mask=False,
        null_value=0.0,
        coerce_to_scalar=False,
    )
    (
        (cmap_left, clim_left),
        (cmap_right, clim_right),
        (cmap, clim),
    ) = make_cmap_f(
        surf, f'cmap_{cmap_name}', parcellation_name, return_both=True
    )

    ret = {'surf': surf}
    if isinstance(target, str):
        target = [target]
    if 'surf_scalars' in target:
        ret['surf_scalars_cmap'] = (cmap_left, cmap_right)
        ret['surf_scalars_clim'] = (clim_left, clim_right)
        ret['surf_scalars_below_color'] = [0, 0, 0, 0]
    if 'node' in target:
        ret['node_cmap'] = cmap
        ret['node_clim'] = clim
        ret['node_below_color'] = [0, 0, 0, 0]
    return ret


def parcellate_surf_scalars_f(
    surf: CortexTriSurface,
    scalars: str,
    sink: str,
    parcellation_name: str,
    surf_scalars: Sequence[str] = (),
    plot: bool = True,
) -> Tuple[CortexTriSurface, Sequence[str]]:
    parcellated = surf.parcellate_vertex_dataset(
        name=scalars,
        parcellation=parcellation_name,
    )
    surf.scatter_into_parcels(
        data=parcellated,
        parcellation=parcellation_name,
        sink=sink,
    )
    if plot:
        surf_scalars = tuple(list(surf_scalars) + [sink])
    return surf, surf_scalars


def scatter_into_parcels_f(
    surf: CortexTriSurface,
    scalars: str,
    parcellated: Tensor,
    parcellation_name: str,
    surf_scalars: Sequence[str] = (),
    plot: bool = True,
) -> Tuple[CortexTriSurface, Sequence[str]]:
    surf.scatter_into_parcels(
        data=parcellated,
        parcellation=parcellation_name,
        sink=scalars,
    )
    if plot:
        surf_scalars = tuple(list(surf_scalars) + [scalars])
    return surf, surf_scalars


def vertex_to_face_f(
    surf: CortexTriSurface,
    scalars: Sequence[str],
    interpolation: Literal['mode', 'mean'] = 'mode',
) -> Tuple[CortexTriSurface, Sequence[str]]:
    surf.vertex_to_face(
        name=scalars,
        interpolation=interpolation,
    )
    return surf


def _copy_dict_from_params(
    params: Mapping[str, Any],
    key_default: Sequence[Tuple[str, Any]] = (),
) -> Mapping[str, Any]:
    ret = {}
    for key, default in key_default:
        ret[key] = params.pop(key, default)
    return ret, params


def _move_params_to_dict(
    params: Mapping[str, Any],
    src_dst_default: Sequence[Tuple[str, str, Any]] = (),
) -> Mapping[str, Any]:
    ret = {}
    for src, dst, default in src_dst_default:
        ret[dst] = params.pop(src, default)
    return ret, params


def add_surface_overlay_f(
    chains: Sequence[callable],
    params: Mapping[str, Any],
) -> Mapping[str, Any]:
    surf_scalars_layers = params.pop('surf_scalars_layers', ([], []))
    store, params = _copy_dict_from_params(
        params,
        (
            ('surf_scalars', SURF_SCALARS_DEFAULT_VALUE),
            ('surf_scalars_cmap', SURF_SCALARS_CMAP_DEFAULT_VALUE),
            ('surf_scalars_clim', SURF_SCALARS_CLIM_DEFAULT_VALUE),
            (
                'surf_scalars_below_color',
                SURF_SCALARS_BELOW_COLOR_DEFAULT_VALUE,
            ),
        ),
    )

    inner_f = ichain(*chains)(_null_op)
    params = inner_f(**params)
    layer_name = params.pop('surf_scalars')
    if not isinstance(layer_name, str): # It's a list or tuple
        layer_name = layer_name[0]

    layer_params, params = _move_params_to_dict(params, (
        ('surf_scalars_cmap', 'cmap', SURF_SCALARS_CMAP_DEFAULT_VALUE),
        ('surf_scalars_clim', 'clim', LAYER_CLIM_DEFAULT_VALUE),
        (
            'surf_scalars_below_color',
            'below_color',
            LAYER_BELOW_COLOR_DEFAULT_VALUE,
        ),
    ))

    layer_params, params = _move_params_to_dict(
        params,
        (
            (f'{layer_name}_cmap', 'cmap', layer_params['cmap']),
            (f'{layer_name}_clim', 'clim', layer_params['clim']),
            (
                f'{layer_name}_cmap_negative',
                'cmap_negative',
                LAYER_CMAP_NEGATIVE_DEFAULT_VALUE,
            ),
            (
                f'{layer_name}_clim_negative',
                'clim_negative',
                LAYER_CLIM_NEGATIVE_DEFAULT_VALUE,
            ),
            (f'{layer_name}_alpha', 'alpha', LAYER_ALPHA_DEFAULT_VALUE),
            (f'{layer_name}_color', 'color', LAYER_COLOR_DEFAULT_VALUE),
            (
                f'{layer_name}_below_color',
                'below_color',
                layer_params['below_color'],
            ),
            (
                f'{layer_name}_blend_mode',
                'blend_mode',
                LAYER_BLEND_MODE_DEFAULT_VALUE,
            ),
        ),
    )

    hemi_params_p = _get_hemisphere_parameters(
        surf_scalars_cmap=layer_params['cmap'],
        surf_scalars_clim=layer_params['clim'],
        surf_scalars_layers=None,
    )
    hemi_params_n = _get_hemisphere_parameters(
        surf_scalars_cmap=layer_params['cmap_negative'],
        surf_scalars_clim=layer_params['clim_negative'],
        surf_scalars_layers=None,
    )
    layer_left = Layer(
        name=layer_name,
        cmap=hemi_params_p.get('left', 'surf_scalars_cmap'),
        clim=hemi_params_p.get('left', 'surf_scalars_clim'),
        cmap_negative=hemi_params_n.get('left', 'surf_scalars_cmap'),
        clim_negative=hemi_params_n.get('left', 'surf_scalars_clim'),
        alpha=layer_params['alpha'],
        color=layer_params['color'],
        below_color=layer_params['below_color'],
        blend_mode=layer_params['blend_mode'],
    )
    layer_right = Layer(
        name=layer_name,
        cmap=hemi_params_p.get('right', 'surf_scalars_cmap'),
        clim=hemi_params_p.get('right', 'surf_scalars_clim'),
        cmap_negative=hemi_params_n.get('right', 'surf_scalars_cmap'),
        clim_negative=hemi_params_n.get('right', 'surf_scalars_clim'),
        alpha=layer_params['alpha'],
        color=layer_params['color'],
        below_color=layer_params['below_color'],
        blend_mode=layer_params['blend_mode'],
    )

    surf_scalars_layers = (
        list(surf_scalars_layers[0]) + [layer_left],
        list(surf_scalars_layers[1]) + [layer_right],
    )

    return {
        **params,
        **{
            'surf_scalars_layers': surf_scalars_layers,
            **store,
        },
    }


def add_points_overlay_f(
    chains: Sequence[callable],
    params: Mapping[str, Any],
) -> Mapping[str, Any]:
    points_scalars_layers = params.pop('points_scalars_layers', [])
    store, params = _copy_dict_from_params(
        params,
        (
            ('points_scalars', POINTS_SCALARS_DEFAULT_VALUE),
            ('points_scalars_cmap', POINTS_SCALARS_CMAP_DEFAULT_VALUE),
            ('points_scalars_clim', POINTS_SCALARS_CLIM_DEFAULT_VALUE),
            (
                'points_scalars_below_color',
                POINTS_SCALARS_BELOW_COLOR_DEFAULT_VALUE,
            ),
        ),
    )

    inner_f = ichain(*chains)(_null_op)
    params = inner_f(**params)
    layer_name = params.pop('points_scalars')
    if not isinstance(layer_name, str): # It's a list or tuple
        layer_name = layer_name[0]

    layer_params, params = _move_params_to_dict(params, (
        ('points_scalars_cmap', 'cmap', POINTS_SCALARS_CMAP_DEFAULT_VALUE),
        ('points_scalars_clim', 'clim', LAYER_CLIM_DEFAULT_VALUE),
        (
            'points_scalars_below_color',
            'below_color',
            LAYER_BELOW_COLOR_DEFAULT_VALUE,
        ),
    ))

    layer_params, params = _move_params_to_dict(
        params,
        (
            (f'{layer_name}_cmap', 'cmap', layer_params['cmap']),
            (f'{layer_name}_clim', 'clim', layer_params['clim']),
            (
                f'{layer_name}_cmap_negative',
                'cmap_negative',
                LAYER_CMAP_NEGATIVE_DEFAULT_VALUE,
            ),
            (
                f'{layer_name}_clim_negative',
                'clim_negative',
                LAYER_CLIM_NEGATIVE_DEFAULT_VALUE,
            ),
            (f'{layer_name}_alpha', 'alpha', LAYER_ALPHA_DEFAULT_VALUE),
            (f'{layer_name}_color', 'color', LAYER_COLOR_DEFAULT_VALUE),
            (
                f'{layer_name}_below_color',
                'below_color',
                layer_params['below_color'],
            ),
            (
                f'{layer_name}_blend_mode',
                'blend_mode',
                LAYER_BLEND_MODE_DEFAULT_VALUE,
            ),
        ),
    )

    layer = Layer(
        name=layer_name,
        cmap=layer_params['cmap'],
        clim=layer_params['clim'],
        cmap_negative=layer_params['cmap_negative'],
        clim_negative=layer_params['clim_negative'],
        alpha=layer_params['alpha'],
        color=layer_params['color'],
        below_color=layer_params['below_color'],
        blend_mode=layer_params['blend_mode'],
    )

    points_scalars_layers = list(points_scalars_layers) + [layer]

    return {
        **params,
        **{
            'points_scalars_layers': points_scalars_layers,
            **store,
        },
    }


def add_network_overlay_f(
    layer_name: str,
    chains: Sequence[callable],
    params: Mapping[str, Any],
) -> Mapping[str, Any]:
    network_layers = params.pop('network_layers', [])
    store, params = _copy_dict_from_params(
        params,
        (
            ('node_cmap', NODE_CMAP_DEFAULT_VALUE),
            ('node_clim', NODE_CLIM_DEFAULT_VALUE),
            ('node_color', NODE_COLOR_DEFAULT_VALUE),
            ('node_radius', NODE_RADIUS_DEFAULT_VALUE),
            ('node_radius_range', NODE_RLIM_DEFAULT_VALUE),
            ('node_alpha', NODE_ALPHA_DEFAULT_VALUE),
            ('edge_cmap', EDGE_CMAP_DEFAULT_VALUE),
            ('edge_clim', EDGE_CLIM_DEFAULT_VALUE),
            ('edge_color', EDGE_COLOR_DEFAULT_VALUE),
            ('edge_alpha', EDGE_ALPHA_DEFAULT_VALUE),
            ('edge_radius', EDGE_RADIUS_DEFAULT_VALUE),
            ('edge_radius_range', EDGE_RLIM_DEFAULT_VALUE),
        )
    )

    inner_f = ichain(*chains)(_null_op)
    params = inner_f(**params)

    node_params, params = _move_params_to_dict(params, (
        ('node_cmap', 'cmap', NODE_CMAP_DEFAULT_VALUE),
        ('node_clim', 'clim', NODE_CLIM_DEFAULT_VALUE),
        ('node_color', 'color', NODE_COLOR_DEFAULT_VALUE),
        ('node_radius', 'radius', NODE_RADIUS_DEFAULT_VALUE),
        ('node_radius_range', 'radius_range', NODE_RLIM_DEFAULT_VALUE),
        ('node_alpha', 'alpha', NODE_ALPHA_DEFAULT_VALUE),
    ))
    edge_params, params = _move_params_to_dict(params, (
        ('edge_cmap', 'cmap', EDGE_CMAP_DEFAULT_VALUE),
        ('edge_clim', 'clim', EDGE_CLIM_DEFAULT_VALUE),
        ('edge_color', 'color', EDGE_COLOR_DEFAULT_VALUE),
        ('edge_alpha', 'alpha', EDGE_ALPHA_DEFAULT_VALUE),
        ('edge_radius', 'radius', EDGE_RADIUS_DEFAULT_VALUE),
        ('edge_radius_range', 'radius_range', EDGE_RLIM_DEFAULT_VALUE),
    ))

    node_params, params = _move_params_to_dict(
        params,
        (
            (f'{layer_name}_node_cmap', 'cmap', node_params['cmap']),
            (f'{layer_name}_node_clim', 'clim', node_params['clim']),
            (f'{layer_name}_node_color', 'color', node_params['color']),
            (f'{layer_name}_node_alpha', 'alpha', node_params['alpha']),
            (f'{layer_name}_node_radius', 'radius', node_params['radius']),
            (
                f'{layer_name}_node_radius_range',
                'radius_range',
                node_params['radius_range'],
            ),
            (
                f'{layer_name}_node_cmap_negative',
                'cmap_negative',
                LAYER_CMAP_NEGATIVE_DEFAULT_VALUE,
            ),
            (
                f'{layer_name}_node_clim_negative',
                'clim_negative',
                LAYER_CLIM_NEGATIVE_DEFAULT_VALUE,
            ),
            (
                f'{layer_name}_node_below_color',
                'below_color',
                NETWORK_LAYER_BELOW_COLOR_DEFAULT_VALUE,
            ),
        ),
    )
    edge_params, params = _move_params_to_dict(
        params,
        (
            (f'{layer_name}_edge_cmap', 'cmap', edge_params['cmap']),
            (f'{layer_name}_edge_clim', 'clim', edge_params['clim']),
            (f'{layer_name}_edge_color', 'color', edge_params['color']),
            (f'{layer_name}_edge_alpha', 'alpha', edge_params['alpha']),
            (f'{layer_name}_edge_radius', 'radius', edge_params['radius']),
            (
                f'{layer_name}_edge_radius_range',
                'radius_range',
                edge_params['radius_range'],
            ),
            (
                f'{layer_name}_edge_cmap_negative',
                'cmap_negative',
                LAYER_CMAP_NEGATIVE_DEFAULT_VALUE,
            ),
            (
                f'{layer_name}_edge_clim_negative',
                'clim_negative',
                LAYER_CLIM_NEGATIVE_DEFAULT_VALUE,
            ),
            (
                f'{layer_name}_edge_below_color',
                'below_color',
                NETWORK_LAYER_BELOW_COLOR_DEFAULT_VALUE,
            ),
        ),
    )

    edge_layer = EdgeLayer(
        name=layer_name,
        cmap=edge_params['cmap'],
        clim=edge_params['clim'],
        cmap_negative=edge_params['cmap_negative'],
        clim_negative=edge_params['clim_negative'],
        color=edge_params['color'],
        alpha=edge_params['alpha'],
        below_color=edge_params['below_color'],
        radius=edge_params['radius'],
        radius_range=edge_params['radius_range'],
    )
    node_layer = NodeLayer(
        name=layer_name,
        cmap=node_params['cmap'],
        clim=node_params['clim'],
        cmap_negative=node_params['cmap_negative'],
        clim_negative=node_params['clim_negative'],
        color=node_params['color'],
        alpha=node_params['alpha'],
        below_color=node_params['below_color'],
        radius=node_params['radius'],
        radius_range=node_params['radius_range'],
        edge_layers=[edge_layer],
    )

    network_layers = list(network_layers) + [node_layer]
    networks = build_network_f(
        name=layer_name,
        node_coor=params['node_coor'],
        node_values=params.get('node_values', None),
        edge_values=params.get('edge_values', None),
        lh_mask=params.get('lh_mask', None),
        networks=params.get('networks', None),
    )

    return {
        **params,
        **{
            'network_layers': network_layers,
            'networks': networks,
            **store,
        },
    }


def build_network_f(
    name: str,
    node_coor: Tensor,
    node_values: Optional[pd.DataFrame] = None,
    edge_values: Optional[pd.DataFrame] = None,
    lh_mask: Optional[Tensor] = None,
    networks: Optional[NetworkDataCollection] = None,
) -> NetworkDataCollection:
    network = NetworkData(
        name=name,
        coor=node_coor,
        nodes=node_values,
        edges=edge_values,
        lh_mask=lh_mask,
    )
    if networks is None:
        networks = NetworkDataCollection([network])
    else:
        networks = networks + NetworkDataCollection([network])
    return networks


def node_coor_from_parcels_f(
    surf: CortexTriSurface,
    surf_projection: str,
    parcellation: str,
    null_value: float = 0,
) -> Tensor:
    coor = surf.parcel_centres_of_mass(
        parcellation=parcellation,
        projection=surf_projection,
    )
    lh_data = surf.point_data['left'][parcellation]
    rh_data = surf.point_data['right'][parcellation]
    parcel_ids = np.unique(np.concatenate((lh_data, rh_data)))
    parcel_ids = parcel_ids[parcel_ids != null_value]
    lh_mask = np.isin(parcel_ids, lh_data)
    return coor, lh_mask


def add_node_variable_f(
    name: str = "node",
    val: Union[Tensor, str] = None,
    threshold: Union[float, int] = 0.0,
    percent_threshold: bool = False,
    topk_threshold: bool = False,
    absolute: bool = True,
    node_selection: Optional[Tensor] = None,
    incident_edge_selection: Optional[Tensor] = None,
    removed_val: Optional[float] = None,
    surviving_val: Optional[float] = 1.0,
) -> pd.DataFrame:
    if isinstance(val, str):
        val = pd.read_csv(val, sep='\t', header=None).values

    node_df = filter_node_data(
        val=val,
        name=name,
        threshold=threshold,
        percent_threshold=percent_threshold,
        topk_threshold=topk_threshold,
        absolute=absolute,
        node_selection=node_selection,
        incident_edge_selection=incident_edge_selection,
        removed_val=removed_val,
        surviving_val=surviving_val,
    )

    return node_df


def add_edge_variable_f(
    name: str,
    adj: Union[Tensor, str] = None,
    threshold: float = 0.0,
    percent_threshold: bool = False,
    topk_threshold_nodewise: bool = False,
    absolute: bool = True,
    incident_node_selection: Optional[Tensor] = None,
    connected_node_selection: Optional[Tensor] = None,
    edge_selection: Optional[Tensor] = None,
    removed_val: Optional[float] = None,
    surviving_val: Optional[float] = 1.0,
    emit_degree: Union[bool, Literal["abs", "+", "-"]] = False,
    emit_incident_nodes: Union[bool, tuple] = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if isinstance(adj, str):
        adj = pd.read_csv(adj, sep='\t', header=None).values

    ret = filter_adjacency_data(
        adj=adj,
        name=name,
        threshold=threshold,
        percent_threshold=percent_threshold,
        topk_threshold_nodewise=topk_threshold_nodewise,
        absolute=absolute,
        incident_node_selection=incident_node_selection,
        connected_node_selection=connected_node_selection,
        edge_selection=edge_selection,
        removed_val=removed_val,
        surviving_val=surviving_val,
        emit_degree=emit_degree,
        emit_incident_nodes=emit_incident_nodes,
    )

    if emit_degree is not False or emit_incident_nodes is not False:
        return ret
    return ret, None


def add_postprocessor_f(
    name: str,
    postprocessor: Callable,
    auxwriter: Optional[Callable] = None,
    postprocessors: Optional[Sequence[Callable]] = None,
) -> Sequence[Callable]:
    if postprocessors is None:
        postprocessors = {}
    if auxwriter is None:
        auxwriter = F(_null_auxwriter)
    postprocessors[name] = (postprocessor, auxwriter)
    return postprocessors


def transform_postprocessor_f(
    name: str,
    transformer: Optional[Callable] = None,
    postprocessor_params: Optional[Mapping] = None,
    aux_transformer: Optional[Callable] = None,
    auxwriter: Optional[Callable] = None,
    auxwriter_params: Optional[Mapping] = None,
    postprocessors: Optional[Sequence[Callable]] = None,
    composition_order: Literal['pre', 'post'] = 'pre',
) -> Sequence[Callable]:
    notfound = False
    if postprocessors is None:
        notfound = True
    postprocessor, _auxwriter = postprocessors.get(name, (None, None))
    if postprocessor is None:
        notfound = True
    if notfound:
        raise ValueError(
            f'Postprocessor {name} not found in postprocessors '
            f'{postprocessors}'
        )
    if postprocessor_params is None:
        postprocessor_params = {}
    if transformer is not None:
        if composition_order == 'pre':
            postprocessor = Composition(
                compositor=direct_compositor,
                outer=postprocessor,
                inner=transformer,
            ).bind_curried(**postprocessor_params)
        else:
            postprocessor = Composition(
                compositor=reversed_args_compositor,
                outer=transformer,
                inner=postprocessor,
            ).bind_curried(**postprocessor_params)
    if aux_transformer is not None:
        if auxwriter is not None:
            raise ValueError(
                'Cannot specify both aux_transformer and auxwriter'
            )
        if auxwriter_params is None:
            auxwriter_params = {}
        if composition_order == 'pre':
            auxwriter = Composition(
                compositor=direct_compositor,
                outer=_auxwriter,
                inner=aux_transformer,
            ).bind_curried(**auxwriter_params)
        else:
            auxwriter = Composition(
                compositor=reversed_args_compositor,
                outer=aux_transformer,
                inner=_auxwriter,
            ).bind_curried(**auxwriter_params)
    postprocessors[name] = (postprocessor, auxwriter)
    return postprocessors


def scalar_focus_camera_f(
    surf: CortexTriSurface,
    hemispheres: Optional[str],
    surf_scalars: str,
    surf_projection: str,
    kind: str,
    plotter: Optional[pv.Plotter] = None,
) -> Mapping:
    hemispheres = hemispheres or ('left', 'right')
    views = []
    for hemisphere in hemispheres:
        if kind == "centroid":
            coor = surf.scalars_centre_of_mass(
                hemisphere=hemisphere,
                scalars=surf_scalars,
                projection=surf_projection,
            )
        elif kind == "peak":
            coor = surf.scalars_peak(
                hemisphere=hemisphere,
                scalars=surf_scalars,
                projection=surf_projection,
            )
        vector, focus = auto_focus(
            vector=coor,
            plotter=surf.__getattribute__(hemisphere),
            slack=2,
        )
        if np.any(np.isnan(vector)):
            # TODO: Let's figure out a good way to signal that there are no
            #       valid coordinates for this hemisphere, and to handle the
            #       plot outputs (since dropping them could disrupt plotting
            #       grids specified by the user).
            # logging.warning(
            #     f"NaN detected in focus coordinates for {hemisphere}, "
            #     f"skipping hemisphere"
            # )
            vector = (0, 0, 0)
        views.append(
            (vector, focus, (0, 0, 1))
        )
    return plotter, views, hemispheres


def scalar_focus_camera_aux_f(
    metadata: Mapping[str, Sequence[str]],
    kind: str,
    hemisphere: Optional[Sequence[Literal['left', 'right', 'both']]] = None,
) -> Mapping:
    if len(hemisphere) == 1 and hemisphere[0] != 'both':
        metadata['view'] = ['focused']
    else:
        metadata['view'] = ['focusedleft', 'focusedright']
    metadata['focus'] = [kind]
    mapper = replicate(
        spec=('view', 'focus', 'hemisphere'),
        broadcast_out_of_spec=True,
    )
    return mapper(**metadata)


def closest_ortho_camera_f(
    surf: CortexTriSurface,
    hemispheres: Optional[str],
    surf_scalars: str,
    surf_projection: str,
    n_ortho: int,
    plotter: Optional[pv.Plotter] = None,
) -> Mapping:
    metric = 'euclidean'
    # if projection == 'sphere':
    #     metric = 'spherical'
    #     cur_proj = surf.projection
    #     surf.left.project('sphere')
    #     surf.right.project('sphere')
    # else:
    #     metric = 'euclidean'
    hemispheres = hemispheres or ('left', 'right')
    if hemispheres == 'both' or 'both' in hemispheres:
        hemispheres = ('left', 'right')
    closest_poles = []
    hemispheres_out = []
    for hemisphere in hemispheres:
        coor = surf.scalars_centre_of_mass(
            hemisphere=hemisphere,
            scalars=surf_scalars,
            projection=surf_projection,
        )
        closest_poles_hemisphere = surf.closest_poles(
            hemisphere=hemisphere,
            coors=coor,
            metric=metric,
            n_poles=n_ortho,
        ).squeeze().tolist()
        hemispheres_out += [hemisphere] * n_ortho
        # You really shouldn't be using this autocamera when rendering both
        # hemispheres, but if you do, this will make sure that the camera
        # provides valid views.
        if hemispheres == ('left', 'right') or hemispheres == 'both':
            other_hemisphere = 'right' if hemisphere == 'left' else 'left'
            closest_poles_hemisphere = [
                p if p != 'lateral' else hemisphere
                for p in closest_poles_hemisphere
            ]
            closest_poles_hemisphere = [
                p if p != 'medial' else other_hemisphere
                for p in closest_poles_hemisphere
            ]
        closest_poles += closest_poles_hemisphere
    # if surf_projection == 'sphere':
    #     surf.left.project(cur_proj)
    #     surf.right.project(cur_proj)
    return plotter, closest_poles, hemispheres


def closest_ortho_camera_aux_f(
    metadata: Mapping[str, Sequence[str]],
    n_ortho: int,
    hemisphere: Optional[Sequence[Literal['left', 'right', 'both']]] = None,
) -> Mapping:
    metadata['index'] = [str(i) for i in range(n_ortho)]
    if len(hemisphere) == 1 and hemisphere[0] != 'both':
        metadata['view'] = ['ortho']
    else:
        metadata['view'] = ['ortholeft', 'orthoright']
    mapper = replicate(
        spec=(['view', 'index'], 'hemisphere'),
        broadcast_out_of_spec=True,
    )
    return mapper(**metadata)


def planar_sweep_camera_f(
    surf: CortexTriSurface,
    hemispheres: Optional[str],
    initial: Sequence,
    normal: Optional[Sequence[float]] = None,
    n_steps: int = 10,
    require_planar: bool = True,
    plotter: Optional[pv.Plotter] = None,
) -> Mapping:
    if normal is None:
        _x, _y, _z = initial
        if (_x, _y, _z) == (0, 0, 1) or (_x, _y, _z) == (0, 0, -1):
            normal = (1, 0, 0)
        elif (_x, _y, _z) == (0, 0, 0):
            raise ValueError("initial view cannot be (0, 0, 0)")
        else:
            ref = np.asarray((0, 0, 1))
            initial = np.asarray(initial) / np.linalg.norm(initial)
            rejection = ref - np.dot(ref, initial) * initial
            normal = rejection / np.linalg.norm(rejection)

    if require_planar:
        assert np.isclose(np.dot(initial, normal), 0), (
            'Initial and normal must be orthogonal for a planar sweep. '
            'Got initial: {}, normal: {}. '
            'Set require_planar=False to allow conical sweeps.'
        )
    angles = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)
    cos = np.cos(angles)
    sin = np.sin(angles)
    ax_x = initial / np.linalg.norm(initial)
    ax_z = np.asarray(normal) / np.linalg.norm(normal)
    ax_y = np.cross(ax_z, ax_x)
    ax = np.stack((ax_x, ax_y, ax_z), axis=-1)

    lin = np.zeros((n_steps, 3, 3))
    lin[:, 0, 0] = cos
    lin[:, 0, 1] = -sin
    lin[:, 1, 0] = sin
    lin[:, 1, 1] = cos
    lin[:, -1, -1] = 1

    lin = ax @ lin
    vectors = lin @ np.asarray(initial)

    if hemispheres is None:
        _hemi = ("left", "right")
    elif isinstance(hemispheres, str):
        _hemi = (hemispheres,)
    else:
        _hemi = hemispheres
    views = []
    for h in _hemi:
        if h == "left":
            vecs_hemi = vectors.copy()
            vecs_hemi[:, 0] = -vecs_hemi[:, 0]
        else:
            vecs_hemi = vectors
        for vector in vecs_hemi:
            v, focus = auto_focus(
                vector=vector,
                plotter=surf.__getattribute__(h),
                slack=2,
            )
            views.append(
                (v, focus, (0, 0, 1))
            )
    return plotter, views, hemispheres


def planar_sweep_camera_aux_f(
    metadata: Mapping[str, Sequence[str]],
    n_steps: int,
    hemisphere: Optional[Sequence[Literal['left', 'right', 'both']]] = None,
) -> Mapping:
    metadata['index'] = [str(i) for i in range(n_steps)]
    if len(hemisphere) == 1 and hemisphere[0] != 'both':
        metadata['view'] = ['planar']
    else:
        metadata['view'] = ['planarleft', 'planarright']
    mapper = replicate(
        spec=(['view', 'index'], 'hemisphere'),
        broadcast_out_of_spec=True,
    )
    return mapper(**metadata)


def auto_camera_f(
    surf: CortexTriSurface,
    surf_scalars: Optional[str],
    surf_projection: str,
    hemispheres: Optional[str],
    n_ortho: int = 0,
    focus: Optional[Literal["centroid", "peak"]] = None,
    n_angles: int = 0,
    initial_angle: Tuple[float, float, float] = (1, 0, 0),
    normal_vector: Optional[Tuple[float, float, float]] = None,
    plotter: Optional[pv.Plotter] = None,
) -> Mapping:
    views_ortho = views_focused = views_planar = []
    if n_ortho > 0:
        _, views_ortho, _ = closest_ortho_camera_f(
            surf=surf,
            hemispheres=hemispheres,
            surf_scalars=surf_scalars,
            surf_projection=surf_projection,
            n_ortho=n_ortho,
            plotter=plotter,
        )
    if focus is not None:
        _, views_focused, _ = scalar_focus_camera_f(
            surf=surf,
            hemispheres=hemispheres,
            surf_scalars=surf_scalars,
            surf_projection=surf_projection,
            kind=focus,
            plotter=plotter,
        )
    if n_angles > 0:
        _, views_planar, _ = planar_sweep_camera_f(
            surf=surf,
            hemispheres=hemispheres,
            n_steps=n_angles,
            initial=initial_angle,
            normal=normal_vector,
            plotter=plotter,
        )
    views = tuple(views_ortho + views_focused + views_planar)
    return plotter, views, hemispheres


def auto_camera_aux_f(
    metadata: Mapping[str, Sequence[str]],
    n_ortho: int = 0,
    focus: Optional[Literal["centroid", "peak"]] = None,
    n_angles: int = 0,
    hemisphere: Optional[Sequence[Literal['left', 'right', 'both']]] = None,
) -> Mapping:
    viewbuilder = metadata.copy() # There shouldn't be any nesting
    viewbuilder = closest_ortho_camera_aux_f(
        metadata=viewbuilder,
        n_ortho=n_ortho,
        hemisphere=hemisphere,
    )
    view, extra, hemi = (
        viewbuilder['view'],
        viewbuilder['index'],
        viewbuilder['hemisphere'],
    )
    viewbuilder = metadata.copy()
    viewbuilder = scalar_focus_camera_aux_f(
        metadata=viewbuilder,
        kind=focus,
        hemisphere=hemisphere,
    )
    view += viewbuilder['view']
    extra += viewbuilder['focus']
    hemi += viewbuilder['hemisphere']
    viewbuilder = metadata.copy()
    viewbuilder = planar_sweep_camera_aux_f(
        metadata=viewbuilder,
        n_steps=n_angles,
        hemisphere=hemisphere,
    )
    view += viewbuilder['view']
    extra += viewbuilder['index']
    hemi += viewbuilder['hemisphere']
    metadata['view'] = view
    metadata['index'] = extra
    metadata['hemisphere'] = hemi
    mapper = replicate(
        spec=('view', 'index', 'hemisphere'),
        broadcast_out_of_spec=True,
    )
    return mapper(**metadata)


def plot_to_image_f(
    plotter: pv.Plotter,
    views: Union[Sequence, Mapping, Literal['__default__']] = '__default__',
    window_size: Tuple[int, int] = (1920, 1080),
    hemispheres: Sequence[Literal['left', 'right', 'both']] = None,
    plot_scalar_bar: bool = False,
    close_plotter: bool = True,
) -> Tuple[Tensor]:
    if len(hemispheres) == 1:
        hemisphere = hemispheres[0]
    elif tuple(hemispheres) == ('left', 'right'):
        hemisphere = 'both'
    else:
        hemisphere = hemispheres
    if isinstance(views, Mapping):
        views = views[hemisphere]
    if views == '__default__':
        views = set_default_views(hemisphere)
    ret = []
    if not plot_scalar_bar:
        # TODO: This breaks if there's more than one scalar bar. We'll
        #       overhaul the bar plotter system when we add overlays.
        try:
            plotter.remove_scalar_bar()
        except (IndexError, ValueError):
            pass
    for cpos in views:
        plotter.camera.zoom('tight')
        plotter.show(
            cpos=cortex_cameras(cpos, plotter=plotter, hemisphere=hemisphere),
            auto_close=False,
        )
        img = plotter.screenshot(
            True,
            window_size=window_size,
            return_img=True,
        )
        ret.append(img)
    if close_plotter:
        plotter.close()
    return tuple(ret)


def plot_to_image_aux_f(
    metadata: Mapping[str, Sequence[str]],
    hemisphere: Optional[Sequence[Literal['left', 'right', 'both']]] = None,
    views: Union[
        Sequence,
        Mapping,
        Literal['__default__', '__final__'],
    ] = '__default__',
    n_scenes: int = 1,
) -> Mapping[str, Sequence[str]]:
    if hemisphere is None:
        hemisphere = 'both'
    elif len(hemisphere) == 1:
        hemisphere = hemisphere[0]
    else:
        hemisphere = 'both'
    if isinstance(views, Mapping):
        views = views[hemisphere]
    if views == '__default__':
        views = set_default_views(hemisphere)
    elif views == '__final__':
        views = [f'final{i}' for i in range(n_scenes)]
    if views != '__final__':
        views = [format_position_as_string(cpos) for cpos in views]
    mapper = replicate(spec=['view'], broadcast_out_of_spec=True)
    return mapper(**metadata, view=views)


def plot_final_view_f(
    plotter: pv.Plotter,
    window_size: Tuple[int, int] = (1920, 1080),
    n_scenes: int = 1,
    plot_scalar_bar: bool = False,
    close_plotter: bool = True,
) -> Tuple[Tensor]:
    if not plot_scalar_bar:
        # TODO: This breaks if there's more than one scalar bar. We'll
        #       overhaul the bar plotter system when we add overlays.
        try:
            plotter.remove_scalar_bar()
        except (IndexError, ValueError):
            pass
    snapshots = [
        plotter.show(
            window_size=window_size,
            auto_close=False,
            return_img=True,
        )
        for _ in range(n_scenes)
    ]
    if close_plotter:
        plotter.close()
    return tuple(snapshots)


def plot_to_html_buffer_f(
    plotter: pv.Plotter,
    window_size: Tuple[int, int] = (1920, 1080),
    close_plotter: bool = True,
) -> StringIO:
    plotter.window_size = window_size
    html_buffer = plotter.export_html(filename=None)
    if close_plotter:
        plotter.close()
    return html_buffer


def save_snapshots_f(
    snapshots: Sequence[Tuple[Tensor, Mapping[str, str]]],
    output_dir: str,
    fname_spec: Optional[str] = None,
    suffix: Optional[str] = None,
    extension: str = 'png',
) -> None:
    def writer(img, fname):
        img = Image.fromarray(img)
        img.save(fname)

    for cimg, cmeta in snapshots:
        write_f(
            writer=writer,
            argument=cimg,
            entities=cmeta,
            output_dir=output_dir,
            fname_spec=fname_spec,
            suffix=suffix,
            extension=extension,
        )


def save_html_f(
    html_buffer: Sequence[Tuple[StringIO, Mapping[str, str]]],
    output_dir: str,
    fname_spec: Optional[str] = None,
    suffix: Optional[str] = None,
    extension: str = 'html',
) -> None:
    def writer(buffer, fname):
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(buffer.read())

    for chtml, cmeta in html_buffer:
        write_f(
            writer=writer,
            argument=chtml,
            entities=cmeta,
            output_dir=output_dir,
            fname_spec=fname_spec,
            suffix=suffix,
            extension=extension,
        )


def plot_to_display_f(
    plotter: Sequence[Tuple[pv.Plotter, Mapping[str, str]]],
    window_size: Tuple[int, int] = (1920, 1080),
) -> None:
    def writer(plotter, fname=None):
        # TODO: window_size apparently does not work. Perhaps it's inheriting
        #       from the theme when the plotter is created?
        plotter.show(window_size=window_size)

    for cplotter, cmeta in plotter:
        write_f(
            writer=writer,
            argument=cplotter,
            entities=cmeta,
            output_dir=None,
            fname_spec=None,
            suffix=None,
            extension=None,
        )
        cplotter.close()


def save_figure_f(
    snapshots: Sequence[Tuple[Tensor, Mapping[str, str]]],
    canvas_size: Tuple[int, int],
    layout: CellLayout,
    output_dir: str,
    sort_by: Optional[Sequence[str]] = None,
    fname_spec: Optional[str] = None,
    suffix: Optional[str] = None,
    extension: str = 'png',
    padding: int = 0,
    canvas_color: Any = (255, 255, 255, 255),
) -> None: # Union[Tuple[Image.Image], Image.Image]:
    panels_per_page = len(layout)
    cell_indices = list(range(panels_per_page))
    try:
        n_scenes = len(snapshots)
        if sort_by is not None:

            def sort_func(snapshot):
                _, cmeta = snapshot
                return tuple(cmeta[cfield] for cfield in sort_by)

            snapshots = sorted(snapshots, key=sort_func)
        if getattr(layout, 'annotations', None) is not None:
            # TODO: This block behaves unexpectedly if there's more than one
            #       page of snapshots.
            # It's an annotated layout, so we'll match the annotations
            # to the snapshot metadata to 'semantically' assign snapshots
            # to cells.
            queries = [meta for (_, meta) in snapshots][:panels_per_page]
            layout, cell_indices = layout.match_and_assign_all(
                queries=queries,
                force_unmatched=True,
            )
        snapshots = list(zip(cell_indices, snapshots))
        if n_scenes > panels_per_page:
            n_panels = panels_per_page
            n_pages = ceil(n_scenes / n_panels)
            snapshots = tuple(
                snapshots[i * n_panels:(i + 1) * n_panels]
                for i in range(n_pages)
            )
        else:
            n_pages = 1
            n_panels = n_scenes
            snapshots = (snapshots,)
    except TypeError:
        # snapshots is a single snapshot
        snapshots = ((snapshots,),)
        n_scenes = 1
        n_pages = 1
        n_panels = 1
    cells = list(layout.partition(*canvas_size, padding=padding))

    def writer(snapshot_group, fname):
        canvas = Image.new('RGBA', canvas_size, color=canvas_color)
        for i, (cimg, cmeta) in snapshot_group:
            panel = cells[i]
            cimg = Image.fromarray(cimg, mode='RGB')
            cimg = scale_image_preserve_aspect_ratio(
                cimg,
                target_size=panel.cell_dim,
            )
            canvas.paste(cimg, panel.cell_loc)
        canvas.save(fname)

    for i, snapshot_group in enumerate(snapshots):
        page = f'{i + 1:{len(str(n_pages))}d}'
        write_f(
            writer=writer,
            argument=snapshot_group,
            entities={'page': page},
            output_dir=output_dir,
            fname_spec=fname_spec,
            suffix=suffix,
            extension=extension,
        )


def save_grid_f(
    snapshots: Sequence[Tuple[Tensor, Mapping[str, str]]],
    canvas_size: Tuple[int, int],
    n_rows: int,
    n_cols: int,
    output_dir: str,
    sort_by: Optional[Sequence[str]] = None,
    annotations: Optional[Mapping[int, Mapping]] = None,
    fname_spec: Optional[str] = None,
    suffix: Optional[str] = None,
    extension: str = 'png',
    order: Literal['row', 'column'] = 'row',
    padding: int = 0,
    canvas_color: Any = (255, 255, 255, 255),
) -> None:
    layout = grid(n_rows=n_rows, n_cols=n_cols, order=order)
    if annotations is not None:
        layout = AnnotatedLayout(layout, annotations=annotations)
    save_figure_f(
        snapshots=snapshots,
        canvas_size=canvas_size,
        layout=layout,
        output_dir=output_dir,
        sort_by=sort_by,
        fname_spec=fname_spec,
        suffix=suffix,
        extension=extension,
        padding=padding,
        canvas_color=canvas_color,
    )


def write_f(
    writer: Callable,
    argument: Any,
    entities: Mapping[str, str],
    *,
    output_dir: str,
    fname_spec: Optional[str] = None,
    suffix: Optional[str] = None,
    extension: str,
) -> None:
    if fname_spec is None:
        fname_spec = '_'.join(f'{k}-{{{k}}}' for k in entities.keys())
    if suffix is not None:
        fname_spec = f'{fname_spec}_{suffix}'
    fname_spec = f'{output_dir}/{fname_spec}.{extension}'
    fname = fname_spec.format(**entities)
    return writer(argument, fname)


def automap_unified_plotter_f(
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
    surf_scalars_layers = SURF_SCALARS_LAYERS_DEFAULT_VALUE,
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
    edge_alpha: Union[float, str] = EDGE_ALPHA_DEFAULT_VALUE,
    network_layers: Union[
        Optional[Sequence[NodeLayer]],
        Tuple[Optional[Sequence[NodeLayer]]]
    ] = None,
    hemisphere: Optional[Literal['left', 'right']] = None,
    hemisphere_slack: Optional[Union[float, Literal['default']]] = 'default',
    off_screen: bool = True,
    copy_actors: bool = False,
    theme: Optional[Any] = None,
    window_size: Optional[Tuple[int, int]] = None,
    use_single_plotter: bool = True,
    postprocessors: Optional[
        Sequence[Mapping[str, Tuple[callable, callable]]]
    ] = None,
    map_spec: Optional[Sequence] = None,
) -> Optional[pv.Plotter]:
    params = locals()
    map_spec = map_spec or [
        'surf_projection',
        'surf_scalars',
        'points_scalars',
        'networks',
        'hemisphere',
    ]
    params.pop('map_spec')
    mapper = replicate(spec=map_spec, weave_type='maximal')

    use_single_plotter = params.pop('use_single_plotter')
    if use_single_plotter:
        plotter = pv.Plotter(
            window_size=window_size,
            off_screen=off_screen,
            theme=theme,
        )
        plotter_param = {'plotter': plotter}
    else:
        plotter_param = {}
    params = {**params, **plotter_param}

    postprocessors = params.pop('postprocessors', None)
    postprocessors = postprocessors or {'plotter': (None, None)}
    postprocessor_names, postprocessors = zip(*postprocessors.items())
    postprocessors, auxwriters = zip(*postprocessors)

    repl_params = mapper(**params)
    repl_vars = set(_flatten(map_spec))
    repl_params = {k: v for k, v in repl_params.items() if k in repl_vars}
    other_params = {k: v for k, v in params.items() if k not in repl_vars}
    params = {**other_params, **repl_params}
    try:
        n_replicates = max([len(v) for v in repl_params.values()])
    except ValueError:
        n_replicates = 1

    output = [
        unified_plotter(
            **{
                k: (v[i % len(v)] if k in repl_vars else v)
                for k, v in params.items()
            },
            postprocessors=postprocessors,
        )
        for i in range(n_replicates)
    ]
    output = {
        k: tuple(_flatten_to_depth(v, 1))
        for k, v in zip(postprocessor_names, zip(*output))
    }
    metadata = [
        plotted_entities(
            **{
                k: (v[i % len(v)] if k in repl_vars else v)
                for k, v in params.items()
            },
            entity_writers=auxwriters,
        )
        for i in range(n_replicates)
    ]
    metadata = {
        k: tuple(_flatten_to_depth([_dict_to_seq(val) for val in v], 1))
        for k, v in zip(postprocessor_names, zip(*metadata))
    }
    output = {k: tuple(zip(output[k], metadata[k])) for k in output.keys()}

    return output


surf_from_archive_p = Primitive(
    surf_from_archive_f,
    'surf_from_archive',
    output=('surf', 'surf_projection'),
    forward_unused=True,
)


surf_scalars_from_cifti_p = Primitive(
    surf_scalars_from_cifti_f,
    'surf_scalars_from_cifti',
    output=('surf', 'surf_scalars'),
    forward_unused=True,
)


surf_scalars_from_gifti_p = Primitive(
    surf_scalars_from_gifti_f,
    'surf_scalars_from_gifti',
    output=('surf', 'surf_scalars'),
    forward_unused=True,
)


surf_scalars_from_array_p = Primitive(
    surf_scalars_from_array_f,
    'surf_scalars_from_array',
    output=('surf', 'surf_scalars'),
    forward_unused=True,
)


points_scalars_from_nifti_p = Primitive(
    points_scalars_from_nifti_f,
    'points_scalars_from_nifti',
    output=('points', 'points_scalars'),
    forward_unused=True,
)


points_scalars_from_array_p = Primitive(
    points_scalars_from_array_f,
    'points_scalars_from_array',
    output=('points', 'points_scalars'),
    forward_unused=True,
)


surf_scalars_from_nifti_p = Primitive(
    surf_scalars_from_nifti_f,
    'surf_scalars_from_nifti',
    output=('surf', 'surf_scalars'),
    forward_unused=True,
)


parcellate_colormap_p = Primitive(
    parcellate_colormap_f,
    'parcellate_colormap',
    output=None,
    forward_unused=True,
)


parcellate_surf_scalars_p = Primitive(
    parcellate_surf_scalars_f,
    'parcellate_surf_scalars',
    output=('surf', 'surf_scalars'),
    forward_unused=True,
)


scatter_into_parcels_p = Primitive(
    scatter_into_parcels_f,
    'scatter_into_parcels',
    output=('surf', 'surf_scalars'),
    forward_unused=True,
)


vertex_to_face_p = Primitive(
    vertex_to_face_f,
    'vertex_to_face',
    output=('surf',),
    forward_unused=True,
)


add_surface_overlay_p = Primitive(
    add_surface_overlay_f,
    'add_surface_overlay',
    output=None,
    forward_unused=True,
)


add_points_overlay_p = Primitive(
    add_points_overlay_f,
    'add_points_overlay',
    output=None,
    forward_unused=True,
)


add_network_overlay_p = Primitive(
    add_network_overlay_f,
    'add_network_overlay',
    output=None,
    forward_unused=True,
)


build_network_p = Primitive(
    build_network_f,
    'build_network',
    output=('networks',),
    forward_unused=True,
)


node_coor_from_parcels_p = Primitive(
    node_coor_from_parcels_f,
    'node_coor_from_parcels',
    output=('node_coor', 'lh_mask'),
    forward_unused=True,
)


add_node_variable_p = Primitive(
    add_node_variable_f,
    'add_node_variable',
    output=('node_values',),
    forward_unused=True,
)


add_edge_variable_p = Primitive(
    add_edge_variable_f,
    'add_edge_variable',
    output=('edge_values', 'node_values'),
    forward_unused=True,
)


add_postprocessor_p = Primitive(
    add_postprocessor_f,
    'add_postprocessor',
    output=('postprocessors',),
    forward_unused=True,
)


transform_postprocessor_p = Primitive(
    transform_postprocessor_f,
    'transform_postprocessor',
    output=('postprocessors',),
    forward_unused=True,
)


plot_to_image_aux_p = Primitive(
    plot_to_image_aux_f,
    'plot_to_image_aux',
    output=None,
    forward_unused=False,
)


scalar_focus_camera_p = Primitive(
    scalar_focus_camera_f,
    'scalar_focus_camera',
    output=('plotter', 'views', 'hemispheres'),
    forward_unused=True,
)


scalar_focus_camera_aux_p = Primitive(
    scalar_focus_camera_aux_f,
    'scalar_focus_camera_aux',
    output=None,
    forward_unused=False,
)


closest_ortho_camera_p = Primitive(
    closest_ortho_camera_f,
    'closest_ortho_camera',
    output=('plotter', 'views', 'hemispheres'),
    forward_unused=True,
)


closest_ortho_camera_aux_p = Primitive(
    closest_ortho_camera_aux_f,
    'closest_ortho_camera_aux',
    output=None,
    forward_unused=False,
)


planar_sweep_camera_p = Primitive(
    planar_sweep_camera_f,
    'planar_sweep_camera',
    output=('plotter', 'views', 'hemispheres'),
    forward_unused=True,
)


planar_sweep_camera_aux_p = Primitive(
    planar_sweep_camera_aux_f,
    'planar_sweep_camera_aux',
    output=None,
    forward_unused=False,
)


auto_camera_p = Primitive(
    auto_camera_f,
    'auto_camera',
    output=('plotter', 'views', 'hemispheres'),
    forward_unused=True,
)


auto_camera_aux_p = Primitive(
    auto_camera_aux_f,
    'auto_camera_aux',
    output=None,
    forward_unused=False,
)


save_snapshots_p = Primitive(
    save_snapshots_f,
    'save_snapshots',
    output=(),
    forward_unused=True,
)


save_html_p = Primitive(
    save_html_f,
    'save_html',
    output=(),
    forward_unused=True,
)


plot_to_display_p = Primitive(
    plot_to_display_f,
    'plot_to_display',
    output=(),
    forward_unused=True,
)


save_figure_p = Primitive(
    save_figure_f,
    'save_figure',
    output=(),
    forward_unused=True,
)


save_grid_p = Primitive(
    save_grid_f,
    'save_grid',
    output=(),
    forward_unused=True,
)


automap_unified_plotter_p = Primitive(
    automap_unified_plotter_f,
    'automap_unified_plotter',
    output=None,
    forward_unused=True,
)
