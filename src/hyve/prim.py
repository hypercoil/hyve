# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Primitive functional atoms
~~~~~~~~~~~~~~~~~~~~~~~~~~
Atomic functional primitives for building more complex functions.
"""
from functools import reduce
import inspect
from io import StringIO
from itertools import chain
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
import svg
from conveyant import (
    Composition,
    Primitive,
    ichain,
)
from conveyant import (
    FunctionWrapper as F,
)
from conveyant import splice_on as splice_on_orig
from conveyant.compositors import (
    _dict_to_seq,
    _seq_to_dict,
    direct_compositor,
    reversed_args_compositor,
)
from conveyant.replicate import _flatten, _flatten_to_depth, replicate
from matplotlib import colors
from PIL import Image

from .const import (
    DEFAULT_WINDOW_SIZE,
    EDGE_ALPHA_DEFAULT_VALUE,
    EDGE_CLIM_DEFAULT_VALUE,
    EDGE_CMAP_DEFAULT_VALUE,
    EDGE_COLOR_DEFAULT_VALUE,
    EDGE_RADIUS_DEFAULT_VALUE,
    EDGE_RLIM_DEFAULT_VALUE,
    EMPIRICAL_DPI,
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
    PXMM,
    SURF_SCALARS_BELOW_COLOR_DEFAULT_VALUE,
    SURF_SCALARS_CLIM_DEFAULT_VALUE,
    SURF_SCALARS_CMAP_DEFAULT_VALUE,
    SURF_SCALARS_DEFAULT_VALUE,
    TEXT_DEFAULT_ANGLE,
    TEXT_DEFAULT_BOUNDING_BOX_HEIGHT,
    TEXT_DEFAULT_BOUNDING_BOX_WIDTH,
    TEXT_DEFAULT_CONTENT,
    TEXT_DEFAULT_FONT,
    TEXT_DEFAULT_FONT_COLOR,
    TEXT_DEFAULT_FONT_OUTLINE_COLOR,
    TEXT_DEFAULT_FONT_OUTLINE_MULTIPLIER,
    TEXT_DEFAULT_FONT_SIZE_MULTIPLIER,
    Tensor,
)
from .elements import (
    ElementBuilder,
    RasterBuilder,
    TextBuilder,
    UnknownBuilder,
    build_raster,
    tile_plot_elements,
)
from .layout import (
    AnnotatedLayout,
    Cell,
    CellLayout,
    GroupSpec,
    grid,
)
from .plot import (
    EdgeLayer,
    Layer,
    NodeLayer,
    _get_hemisphere_parameters,
    _null_auxwriter,
    _null_op,
    hemisphere_slack_fit,
    plot_network_f,
    plot_points_f,
    plot_surf_f,
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
    sanitise,
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


def surf_from_gifti_f(
    left_surf: Union[str, nb.gifti.gifti.GiftiImage],
    right_surf: Union[str, nb.gifti.gifti.GiftiImage],
    left_mask: Optional[Union[str, nb.gifti.gifti.GiftiImage]] = None,
    right_mask: Optional[Union[str, nb.gifti.gifti.GiftiImage]] = None,
    projection: Optional[str] = None,
) -> Tuple[CortexTriSurface, Sequence[str]]:
    if projection is None:
        projection = 'unknown'
    surf = CortexTriSurface.from_gifti(
        left={projection: left_surf},
        right={projection: right_surf},
        left_mask=left_mask,
        right_mask=right_mask,
        projection=projection,
    )
    return surf, (projection,)


def surf_from_freesurfer_f(
    left_surf: Union[str, Tuple[Tensor, Tensor]],
    right_surf: Union[str, Tuple[Tensor, Tensor]],
    projection: Optional[str] = None,
) -> Tuple[CortexTriSurface, Sequence[str]]:
    if projection is None:
        projection = 'unknown'
    if isinstance(left_surf, str):
        left_surf = nb.freesurfer.io.read_geometry(left_surf)
    if isinstance(right_surf, str):
        right_surf = nb.freesurfer.io.read_geometry(right_surf)
    left_surf = {projection: left_surf}
    right_surf = {projection: right_surf}
    surf = CortexTriSurface.from_darrays(
        left=left_surf,
        right=right_surf,
        projection=projection,
    )
    return surf, (projection,)


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


def surf_scalars_from_freesurfer_f(
    surf: CortexTriSurface,
    scalars: str,
    left_morph: Optional[str] = None,
    right_morph: Optional[str] = None,
    surf_scalars: Sequence[str] = (),
    is_masked: bool = False,
    apply_mask: bool = False,
    null_value: Optional[float] = 0.0,
    allow_multihemisphere: bool = True,
    coerce_to_scalar: bool = False,
    plot: bool = True,
) -> Tuple[CortexTriSurface, Sequence[str]]:
    left_array = nb.freesurfer.io.read_morph_data(left_morph)
    right_array = nb.freesurfer.io.read_morph_data(right_morph)
    scalar_names = surf.add_vertex_dataset(
        name=scalars,
        data=None,
        left_data=left_array,
        right_data=right_array,
        left_slice=None,
        right_slice=None,
        default_slices=False,
        is_masked=is_masked,
        apply_mask=apply_mask,
        null_value=null_value,
        select=None,
        exclude=None,
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
    cmap = colors.ListedColormap(colours[start:stop, :3])
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
        cmap = colors.ListedColormap(colours[:, :3])
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
    cmap: Tuple[str, str],
    target: Union[str, Sequence[str]] = ('surf_scalars', 'node'),
):
    cmap_L, cmap_R = cmap
    surf.add_gifti_dataset(
        name=f'cmap_{cmap_name}',
        left_gifti=cmap_L,
        right_gifti=cmap_R,
        is_masked=False,
        apply_mask=True,
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
    scalar_names = surf.scatter_into_parcels(
        data=parcellated,
        parcellation=parcellation_name,
        sink=sink,
    )
    if plot:
        surf_scalars = tuple(list(surf_scalars) + list(scalar_names))
    return surf, surf_scalars


def scatter_into_parcels_f(
    surf: CortexTriSurface,
    scalars: str,
    parcellated: Tensor,
    parcellation_name: str,
    surf_scalars: Sequence[str] = (),
    plot: bool = True,
) -> Tuple[CortexTriSurface, Sequence[str]]:
    scalar_names = surf.scatter_into_parcels(
        data=parcellated,
        parcellation=parcellation_name,
        sink=scalars,
    )
    if plot:
        surf_scalars = tuple(list(surf_scalars) + list(scalar_names))
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
        candidate = params.pop(src, default)
        # TODO: This is bad! Sometimes we might want to overwrite an assigned
        #       default value with None, but this will prevent that.
        #       Unfortunately, the most reasonable solution is accepting an
        #       additional condition when using overlays: if None is a valid
        #       value, then the default must be None. We might get away with
        #       breaking this rule if we can guarantee that the arguments are
        #       not slated for assignment to an overlay using the below
        if default is None or candidate is not None:
            ret[dst] = candidate
        else:
            ret[dst] = default
    return ret, params


def _get_inner_signature(
    paramstr: str,
    inner_f: callable,
    basefunc: callable = Layer.__init__,
    infix: str = '',
) -> inspect.Signature:
    parameters = {
        f'{paramstr}_{k}': v.replace(
            name=f'{paramstr}{infix}_{k}',
            kind=inspect.Parameter.KEYWORD_ONLY,
        )
        for k, v in inspect.signature(basefunc).parameters.items()
        if str(k) not in {'self', 'name', 'hide_subthreshold'}
    }
    inner_signature = inspect.signature(inner_f)
    inner_signature = inner_signature.replace(
        parameters=tuple(
            [
                p for p in inner_signature.parameters.values()
                if p.kind != p.VAR_KEYWORD
            ]
            + list(parameters.values())
        ),
    )
    inner_metadata = getattr(inner_f, '__meta__', {})
    doc_metadata = inner_metadata.get('__doc__', {})
    doc_metadata['subs'] = {
        **doc_metadata.get('subs', {}),
        **{
            f'{paramstr}_{k}': (f'layer_{k}', {'layer_name': paramstr})
            for k in inspect.signature(basefunc).parameters.keys()
            if str(k) not in {'self', 'name', 'hide_subthreshold'}
        },
    }
    inner_metadata['__doc__'] = doc_metadata
    return inner_signature, inner_metadata


def add_surface_overlay_f(
    layer_name: str,
    chains: Sequence[callable],
) -> Tuple[Primitive, inspect.Signature]:
    inner_f = ichain(*chains)(_null_op)
    paramstr = sanitise(layer_name)

    def _add_surface_overlay(
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        surf_scalars_layers = params.pop('surf_scalars_layers', None)
        if surf_scalars_layers is None:
            surf_scalars_layers = ([], [])
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

        params = inner_f(**params)

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
                (f'{paramstr}_cmap', 'cmap', layer_params['cmap']),
                (f'{paramstr}_clim', 'clim', layer_params['clim']),
                (
                    f'{paramstr}_cmap_negative',
                    'cmap_negative',
                    LAYER_CMAP_NEGATIVE_DEFAULT_VALUE,
                ),
                (
                    f'{paramstr}_clim_negative',
                    'clim_negative',
                    LAYER_CLIM_NEGATIVE_DEFAULT_VALUE,
                ),
                (f'{paramstr}_alpha', 'alpha', LAYER_ALPHA_DEFAULT_VALUE),
                (f'{paramstr}_color', 'color', LAYER_COLOR_DEFAULT_VALUE),
                (
                    f'{paramstr}_below_color',
                    'below_color',
                    layer_params['below_color'],
                ),
                (
                    f'{paramstr}_blend_mode',
                    'blend_mode',
                    LAYER_BLEND_MODE_DEFAULT_VALUE,
                ),
                (
                    f'{paramstr}_scalar_bar_style',
                    'scalar_bar_style',
                    {},
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
            scalar_bar_style=layer_params['scalar_bar_style'],
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
            scalar_bar_style=layer_params['scalar_bar_style'],
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

    _add_surface_overlay_p = Primitive(
        name='add_surface_overlay',
        f=_add_surface_overlay,
        output=None,
        forward_unused=True,
    )
    inner_signature, inner_metadata = _get_inner_signature(
        paramstr=paramstr, inner_f=inner_f
    )
    return _add_surface_overlay_p, inner_signature, inner_metadata


def add_points_overlay_f(
    layer_name: str,
    chains: Sequence[callable],
) -> Tuple[Primitive, inspect.Signature]:
    inner_f = ichain(*chains)(_null_op)
    paramstr = sanitise(layer_name)

    def _add_points_overlay(
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        points_scalars_layers = params.pop('points_scalars_layers', None)
        if points_scalars_layers is None:
            points_scalars_layers = []
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

        params = inner_f(**params)

        layer_params, params = _move_params_to_dict(params, (
            ('points_scalars_cmap', 'cmap', POINTS_SCALARS_CMAP_DEFAULT_VALUE),
            ('points_scalars_clim', 'clim', LAYER_CLIM_DEFAULT_VALUE),
            (
                'points_scalars_below_color',
                'below_color',
                LAYER_BELOW_COLOR_DEFAULT_VALUE,
            ),
        ))

        paramstr = sanitise(layer_name)
        layer_params, params = _move_params_to_dict(
            params,
            (
                (f'{paramstr}_cmap', 'cmap', layer_params['cmap']),
                (f'{paramstr}_clim', 'clim', layer_params['clim']),
                (
                    f'{paramstr}_cmap_negative',
                    'cmap_negative',
                    LAYER_CMAP_NEGATIVE_DEFAULT_VALUE,
                ),
                (
                    f'{paramstr}_clim_negative',
                    'clim_negative',
                    LAYER_CLIM_NEGATIVE_DEFAULT_VALUE,
                ),
                (f'{paramstr}_alpha', 'alpha', LAYER_ALPHA_DEFAULT_VALUE),
                (f'{paramstr}_color', 'color', LAYER_COLOR_DEFAULT_VALUE),
                (
                    f'{paramstr}_below_color',
                    'below_color',
                    layer_params['below_color'],
                ),
                (
                    f'{paramstr}_blend_mode',
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

    _add_points_overlay_p = Primitive(
        name='add_points_overlay',
        f=_add_points_overlay,
        output=None,
        forward_unused=True,
    )
    inner_signature, inner_metadata = _get_inner_signature(
        paramstr=paramstr, inner_f=inner_f
    )
    return _add_points_overlay_p, inner_signature, inner_metadata


def add_network_overlay_f(
    layer_name: str,
    chains: Sequence[callable],
) -> Tuple[Primitive, inspect.Signature]:
    inner_f = ichain(*chains)(_null_op)
    paramstr = sanitise(layer_name)

    def _add_network_overlay(
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        network_layers = params.pop('network_layers', None)
        if network_layers is None:
            network_layers = []
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
                (f'{paramstr}_node_cmap', 'cmap', node_params['cmap']),
                (f'{paramstr}_node_clim', 'clim', node_params['clim']),
                (f'{paramstr}_node_color', 'color', node_params['color']),
                (f'{paramstr}_node_alpha', 'alpha', node_params['alpha']),
                (f'{paramstr}_node_radius', 'radius', node_params['radius']),
                (
                    f'{paramstr}_node_radius_range',
                    'radius_range',
                    node_params['radius_range'],
                ),
                (
                    f'{paramstr}_node_cmap_negative',
                    'cmap_negative',
                    LAYER_CMAP_NEGATIVE_DEFAULT_VALUE,
                ),
                (
                    f'{paramstr}_node_clim_negative',
                    'clim_negative',
                    LAYER_CLIM_NEGATIVE_DEFAULT_VALUE,
                ),
                (
                    f'{paramstr}_node_below_color',
                    'below_color',
                    NETWORK_LAYER_BELOW_COLOR_DEFAULT_VALUE,
                ),
            ),
        )
        edge_params, params = _move_params_to_dict(
            params,
            (
                (f'{paramstr}_edge_cmap', 'cmap', edge_params['cmap']),
                (f'{paramstr}_edge_clim', 'clim', edge_params['clim']),
                (f'{paramstr}_edge_color', 'color', edge_params['color']),
                (f'{paramstr}_edge_alpha', 'alpha', edge_params['alpha']),
                (f'{paramstr}_edge_radius', 'radius', edge_params['radius']),
                (
                    f'{paramstr}_edge_radius_range',
                    'radius_range',
                    edge_params['radius_range'],
                ),
                (
                    f'{paramstr}_edge_cmap_negative',
                    'cmap_negative',
                    LAYER_CMAP_NEGATIVE_DEFAULT_VALUE,
                ),
                (
                    f'{paramstr}_edge_clim_negative',
                    'clim_negative',
                    LAYER_CLIM_NEGATIVE_DEFAULT_VALUE,
                ),
                (
                    f'{paramstr}_edge_below_color',
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

    _add_network_overlay_p = Primitive(
        name='add_network_overlay',
        f=_add_network_overlay,
        output=None,
        forward_unused=True,
    )
    inner_signature, inner_metadata = _get_inner_signature(
        paramstr=paramstr,
        inner_f=inner_f,
        basefunc=NodeLayer.__init__,
        infix='_node',
    )
    inner_f.__signature__ = inner_signature
    inner_f.__meta__ = inner_metadata
    inner_signature, inner_metadata = _get_inner_signature(
        paramstr=paramstr,
        inner_f=inner_f,
        basefunc=EdgeLayer.__init__,
        infix='_edge',
    )
    return _add_network_overlay_p, inner_signature, inner_metadata


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
    return coor, lh_mask, surf, surf_projection


def add_node_variable_f(
    name: str = "node",
    val: Union[pd.DataFrame, str] = None,
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


# Note: For camera primitives, we currently carry forward a `close_plotter`
#       parameter, which specifies whether to close the plotter after the
#       primitive is executed, when another downstream primitive like
#       `plot_to_image` is called. It would probably be better to either
#       handling the close operation in a base postprocessor that is
#       pre-transformed into the postprocessor chain, or to automatically
#       splice the close parameter into the postprocessor chain.
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
    close_plotter: bool = True,
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
    return plotter, views, hemispheres, close_plotter


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
    close_plotter: bool = True,
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
    return plotter, closest_poles, hemispheres, close_plotter


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
    close_plotter: bool = True,
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
    return plotter, views, hemispheres, close_plotter


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
    close_plotter: bool = True,
) -> Mapping:
    views_ortho = views_focused = views_planar = []
    if n_ortho > 0:
        _, views_ortho, _, _ = closest_ortho_camera_f(
            surf=surf,
            hemispheres=hemispheres,
            surf_scalars=surf_scalars,
            surf_projection=surf_projection,
            n_ortho=n_ortho,
            plotter=plotter,
        )
    if focus is not None:
        _, views_focused, _, _ = scalar_focus_camera_f(
            surf=surf,
            hemispheres=hemispheres,
            surf_scalars=surf_scalars,
            surf_projection=surf_projection,
            kind=focus,
            plotter=plotter,
        )
    if n_angles > 0:
        _, views_planar, _, _ = planar_sweep_camera_f(
            surf=surf,
            hemispheres=hemispheres,
            n_steps=n_angles,
            initial=initial_angle,
            normal=normal_vector,
            plotter=plotter,
        )
    views = tuple(views_ortho + views_focused + views_planar)
    return plotter, views, hemispheres, close_plotter


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
    window_size: Tuple[int, int] = DEFAULT_WINDOW_SIZE,
    hemispheres: Sequence[Literal['left', 'right', 'both']] = None,
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
    window_size: Tuple[int, int] = DEFAULT_WINDOW_SIZE,
    n_scenes: int = 1,
    close_plotter: bool = True,
) -> Tuple[Tensor]:
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
    window_size: Tuple[int, int] = DEFAULT_WINDOW_SIZE,
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
    window_size: Tuple[int, int] = DEFAULT_WINDOW_SIZE,
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
    output_dir: str,
    layout_kernel: CellLayout = Cell(),
    sort_by: Optional[Sequence[str]] = None,
    group_spec: Optional[Sequence[GroupSpec]] = None,
    fname_spec: Optional[str] = None,
    suffix: Optional[str] = None,
    extension: str = 'svg',
    padding: int = 0,
    canvas_color: Any = (1, 1, 1, 1),
) -> None: # Union[Tuple[Image.Image], Image.Image]:
    # Helper function: write a single snapshot group to a file.
    def writer(snapshot_group, fname):
        _canvas_color = colors.to_hex(canvas_color)
        bg = svg.Rect(
            id='background',
            height=canvas_size[1],
            width=canvas_size[0],
            fill=_canvas_color,
        )
        groups = [
            svg.G(
                id='background',
                elements=[bg],
                transform=[],
            )
        ]
        for i, (cimg, cmeta) in snapshot_group:
            panel = cells[i]
            if not isinstance(cimg, svg.SVG):
                builder = RasterBuilder(
                    content=cimg,
                    bounding_box_height=panel.cell_dim[1],
                    bounding_box_width=panel.cell_dim[0],
                    fmt='png',
                )
                cgroup = build_raster(**builder).elements[0]
            else:
                cgroup = cimg.elements[0]
            ctransform = cgroup.transform or []
            transform = [svg.Translate(*panel.cell_loc)]
            cgroup = svg.G(
                id=cgroup.id,
                elements=cgroup.elements,
                transform=ctransform + transform,
            )
            groups.append(cgroup)
        canvas = svg.SVG(
            id='canvas',
            elements=groups,
            height=canvas_size[1],
            width=canvas_size[0],
        )
        with open(fname, 'w') as f:
            f.write(canvas.__str__())

    # Step 0: Configure inputs
    try:
        n_scenes = len(snapshots)
    except TypeError:
        n_scenes = 1
        snapshots = (snapshots,)
    if sort_by is not None:

        def sort_func(snapshot):
            _, cmeta = snapshot
            return tuple(cmeta[cfield] for cfield in sort_by)

        snapshots = sorted(snapshots, key=sort_func)

    meta = [meta for (_, meta) in snapshots]

    # Step 1: Apply any grouping modulators to the provided layout template.
    if group_spec:
        group_layouts, group_spec, bp, nb = tuple(
            zip(*[spec(meta) for spec in group_spec])
        )
        bp, nb = bp[0], nb[0]
        bp_factor = 1
        if group_layouts[0].layout.split_orientation == 'v':
            if group_spec[0].n_rows == 1:
                for spec in group_spec[1:]:
                    if spec.order == 'row' and spec.n_rows > 1:
                        break
                    bp_factor *= spec.n_cols
                    if spec.n_rows > 1:
                        break
        else:
            if group_spec[0].n_cols == 1:
                for spec in group_spec[1:]:
                    if spec.order == 'col' and spec.n_cols > 1:
                        break
                    bp_factor *= spec.n_rows
                    if spec.n_cols > 1:
                        break

        if bp is not None:
            bp = (bp + 1) * bp_factor - 1
        group_layouts = reduce(
            (lambda x, y: x * y),
            group_layouts,
        )
        if not hasattr(layout_kernel, 'annotations'):
            layout_kernel = layout_kernel.annotate({})
        layout = group_layouts * layout_kernel
    else:
        layout = layout_kernel
        bp = None
        nb = 0

    # Step 2: Determine the layout of cells on the canvas
    # Case a: layout is an annotated CellLayout. Here we assume that the
    #         provided layout specifies the layout of all pages together.
    if getattr(layout, 'annotations', None) is not None:
        # It's an annotated layout, so we'll match the annotations
        # to the snapshot metadata to 'semantically' assign snapshots
        # to cells.
        layout, cell_indices = layout.match_and_assign_all(
            queries=meta,
            force_unmatched=True,
            # drop={'elements'},
        )
        assert len(cell_indices) == len(meta)
        index_map = dict(zip(cell_indices, snapshots))
        # Now, we use the breakpoint to split the layout into pages.
        pages = ()
        snapshots = ()
        maximum = 0
        nb_seen = 0
        terminal = False
        while not terminal:
            minimum = maximum
            if nb_seen == nb:
                terminal = True
            try:
                # Here's a dumb hack to skip to the except block if we're on
                # the last page.
                if terminal:
                    raise IndexError
                left, right = layout @ (bp - 1)
                maximum += left.count
                pages += (left,)
                layout = right
            except (IndexError, TypeError):
                maximum += layout.count
                pages += (layout,)
                terminal = True
            nb_seen += 1
            snapshots += ([
                (i - minimum, index_map.get(i, None))
                for i in range(minimum, maximum)
                if index_map.get(i, None) is not None
            ],)
        panels_per_page = pages[0].count
        n_panels = panels_per_page
        n_pages = len(pages)
    # Case b: layout is a regular CellLayout. Here we assume that the provided
    #         layout specifies the layout of a single page, so we don't need
    #         to handle the layout splitting.
    else:
        panels_per_page = layout.count
        cell_indices = list(range(panels_per_page))
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
        pages = tuple(layout.copy() for _ in range(n_pages))
        snapshots = tuple(
            list(zip(cell_indices, snapshot_group))
            for snapshot_group in snapshots
        )

    # Step 3: Write to the canvas
    for i, (page, snapshot_group) in enumerate(zip(pages, snapshots)):
        cells = list(page.partition(*canvas_size, padding=padding))
        # Build page-level metadata. This includes the page number as well as
        # any metadata that is constant across all snapshots on the page.
        page = f'{i + 1:{len(str(n_pages))}d}'
        page_entities = [meta for (_, (_, meta)) in snapshot_group]
        page_entities = _seq_to_dict(page_entities, merge_type='union')
        page_entities = {
            k : set(v) - {None} for k, v in page_entities.items()
            if k != 'elements'
        }
        page_entities = {
            k : list(v)[0] for k, v in page_entities.items() if len(v) == 1
        }
        page_entities['page'] = page
        # Time to handle any additional plot elements that need to be added
        # to the canvas. This includes things like colorbars, legends, etc.
        # We can only semantically add these if we have an annotated layout.
        elements_asgt = {}
        if hasattr(layout, 'annotations'):
            elements_fields = list(
                set(
                    sum(
                        [
                            list(cmeta.get('elements', {}).keys())
                            for (_, (_, cmeta)) in snapshot_group
                        ],
                        [],
                    )
                )
            )
            if elements_fields:
                queries = [{'elements': cfield} for cfield in elements_fields]
                # Each assignment should not "fill" a slot since we might want
                # to assign multiple elements fields to a single cell. Because
                # layout is not mutable, this shouldn't be a problem.
                elements_asgt = [
                    layout.match_and_assign(query=q) for q in queries
                ]
                _, elements_asgt = zip(*elements_asgt)
                elements_asgt = {
                    k: v
                    for k, v in zip(elements_fields, elements_asgt)
                    if v < len(cells)
                }
                elements = [
                    cmeta['elements'] for (_, (_, cmeta)) in snapshot_group
                ]
                elements = _seq_to_dict(elements, merge_type='union')
        for cell_idx in set(elements_asgt.values()):
            if cell_idx >= len(cells):
                continue
            elements_cell = [
                k for k, v in elements_asgt.items() if v == cell_idx
            ]
            builders_cell = list(chain(*[
                elements[k] for k in elements_cell
            ]))
            cellimg = tile_plot_elements(
                builders=builders_cell,
                max_dimension=cells[cell_idx].cell_dim,
            )
            snapshot_group.append((cell_idx, (cellimg, {})))

        write_f(
            writer=writer,
            argument=snapshot_group,
            entities=page_entities,
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
    layout_kernel: CellLayout = Cell(),
    sort_by: Optional[Sequence[str]] = None,
    group_spec: Optional[Sequence[GroupSpec]] = None,
    annotations: Optional[Mapping[int, Mapping]] = None,
    fname_spec: Optional[str] = None,
    suffix: Optional[str] = None,
    extension: str = 'svg',
    order: Literal['row', 'column'] = 'row',
    padding: int = 0,
    canvas_color: Any = (1, 1, 1, 1),
) -> None:
    layout = grid(
        n_rows=n_rows,
        n_cols=n_cols,
        order=order,
        kernel=lambda: layout_kernel,
    )
    if annotations is not None:
        layout = AnnotatedLayout(layout, annotations=annotations)
    save_figure_f(
        snapshots=snapshots,
        canvas_size=canvas_size,
        output_dir=output_dir,
        layout_kernel=layout,
        sort_by=sort_by,
        group_spec=group_spec,
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


def svg_element_f(
    name: str,
    src_file: str,
    height: Optional[int] = None,
    width: Optional[int] = None,
    height_mm: Optional[int] = None,
    width_mm: Optional[int] = None,
    priority: int = 0,
    elements: Optional[Mapping[str, Sequence[ElementBuilder]]] = None,
) -> Mapping[str, Sequence[ElementBuilder]]:
    if elements is None:
        elements = {}
    if height is None:
        height = height_mm / PXMM
    if width is None:
        width = width_mm / PXMM
    with open(src_file, encoding='utf-8') as f:
        content = f.read()
    elements[name] = (
        UnknownBuilder(
            content=content,
            height=height,
            width=width,
            priority=priority,
        ),
    )
    return elements


def pyplot_element_f(
    name: str,
    plotter: callable,
    priority: int = 0,
    elements: Optional[Mapping[str, Sequence[ElementBuilder]]] = None,
    plotter_params: Mapping[str, Any] = None,
) -> Mapping[str, Sequence[ElementBuilder]]:
    if elements is None:
        elements = {}
    plotter_params = plotter_params or {}
    fig = plotter(**plotter_params)
    try:
        width_in, height_in = fig.get_size_inches()
    except AttributeError: # some seaborn figure-level functions return
                           # a FacetGrid instead of a Figure
         width_in, height_in = fig.figure.get_size_inches()
    height = height_in * EMPIRICAL_DPI
    width = width_in * EMPIRICAL_DPI

    buffer = StringIO()
    fig.savefig(buffer, format='svg')
    buffer.seek(0)
    content = buffer.read()
    elements[name] = (
        UnknownBuilder(
            content=content,
            height=height,
            width=width,
            priority=priority,
        ),
    )
    return elements


def text_element_f(
    name: str,
    content: str = TEXT_DEFAULT_CONTENT,
    font: str = TEXT_DEFAULT_FONT,
    font_size_multiplier: int = TEXT_DEFAULT_FONT_SIZE_MULTIPLIER,
    font_color: Any = TEXT_DEFAULT_FONT_COLOR,
    font_outline_color: Any = TEXT_DEFAULT_FONT_OUTLINE_COLOR,
    font_outline_multiplier: float = TEXT_DEFAULT_FONT_OUTLINE_MULTIPLIER,
    bounding_box_width: Optional[int] = TEXT_DEFAULT_BOUNDING_BOX_WIDTH,
    bounding_box_height: Optional[int] = TEXT_DEFAULT_BOUNDING_BOX_HEIGHT,
    angle: int = TEXT_DEFAULT_ANGLE,
    priority: int = 0,
    elements: Optional[Mapping[str, Sequence[ElementBuilder]]] = None,
) -> Mapping[str, Sequence[ElementBuilder]]:
    if elements is None:
        elements = {}
    elements[name] = (
        TextBuilder(
            content=content,
            font=font,
            font_size_multiplier=font_size_multiplier,
            font_color=font_color,
            font_outline_color=font_outline_color,
            font_outline_multiplier=font_outline_multiplier,
            bounding_box_width=bounding_box_width,
            bounding_box_height=bounding_box_height,
            angle=angle,
            priority=priority,
        ),
    )
    return elements


def splice_on(f):
    return splice_on_orig(
        f, kwonly_only=True, strict_emulation=False, allow_variadic=True
    )


@splice_on(plot_surf_f)
@splice_on(plot_points_f)
@splice_on(plot_network_f)
@splice_on(hemisphere_slack_fit)
def automap_unified_plotter_f(
    *,
    hemisphere: Optional[Literal['left', 'right']] = None,
    off_screen: bool = True,
    copy_actors: bool = False,
    theme: Optional[Any] = None,
    window_size: Tuple[int, int] = DEFAULT_WINDOW_SIZE,
    use_single_plotter: bool = True,
    sbprocessor: Optional[callable] = None,
    empty_builders: bool = False,
    postprocessors: Optional[
        Sequence[Mapping[str, Tuple[callable, callable]]]
    ] = None,
    elements: Optional[Sequence] = None,
    map_spec: Optional[Sequence] = None,
    **params,
) -> Optional[pv.Plotter]:
    params = {**params, **locals()}
    _ = params.pop('params')
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

    sbprocessor = params.pop('sbprocessor', None)

    postprocessors = params.pop('postprocessors', None)
    postprocessors = postprocessors or {'plotter': (None, None)}
    postprocessor_names, postprocessors = zip(*postprocessors.items())
    postprocessors, auxwriters = zip(*postprocessors)

    builders = params.pop('elements', None)
    if builders is None:
        builders = {}

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
            sbprocessor=sbprocessor,
            postprocessors=postprocessors,
            return_builders=True,
        )
        for i in range(n_replicates)
    ]
    output, elements = zip(*output)
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
    # TODO: we're sticking the 2D plot elements into the metadata here as a
    #       hack to access them later, but this is not really where they
    #       belong.
    for cmeta, celements in zip(metadata, elements):
        len_cmeta = len(next(iter(cmeta[0].values())))
        meta_asgts = [
            {k: v[i] for k, v in cmeta[0].items()} for i in range(len_cmeta)
        ]
        celements = {**builders, **celements}
        celements = {k: v for k, v in celements.items() if v is not None}
        celements = [
            {
                k: tuple(
                    c.eval_spec(asgt)
                    for c in v
                )
                for k, v in celements.items()
            }
            for asgt in meta_asgts
        ]
        cmeta[0]['elements'] = celements
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


surf_from_gifti_p = Primitive(
    surf_from_gifti_f,
    'surf_from_gifti',
    output=('surf', 'surf_projection'),
    forward_unused=True,
)


surf_from_freesurfer_p = Primitive(
    surf_from_freesurfer_f,
    'surf_from_freesurfer',
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


surf_scalars_from_freesurfer_p = Primitive(
    surf_scalars_from_freesurfer_f,
    'surf_scalars_from_freesurfer',
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
    output=('prim', 'signature', 'metadata'),
    forward_unused=True,
)


add_points_overlay_p = Primitive(
    add_points_overlay_f,
    'add_points_overlay',
    output=('prim', 'signature', 'metadata'),
    forward_unused=True,
)


add_network_overlay_p = Primitive(
    add_network_overlay_f,
    'add_network_overlay',
    output=('prim', 'signature', 'metadata'),
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
    output=('node_coor', 'lh_mask', 'surf', 'surf_projection'),
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
    output=('plotter', 'views', 'hemispheres', 'close_plotter'),
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
    output=('plotter', 'views', 'hemispheres', 'close_plotter'),
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
    output=('plotter', 'views', 'hemispheres', 'close_plotter'),
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
    output=('plotter', 'views', 'hemispheres', 'close_plotter'),
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


svg_element_p = Primitive(
    svg_element_f,
    'svg_element',
    output=('elements',),
    forward_unused=True,
)


pyplot_element_p = Primitive(
    pyplot_element_f,
    'pyplot_element',
    output=('elements',),
    forward_unused=True,
)


text_element_p = Primitive(
    text_element_f,
    'text_element',
    output=('elements',),
    forward_unused=True,
)


automap_unified_plotter_p = Primitive(
    automap_unified_plotter_f,
    'automap_unified_plotter',
    output=None,
    forward_unused=True,
)
