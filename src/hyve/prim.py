# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Primitive functional atoms
~~~~~~~~~~~~~~~~~~~~~~~~~~
Atomic functional primitives for building more complex functions.
"""
from io import StringIO
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
)
from conveyant import (
    SanitisedFunctionWrapper as F,
)
from conveyant.compositors import (
    _dict_to_seq,
    direct_compositor,
    reversed_args_compositor
)
from conveyant.replicate import _flatten, _flatten_to_depth, replicate
from matplotlib.colors import ListedColormap

from .const import Tensor
from .plot import _null_auxwriter, plotted_entities, unified_plotter
from .surf import CortexTriSurface
from .util import (
    auto_focus,
    cortex_cameras,
    filter_adjacency_data,
    filter_node_data,
    format_position_as_string,
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


def scalars_from_cifti_f(
    surf: CortexTriSurface,
    scalars: str,
    cifti: nb.Cifti2Image,
    surf_scalars: Sequence[str] = (),
    is_masked: bool = True,
    apply_mask: bool = False,
    null_value: Optional[float] = 0.0,
    plot: bool = False,
) -> Tuple[CortexTriSurface, Sequence[str]]:
    surf.add_cifti_dataset(
        name=scalars,
        cifti=cifti,
        is_masked=is_masked,
        apply_mask=apply_mask,
        null_value=null_value,
    )
    if plot:
        surf_scalars = tuple(list(surf_scalars) + [scalars])
    return surf, surf_scalars


def scalars_from_gifti_f(
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
    plot: bool = False,
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
    )
    if plot:
        surf_scalars = tuple(list(surf_scalars) + list(scalar_names))
    return surf, surf_scalars


def scalars_from_nifti_f(
    nifti: nb.Nifti1Image,
    threshold: float = 0.0
) -> Mapping:
    if not isinstance(nifti, nb.Nifti1Image):
        nifti = nb.load(nifti)
    vol = nifti.get_fdata()
    loc = np.where(vol > threshold)

    vol_scalars = vol[loc]
    vol_coor = np.stack(loc)
    vol_coor = (nifti.affine @ np.concatenate(
        (vol_coor, np.ones((1, vol_coor.shape[-1])))
    ))[:3].T
    vol_voxdim = nifti.header.get_zooms()
    return vol_scalars, vol_coor, vol_voxdim


def resample_to_surface_f(
    surf: CortexTriSurface,
    scalars: str,
    nifti: nb.Nifti1Image,
    f_resample: callable,
    surf_scalars: Sequence[str] = (),
    null_value: Optional[float] = 0.0,
    select: Optional[Sequence[int]] = None,
    exclude: Optional[Sequence[int]] = None,
    plot: bool = False,
) -> Mapping:
    left, right = f_resample(nifti)
    scalar_names = surf.add_gifti_dataset(
        name=scalars,
        left_gifti=left,
        right_gifti=right,
        is_masked=False,
        apply_mask=True,
        null_value=null_value,
        select=select,
        exclude=exclude,
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
    if 'node' in target:
        ret['node_cmap'] = cmap
        ret['node_clim'] = clim
    return ret


def parcellate_scalars_f(
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


def add_node_variable_f(
    name: str = "node",
    val: Union[np.ndarray, str] = None,
    threshold: Union[float, int] = 0.0,
    percent_threshold: bool = False,
    topk_threshold: bool = False,
    absolute: bool = True,
    node_selection: Optional[np.ndarray] = None,
    incident_edge_selection: Optional[np.ndarray] = None,
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
    adj: Union[np.ndarray, str] = None,
    threshold: float = 0.0,
    percent_threshold: bool = False,
    topk_threshold_nodewise: bool = False,
    absolute: bool = True,
    incident_node_selection: Optional[np.ndarray] = None,
    connected_node_selection: Optional[np.ndarray] = None,
    edge_selection: Optional[np.ndarray] = None,
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
    if hemisphere is None:
        metadata['hemisphere'] = ['both']
        metadata['view'] = ['focusedleft', 'focusedright']
    elif len(hemisphere) == 1 and hemisphere[0] != 'both':
        metadata['view'] = ['focused']
    else:
        metadata['hemisphere'] = ['both']
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
    if hemisphere is None:
        metadata['hemisphere'] = ['both']
        metadata['view'] = ['ortholeft', 'orthoright']
    elif len(hemisphere) == 1 and hemisphere[0] != 'both':
        metadata['view'] = ['ortho']
    else:
        metadata['hemisphere'] = ['both']
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
    if hemisphere is None:
        metadata['hemisphere'] = ['both']
        metadata['view'] = ['planarleft', 'planarright']
    elif len(hemisphere) == 1 and hemisphere[0] != 'both':
        metadata['view'] = ['planar']
    else:
        metadata['hemisphere'] = ['both']
        metadata['view'] = ['planarleft', 'planarright']
    mapper = replicate(
        spec=(['view', 'index'], 'hemisphere'),
        broadcast_out_of_spec=True,
    )
    return mapper(**metadata)


def plot_to_image_f(
    plotter: pv.Plotter,
    views: Union[Sequence, Literal['__default__']] = '__default__',
    window_size: Tuple[int, int] = (1920, 1080),
    hemispheres: Sequence[Literal['left', 'right', 'both']] = None,
    plot_scalar_bar: bool = False,
) -> Tuple[np.ndarray]:
    if len(hemispheres) == 1:
        hemisphere = hemispheres[0]
    elif tuple(hemispheres) == ('left', 'right'):
        hemisphere = 'both'
    else:
        hemisphere = hemispheres
    if views == '__default__':
        views = set_default_views(hemisphere)
    ret = []
    if not plot_scalar_bar:
        try:
            plotter.remove_scalar_bar()
        except IndexError:
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
    plotter.close()
    return tuple(ret)


def plot_to_image_aux_f(
    metadata: Mapping[str, Sequence[str]],
    hemisphere: Optional[Sequence[Literal['left', 'right', 'both']]] = None,
    views: Union[Sequence, Literal['__default__']] = '__default__',
) -> Mapping[str, Sequence[str]]:
    if views == '__default__':
        if hemisphere is None:
            hemisphere = 'both'
        elif len(hemisphere) == 1:
            hemisphere = hemisphere[0]
        else:
            hemisphere = 'both'
        views = set_default_views(hemisphere)
    views = [format_position_as_string(cpos) for cpos in views]
    mapper = replicate(spec=['view'], broadcast_out_of_spec=True)
    return mapper(**metadata, view=views)


def plot_to_html_buffer_f(
    plotter: pv.Plotter,
    window_size: Tuple[int, int] = (1920, 1080),
) -> StringIO:
    plotter.window_size = window_size
    return plotter.export_html(filename=None)


def save_screenshots_f(
    screenshots: Sequence[Tuple[np.ndarray, Mapping[str, str]]],
    output_dir: str,
    fname_spec: Optional[str] = None,
    suffix: Optional[str] = None,
    extension: str = 'png',
) -> None:
    def writer(img, fname):
        from PIL import Image
        img = Image.fromarray(img)
        img.save(fname)

    for cimg, cmeta in screenshots:
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
    writer(argument, fname)


def automap_unified_plotter_f(
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
    postprocessors: Optional[
        Sequence[Mapping[str, Tuple[callable, callable]]]
    ] = None,
    map_spec: Optional[Sequence] = None,
) -> Optional[pv.Plotter]:
    params = locals()
    map_spec = map_spec or [
        'surf_projection',
        'surf_scalars',
        'vol_scalars',
        ('node_values', 'node_coor', 'node_parcel_scalars', 'edge_values'),
        'hemisphere',
    ]
    params.pop('map_spec')
    mapper = replicate(spec=map_spec, weave_type='maximal')

    postprocessors = params.pop('postprocessors', None)
    postprocessors = postprocessors or {'plotter': None}
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


scalars_from_cifti_p = Primitive(
    scalars_from_cifti_f,
    'scalars_from_cifti',
    output=('surf', 'surf_scalars'),
    forward_unused=True,
)


scalars_from_gifti_p = Primitive(
    scalars_from_gifti_f,
    'scalars_from_gifti',
    output=('surf', 'surf_scalars'),
    forward_unused=True,
)


scalars_from_nifti_p = Primitive(
    scalars_from_nifti_f,
    'scalars_from_nifti',
    output=('vol_scalars', 'vol_coor', 'vol_voxdim'),
    forward_unused=True,
)


resample_to_surface_p = Primitive(
    resample_to_surface_f,
    'resample_to_surface',
    output=('surf', 'surf_scalars'),
    forward_unused=True,
)


parcellate_colormap_p = Primitive(
    parcellate_colormap_f,
    'parcellate_colormap',
    output=None,
    forward_unused=True,
)


parcellate_scalars_p = Primitive(
    parcellate_scalars_f,
    'parcellate_scalars',
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


save_screenshots_p = Primitive(
    save_screenshots_f,
    'save_screenshots',
    output=(),
    forward_unused=True,
)


save_html_p = Primitive(
    save_html_f,
    'save_html',
    output=(),
    forward_unused=True,
)


automap_unified_plotter_p = Primitive(
    automap_unified_plotter_f,
    'automap_unified_plotter',
    output=None,
    forward_unused=True,
)
