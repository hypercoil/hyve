# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Primitive functional atoms
~~~~~~~~~~~~~~~~~~~~~~~~~~
Atomic functional primitives for building more complex functions.
"""
import dataclasses
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
    Primitive,
    SanitisedFunctionWrapper as F,
)
from conveyant.compositors import _dict_to_seq
from conveyant.replicate import _flatten, _flatten_to_depth, replicate
from matplotlib.colors import ListedColormap

from .const import Tensor
from .plot import unified_plotter, plotted_entities, _null_auxwriter
from .surf import CortexTriSurface
from .util import (
    cortex_cameras,
    format_position_as_string,
)


def surf_from_archive_f(
    template: str,
    load_mask: bool,
    projections: Sequence[str],
    archives: Mapping[str, Callable],
):
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
    scalars_to_plot: Sequence[str] = (),
    is_masked: bool = True,
    apply_mask: bool = False,
    null_value: Optional[float] = 0.0,
    plot: bool = False,
) -> Mapping:
    surf.add_cifti_dataset(
        name=scalars,
        cifti=cifti,
        is_masked=is_masked,
        apply_mask=apply_mask,
        null_value=null_value,
    )
    if plot:
        scalars_to_plot = tuple(list(scalars_to_plot) + [scalars])
    return surf, scalars_to_plot


def scalars_from_gifti_f(
    surf: CortexTriSurface,
    scalars: str,
    left_gifti: Optional[nb.gifti.GiftiImage] = None,
    right_gifti: Optional[nb.gifti.GiftiImage] = None,
    scalars_to_plot: Sequence[str] = (),
    is_masked: bool = True,
    apply_mask: bool = False,
    null_value: Optional[float] = 0.0,
    select: Optional[Sequence[int]] = None,
    exclude: Optional[Sequence[int]] = None,
    plot: bool = False,
) -> Mapping:
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
        scalars_to_plot = tuple(list(scalars_to_plot) + list(scalar_names))
    return surf, scalars_to_plot


def resample_to_surface_f(
    surf: CortexTriSurface,
    scalars: str,
    nifti: nb.Nifti1Image,
    f_resample: callable,
    scalars_to_plot: Sequence[str] = (),
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
        scalars_to_plot = tuple(list(scalars_to_plot) + list(scalar_names))
    return surf, scalars_to_plot


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

    return (
        surf,
        (cmap_left, cmap_right),
        (clim_left, clim_right),
        cmap,
        clim,
    )


def parcellate_scalars_f(
    surf: CortexTriSurface,
    scalars: str,
    sink: str,
    parcellation_name: str,
    scalars_to_plot: Optional[Sequence[str]] = None,
    plot: bool = True,
) -> Mapping:
    parcellated = surf.parcellate_vertex_dataset(
        name=scalars,
        parcellation=parcellation_name
    )
    surf.scatter_into_parcels(
        data=parcellated,
        parcellation=parcellation_name,
        sink=sink,
    )
    if plot:
        scalars_to_plot = tuple(list(scalars_to_plot) + [sink])
    return surf, scalars_to_plot


def scatter_into_parcels_f(
    surf: CortexTriSurface,
    scalars: str,
    parcellated: Tensor,
    parcellation_name: str,
    scalars_to_plot: Optional[Sequence[str]] = None,
    plot: bool = True,
) -> Mapping:
    surf.scatter_into_parcels(
        data=parcellated,
        parcellation=parcellation_name,
        sink=scalars,
    )
    if plot:
        scalars_to_plot = tuple(list(scalars_to_plot) + [scalars])
    return surf, scalars_to_plot


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


def plot_to_image_f(
    plotter: pv.Plotter,
    views: Sequence = (
        'medial',
        'lateral',
        'dorsal',
        'ventral',
        'anterior',
        'posterior',
    ),
    window_size: Tuple[int, int] = (1920, 1080),
    hemisphere: Optional[Literal['left', 'right', 'both']] = None,
    plot_scalar_bar: bool = False,
) -> Tuple[np.ndarray]:
    screenshot = [True] * len(views)
    ret = []
    if not plot_scalar_bar:
        try:
            plotter.remove_scalar_bar()
        except IndexError:
            pass
    for cpos, fname in zip(views, screenshot):
        plotter.camera.zoom('tight')
        plotter.show(
            cpos=cortex_cameras(cpos, plotter=plotter, hemisphere=hemisphere),
            auto_close=False,
        )
        img = plotter.screenshot(
            fname,
            window_size=window_size,
            return_img=True,
        )
        ret.append(img)
    plotter.close()
    return tuple(ret)


def plot_to_image_aux_f(
    metadata: Mapping[str, Sequence[str]],
    views: Sequence = (
        'medial',
        'lateral',
        'dorsal',
        'ventral',
        'anterior',
        'posterior',
    ),
) -> Mapping[str, Sequence[str]]:
    views = [format_position_as_string(cpos) for cpos in views]
    mapper = replicate(spec=['view'], broadcast_out_of_spec=True)
    return mapper(**metadata, view=views)


def plot_to_html_f(
    plotter: pv.Plotter,
    filename: str,
    backend: str = 'panel',
) -> str:
    plotter.export_html(filename, backend=backend)
    return filename


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
    mapper = replicate(spec=map_spec)

    postprocessors = params.pop('postprocessors', None)
    postprocessors = postprocessors or {'plotter': None}
    postprocessor_names, postprocessors = zip(*postprocessors.items())
    postprocessors, auxwriters = zip(*postprocessors)

    repl_params = mapper(**params)
    repl_vars = set(_flatten(map_spec))
    repl_params = {k: v for k, v in repl_params.items() if k in repl_vars}
    other_params = {k: v for k, v in params.items() if k not in repl_vars}
    params = {**other_params, **repl_params}
    n_replicates = max([len(v) for v in repl_params.values()])

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


resample_to_surface_p = Primitive(
    resample_to_surface_f,
    'resample_to_surface',
    output=('surf', 'surf_scalars'),
    forward_unused=True,
)


parcellate_colormap_p = Primitive(
    parcellate_colormap_f,
    'parcellate_colormap',
    output=(
        'surf',
        'surf_scalars_cmap',
        'surf_scalars_clim',
        'node_cmap',
        'node_clim',
    ),
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


add_postprocessor_p = Primitive(
    add_postprocessor_f,
    'add_postprocessor',
    output=('postprocessors',),
    forward_unused=True,
)


plot_to_image_aux_p = Primitive(
    plot_to_image_aux_f,
    'plot_to_image_aux',
    output=None,
    forward_unused=False,
)


plot_to_html_p = Primitive(
    plot_to_html_f,
    'plot_to_html',
    output=('html',),
    forward_unused=True,
)


save_screenshots_p = Primitive(
    save_screenshots_f,
    'save_screenshots',
    output=(),
    forward_unused=True,
)


automap_unified_plotter_p = Primitive(
    automap_unified_plotter_f,
    'automap_unified_plotter',
    output=None,
    forward_unused=True,
)
