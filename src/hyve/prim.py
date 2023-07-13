# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Primitive functional atoms
~~~~~~~~~~~~~~~~~~~~~~~~~~
Atomic functional primitives for building more complex functions.
"""
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

from conveyant import Primitive
from conveyant.replicate import replicate, _flatten
from .plot import unified_plotter
from .surf import CortexTriSurface
from .util import (
    format_position_as_string,
    cortex_cameras,
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
            return surf
        except Exception as e:
            continue
    raise ValueError(
        f"Could not load {template} with projections {projections} "
        f"from any of {tuple(archives.keys())}."
    )


def resample_to_surface_f(
    nii: nb.Nifti1Image,
    surf: CortexTriSurface,
    scalars: str,
    f_resample: callable,
    scalars_to_plot: Optional[Sequence[str]] = None,
    null_value: Optional[float] = 0.,
    select: Optional[Sequence[int]] = None,
    exclude: Optional[Sequence[int]] = None,
    plot: bool = False,
) -> Mapping:
    left, right = f_resample(nii)
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
        scalars_to_plot = tuple(
            list(scalars_to_plot) + list(scalar_names)
        )
    return surf, scalars_to_plot


def plot_to_image_f(
    p: pv.Plotter,
    views: Sequence = (
        'medial',
        'lateral',
        'dorsal',
        'ventral',
        'anterior',
        'posterior',
    ),
    window_size: Tuple[int, int] = (1920, 1080),
    basename: Optional[str] = None,
    hemisphere: Optional[Literal['left', 'right', 'both']] = None,
) -> Tuple[np.ndarray]:
    if basename is None:
        screenshot = [True] * len(views)
    else:
        screenshot = [
            f'{basename}_{format_position_as_string(cpos)}.png'
            for cpos in views
        ]
    ret = []
    try:
        p.remove_scalar_bar()
    except IndexError:
        pass
    for cpos, fname in zip(views, screenshot):
        p.camera.zoom('tight')
        p.show(
            cpos=cortex_cameras(cpos, plotter=p, hemisphere=hemisphere),
            auto_close=False,
        )
        img = p.screenshot(fname, window_size=window_size, return_img=True)
        ret.append(img)
    p.close()
    return tuple(ret)


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
    # views: Sequence = (),
    # return_plotter: bool = False,
    # return_screenshot: bool = True,
    # return_html: bool = False,
    map_spec: Optional[Sequence] = None,
) -> Optional[pv.Plotter]:
    map_spec = map_spec or [
        'surf_projection',
        'surf_scalars',
        'vol_scalars',
        ('node_values', 'node_coor', 'node_parcel_scalars', 'edge_values'),
        'hemisphere',
    ]
    params = locals()
    params.pop('map_spec')
    mapper = replicate(spec=map_spec)
    repl_params = mapper(**params)
    repl_vars = set(_flatten(map_spec))
    repl_params = {k: v for k, v in repl_params.items() if k in repl_vars}
    other_params = {k: v for k, v in params.items() if k not in repl_vars}
    params = {**other_params, **repl_params}
    n_replicates = max([len(v) for v in repl_params.values()])
    p = [
        unified_plotter(
            **{
                k: (v[i % len(v)] if k in repl_vars else v)
                for k, v in params.items()
            }
        ) for i in range(n_replicates)
    ]
    # for i, _p in enumerate(p):
    #     _p.show()

    return p


surf_from_archive_p = Primitive(
    surf_from_archive_f,
    'surf_from_archive',
    output=('surf',),
    forward_unused=True,
)


resample_to_surface_p = Primitive(
    resample_to_surface_f,
    'resample_to_surface',
    output=('surf', 'surf_scalars'),
    forward_unused=True,
)


plot_to_image_p = Primitive(
    plot_to_image_f,
    'plot_to_image',
    output=('screenshots',),
    forward_unused=True,
)


automap_unified_plotter_p = Primitive(
    automap_unified_plotter_f,
    'automap_unified_plotter',
    output=('p'),
    forward_unused=True,
)
