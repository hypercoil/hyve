# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Visualisation transforms
~~~~~~~~~~~~~~~~~~~~~~~~
Functions for transforming the input and output of visualisation functions.
See also ``flows.py`` for functions that transform the control flow of
visualisation functions.
"""
from typing import (
    Any,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import nibabel as nb
import numpy as np
import pandas as pd
from conveyant import FunctionWrapper as F
from conveyant import PartialApplication as Partial
from conveyant import direct_compositor
from conveyant import splice_on as splice_on_orig
from lytemaps.transforms import mni152_to_fsaverage, mni152_to_fslr
from pkg_resources import resource_filename as pkgrf

from .const import REQUIRED, Tensor
from .layout import CellLayout
from .plot import _null_sbprocessor, overlay_scalar_bars
from .prim import (
    add_edge_variable_p,
    add_network_overlay_p,
    add_node_variable_p,
    add_points_overlay_p,
    add_postprocessor_p,
    add_surface_overlay_p,
    auto_camera_aux_p,
    auto_camera_p,
    build_network_p,
    closest_ortho_camera_aux_p,
    closest_ortho_camera_p,
    node_coor_from_parcels_p,
    parcellate_colormap_p,
    parcellate_surf_scalars_p,
    planar_sweep_camera_aux_p,
    planar_sweep_camera_p,
    plot_final_view_f,
    plot_to_display_p,
    plot_to_html_buffer_f,
    plot_to_image_aux_p,
    plot_to_image_f,
    points_scalars_from_array_p,
    points_scalars_from_nifti_p,
    save_figure_p,
    save_grid_p,
    save_html_p,
    save_snapshots_p,
    scalar_focus_camera_aux_p,
    scalar_focus_camera_p,
    scatter_into_parcels_p,
    surf_from_archive_p,
    surf_from_gifti_p,
    surf_scalars_from_array_p,
    surf_scalars_from_cifti_p,
    surf_scalars_from_gifti_p,
    surf_scalars_from_nifti_p,
    transform_postprocessor_p,
    vertex_to_face_p,
)
from .surf import CortexTriSurface
from .util import (
    NetworkDataCollection,
    PointDataCollection,
    sanitise,
)


def splice_on(
    g: callable,
    occlusion: Sequence[str] = (),
    expansion: Optional[Mapping[str, Tuple[Type, Any]]] = None,
    allow_variadic: bool = True,
    strict_emulation: bool = True,
) -> callable:
    """
    Patch ``splice_on`` to allow for variadic functions by default.
    """
    return splice_on_orig(
        g,
        occlusion=occlusion,
        expansion=expansion,
        allow_variadic=allow_variadic,
        strict_emulation=strict_emulation,
    )


def surf_from_archive(
    allowed: Sequence[str] = ('templateflow', 'neuromaps')
) -> callable:
    """
    Load a surface from a cloud-based data archive.

    Parameters
    ----------
    allowed : sequence of str (default: ("templateflow", "neuromaps")
        The archives to search for the surface.

    Returns
    -------
    callable
        A transform function. Transform functions accept a plotting
        function and return a new plotting function with different input and
        output arguments.
        * The transformed plotter will no longer require a ``surf`` argument,
          but will require a ``template`` argument.
        * The ``template`` argument should be a string that identifies the
          template space to load the surface from. The ``template`` argument
          will be passed to the archive loader function.
        * An optional ``load_mask`` argument can be passed to the transformed
          plotter to indicate whether the surface mask should be loaded
        (defaults to ``True``).
        * An optional ``projections`` argument can be passed to the
          transformed plotter to indicate which projections to load
          (defaults to ``("veryinflated",)``).
    """
    archives = {
        'templateflow': CortexTriSurface.from_tflow,
        'neuromaps': CortexTriSurface.from_nmaps,
    }
    archives = {k: v for k, v in archives.items() if k in allowed}
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_f = Partial(surf_from_archive_p, archives=archives)

        @splice_on(f, occlusion=surf_from_archive_p.output)
        def f_transformed(
            *,
            template: str = 'fsLR',
            load_mask: bool = True,
            surf_projection: Optional[Sequence[str]] = ('veryinflated',),
            **params: Mapping,
        ):
            return compositor(f, transformer_f)(**params)(
                template=template,
                load_mask=load_mask,
                projections=surf_projection,
            )

        return f_transformed
    return transform


def surf_from_gifti(
    projection: str = 'very_inflated',
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_f = Partial(surf_from_gifti_p, projection=projection)

        @splice_on(f, occlusion=surf_from_gifti_p.output)
        def f_transformed(
            *,
            left_surf: Union[str, nb.gifti.gifti.GiftiImage],
            right_surf: Union[str, nb.gifti.gifti.GiftiImage],
            left_mask: Optional[Union[str, nb.gifti.gifti.GiftiImage]] = None,
            right_mask: Optional[Union[str, nb.gifti.gifti.GiftiImage]] = None,
            **params: Mapping,
        ):
            return compositor(f, transformer_f)(**params)(
                left_surf=left_surf,
                right_surf=right_surf,
                left_mask=left_mask,
                right_mask=right_mask,
            )

        return f_transformed
    return transform


def surf_scalars_from_cifti(
    scalars: str,
    is_masked: bool = True,
    apply_mask: bool = False,
    null_value: Optional[float] = 0.0,
    select: Optional[Sequence[int]] = None,
    exclude: Optional[Sequence[int]] = None,
    allow_multihemisphere: bool = True,
    coerce_to_scalar: bool = True,
    plot: bool = True,
) -> callable:
    """
    Load a scalar dataset from a CIFTI file onto a CortexTriSurface.

    Parameters
    ----------
    scalars : str
        The name that the scalar dataset loaded from the CIFTI file is given
        on the surface.
    is_masked : bool (default: True)
        Indicates whether the CIFTI file contains a dataset that is already
        masked.
    apply_mask : bool (default: False)
        Indicates whether the surface mask should be applied to the CIFTI
        dataset.
    null_value : float or None (default: 0.)
        The value to use for masked-out vertices.
    plot : bool (default: False)
        Indicates whether the scalar dataset should be plotted.

    Returns
    -------
    callable
        A transform function. Transform functions accept a plotting
        function and return a new plotting function with different input and
        output arguments.
        * The transformed plotter will now require a ``<scalars>_cifti``
          argument, where ``<scalars>`` is the name of the scalar dataset
          provided as an argument to this function. The value of this
          argument should be either a ``Cifti2Image`` object or a path to a
          CIFTI file. This is the CIFTI image whose data will be loaded onto
          the surface.
        * If ``plot`` is ``True``, the transformed function will automatically
          add the scalar dataset obtained from the CIfTI image to the sequence
          of scalars to plot.
    """
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        paramstr = f'{sanitise(scalars)}_cifti'
        transformer_f = Partial(
            surf_scalars_from_cifti_p,
            scalars=scalars,
            is_masked=is_masked,
            apply_mask=apply_mask,
            null_value=null_value,
            select=select,
            exclude=exclude,
            allow_multihemisphere=allow_multihemisphere,
            coerce_to_scalar=coerce_to_scalar,
            plot=plot,
        )

        @splice_on(
            f,
            occlusion=surf_scalars_from_cifti_p.output,
            expansion={paramstr: (Union[nb.Cifti2Image, str], REQUIRED)},
        )
        def f_transformed(
            *,
            surf: CortexTriSurface,
            **params: Mapping,
        ):
            try:
                cifti = params.pop(paramstr)
            except KeyError:
                raise TypeError(
                    'Transformed plot function missing one required '
                    f'keyword-only argument: {paramstr}'
                )
            surf_scalars = params.pop('surf_scalars', ())
            return compositor(f, transformer_f)(**params)(
                cifti=cifti,
                surf=surf,
                surf_scalars=surf_scalars,
            )

        return f_transformed
    return transform


def surf_scalars_from_gifti(
    scalars: str,
    is_masked: bool = True,
    apply_mask: bool = False,
    null_value: Optional[float] = 0.0,
    select: Optional[Sequence[int]] = None,
    exclude: Optional[Sequence[int]] = None,
    allow_multihemisphere: bool = True,
    coerce_to_scalar: bool = True,
    plot: bool = True,
) -> callable:
    """
    Load a scalar dataset from a GIfTI file onto a CortexTriSurface.

    Parameters
    ----------
    scalars : str
        The name that the scalar dataset loaded from the GIfTI file is given
        on the surface.
    is_masked : bool (default: True)
        Indicates whether the GIfTI file contains a dataset that is already
        masked.
    apply_mask : bool (default: False)
        Indicates whether the surface mask should be applied to the GIfTI
        dataset.
    null_value : float or None (default: 0.)
        The value to use for masked-out vertices.
    plot : bool (default: False)
        Indicates whether the scalar dataset should be plotted.

    Returns
    -------
    callable
        A transform function. Transform functions accept a plotting
        function and return a new plotting function with different input and
        output arguments.
        * The transformed plotter will now require a ``<scalars>_gifti_left``
          and/or ``<scalars>_gifti_right`` argument, where ``<scalars>`` is
          the name of the scalar dataset provided as an argument to this
          function. The value provided for each of these arguments should be
          either a ``GIfTIImage`` object or a path to a GIfTI file. These are
          the GIfTI images whose data will be loaded onto the surface.
        * If ``plot`` is ``True``, the transformed function will automatically
          add the scalar dataset obtained from the GIfTI image to the sequence
          of scalars to plot.
    """
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        paramstr_left = f'{sanitise(scalars)}_gifti_left'
        paramstr_right = f'{sanitise(scalars)}_gifti_right'
        transformer_f = Partial(
            surf_scalars_from_gifti_p,
            scalars=scalars,
            is_masked=is_masked,
            apply_mask=apply_mask,
            null_value=null_value,
            select=select,
            exclude=exclude,
            allow_multihemisphere=allow_multihemisphere,
            coerce_to_scalar=coerce_to_scalar,
            plot=plot,
        )

        @splice_on(
            f,
            occlusion=surf_scalars_from_gifti_p.output,
            expansion={
                paramstr_left: (Union[nb.GiftiImage, str], REQUIRED),
                paramstr_right: (Union[nb.GiftiImage, str], REQUIRED),
            },
        )
        def f_transformed(
            *,
            surf: CortexTriSurface,
            **params: Mapping,
        ):
            left_gifti = params.pop(paramstr_left, None)
            right_gifti = params.pop(paramstr_right, None)
            if left_gifti is None and right_gifti is None:
                raise TypeError(
                    'Transformed plot function missing a required '
                    f'keyword-only argument: either {paramstr_left} or '
                    f'{paramstr_right}'
                )
            surf_scalars = params.pop('surf_scalars', ())
            return compositor(f, transformer_f)(**params)(
                left_gifti=left_gifti,
                right_gifti=right_gifti,
                surf=surf,
                surf_scalars=surf_scalars,
            )

        return f_transformed
    return transform


def surf_scalars_from_array(
    scalars: str,
    left_slice: Optional[slice] = None,
    right_slice: Optional[slice] = None,
    default_slices: bool = True,
    is_masked: bool = True,
    apply_mask: bool = False,
    null_value: Optional[float] = 0.0,
    select: Optional[Sequence[int]] = None,
    exclude: Optional[Sequence[int]] = None,
    allow_multihemisphere: bool = True,
    coerce_to_scalar: bool = True,
    plot: bool = True,
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        paramstr = f'{sanitise(scalars)}_array'
        paramstr_left = f'{sanitise(scalars)}_array_left'
        paramstr_right = f'{sanitise(scalars)}_array_right'
        transformer_f = Partial(
            surf_scalars_from_array_p,
            scalars=scalars,
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
            plot=plot,
        )

        @splice_on(
            f,
            occlusion=surf_scalars_from_array_p.output,
            expansion={
                paramstr: (Tensor, None),
                paramstr_left: (Tensor, None),
                paramstr_right: (Tensor, None),
            },
        )
        def f_transformed(
            *,
            surf: CortexTriSurface,
            **params: Mapping,
        ):
            left_array = params.pop(paramstr_left, None)
            right_array = params.pop(paramstr_right, None)
            array = params.pop(paramstr, None)
            if left_array is None and right_array is None and array is None:
                raise TypeError(
                    'Transformed plot function missing one or more required '
                    f'keyword-only argument(s): either {paramstr} or '
                    f'{paramstr_left} and/or {paramstr_right}'
                )
            surf_scalars = params.pop('surf_scalars', ())
            return compositor(f, transformer_f)(**params)(
                surf=surf,
                surf_scalars=surf_scalars,
                array=array,
                left_array=left_array,
                right_array=right_array,
            )

        return f_transformed
    return transform


#TODO: replace null_value arg with the option to provide one of our hypermaths
#      expressions.
def points_scalars_from_nifti(
    scalars: str,
    null_value: Optional[float] = 0.0,
    point_size: Optional[float] = None,
    plot: bool = True,
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        paramstr = f'{sanitise(scalars)}_nifti'
        transformer_f = Partial(
            points_scalars_from_nifti_p,
            scalars=scalars,
            null_value=null_value,
            point_size=point_size,
            plot=plot,
        )

        @splice_on(
            f,
            occlusion=points_scalars_from_nifti_p.output,
            expansion={paramstr: (Union[nb.Nifti1Image, str], REQUIRED)},
        )
        def f_transformed(
            *,
            points_scalars: Sequence[str] = (),
            points: Optional[PointDataCollection] = None,
            **params: Mapping,
        ):
            try:
                nifti = params.pop(paramstr)
            except KeyError:
                raise TypeError(
                    'Transformed plot function missing one required '
                    f'keyword-only argument: {paramstr}'
                )
            return compositor(f, transformer_f)(**params)(
                nifti=nifti,
                points=points,
                points_scalars=points_scalars,
            )

        return f_transformed
    return transform


def points_scalars_from_array(
    scalars: str,
    point_size: float = 1.0,
    plot: bool = True,
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        paramstr_coor = f'{sanitise(scalars)}_coor'
        paramstr_values = f'{sanitise(scalars)}_values'
        transformer_f = Partial(
            points_scalars_from_array_p,
            scalars=scalars,
            point_size=point_size,
            plot=plot,
        )

        @splice_on(
            f,
            occlusion=points_scalars_from_array_p.output,
            expansion={
                paramstr_coor: (Tensor, REQUIRED),
                paramstr_values: (Tensor, REQUIRED),
            },
        )
        def f_transformed(
            *,
            points: Optional[PointDataCollection] = None,
            points_scalars: Sequence[str] = (),
            **params: Mapping,
        ):
            try:
                coor = params.pop(paramstr_coor)
            except KeyError:
                raise TypeError(
                    'Transformed plot function missing one required '
                    f'keyword-only argument: {paramstr_coor}'
                )
            try:
                values = params.pop(paramstr_values)
            except KeyError:
                raise TypeError(
                    'Transformed plot function missing one required '
                    f'keyword-only argument: {paramstr_values}'
                )
            return compositor(f, transformer_f)(**params)(
                coor=coor,
                values=values,
                points=points,
                points_scalars=points_scalars,
            )

        return f_transformed
    return transform


def surf_scalars_from_nifti(
    scalars: str,
    template: str = 'fsLR',
    method: Literal['nearest', 'linear'] = 'linear',
    null_value: Optional[float] = 0.0,
    threshold: Optional[float] = None,
    select: Optional[Sequence[int]] = None,
    exclude: Optional[Sequence[int]] = None,
    allow_multihemisphere: bool = True,
    coerce_to_scalar: bool = True,
    plot: bool = True,
) -> callable:
    """
    Resample a scalar dataset from volumetric MNI152 space to a surface.

    Parameters
    ----------
    scalars : str
        The name that the scalar dataset resampled from volumetric space is
        given on the surface.
    template : str (default: "fsLR")
        The name of the template to which the scalar dataset should be
        resampled. Currently, only "fsLR" and "fsaverage" are supported.
    null_value : float or None (default: 0.)
        The value to use for voxels that are outside the brain mask.
    select : Sequence[int] or None (default: None)
        If not None, the indices of the scalar maps to load from the
        volumetric dataset. If None, all scalar maps are loaded.
    exclude : Sequence[int] or None (default: None)
        If not None, the indices of the scalar maps to exclude from the
        volumetric dataset. If None, no scalar maps are excluded.
    plot : bool (default: False)
        Indicates whether the scalar dataset should be plotted.

    Returns
    -------
    callable
        A transform function. Transform functions accept a plotting
        function and return a new plotting function with different input and
        output arguments.
        * The transformed plotter will now require a ``<scalars>_nifti``
          argument, where ``<scalars>`` is the name of the scalar dataset
          provided as an argument to this function. The value of this
          argument should be an instance of a ``nibabel`` ``Nifti1Image``
          or a path to a NIfTI file.
        * If ``plot`` is ``True``, the transformed function will automatically
          add the resampled scalar dataset to the sequence of scalars to plot.
    """
    templates = {
        'fsLR': F(mni152_to_fslr),
        'fsaverage': F(mni152_to_fsaverage),
    }
    f_resample = templates[template]
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        paramstr = f'{sanitise(scalars)}_nifti'
        transformer_f = Partial(
            surf_scalars_from_nifti_p,
            scalars=scalars,
            f_resample=f_resample,
            method=method,
            null_value=null_value,
            threshold=threshold,
            select=select,
            exclude=exclude,
            allow_multihemisphere=allow_multihemisphere,
            coerce_to_scalar=coerce_to_scalar,
            plot=plot,
        )

        @splice_on(
            f,
            occlusion=surf_scalars_from_nifti_p.output,
            expansion={paramstr: (Union[nb.Nifti1Image, str], REQUIRED)},
        )
        def f_transformed(
            *,
            surf: CortexTriSurface,
            surf_scalars: Sequence[str] = (),
            **params: Mapping,
        ):
            try:
                nifti = params.pop(paramstr)
            except KeyError:
                raise TypeError(
                    'Transformed plot function missing one required '
                    f'keyword-only argument: {paramstr}'
                )
            return compositor(f, transformer_f)(**params)(
                nifti=nifti,
                surf=surf,
                surf_scalars=surf_scalars,
            )

        return f_transformed
    return transform


def parcellate_colormap(
    cmap_name: str,
    parcellation_name: str,
    target: Union[str, Sequence[str]] = ('surf_scalars', 'node'),
) -> callable:
    """
    Add a colormap to a surface, and then parcellate it to obtain colours for
    each parcel.

    Parameters
    ----------
    cmap_name : str
        The name of the colormap to add to the surface. Currently, only
        "network" and "modal" are supported. The "network" colormap colours
        each parcel based on its overlap with the 7 resting-state networks
        defined by Yeo et al. (2011). The "modal" colormap colours each
        parcel based on its affiliation with the 3 modal domains defined by
        Glasser et al. (2016).
    parcellation_name : str
        The name of the parcellation to use to parcellate the colormap. The
        parcellation must be present on the surface.

    Returns
    -------
    callable
        A transform function. Transform functions accept a plotting
        function and return a new plotting function with different input and
        output arguments.
        * The transformed function will no longer accept the arguments
          ``cmap``, ``clim``, ``node_cmap``, or ``node_cmap_range``.
    """
    cmaps = {
        'network': 'data/cmap/cmap_network.nii',
        'modal': 'data/cmap/cmap_modal.nii',
    }
    cmap = pkgrf(
        'hyve',
        cmaps[cmap_name],
    )
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_f = Partial(
            parcellate_colormap_p,
            cmap_name=cmap_name,
            parcellation_name=parcellation_name,
            cmap=cmap,
            target=target,
        )

        @splice_on(
            f,
            occlusion=(
                'surf_scalars_cmap',
                'surf_scalars_clim',
                'surf_scalars_below_color',
            ),
        )
        def f_transformed(
            *,
            surf: CortexTriSurface,
            **params: Mapping,
        ):
            return compositor(f, transformer_f)(**params)(surf=surf)

        return f_transformed
    return transform


def parcellate_surf_scalars(
    scalars: str,
    parcellation_name: str,
    plot: bool = True,
) -> callable:
    """
    Add a scalar dataset to a surface, and then parcellate it to obtain
    values for each parcel.

    Parameters
    ----------
    scalars : str
        The name of the scalar dataset to add to the surface.
    parcellation_name : str
        The name of the parcellation to use to parcellate the scalar dataset.
        The parcellation must be present on the surface.
    plot : bool (default: True)
        Indicates whether the parcellated scalar dataset should be plotted.

    Returns
    -------
    callable
        A transform function. Transform functions accept a plotting
        function and return a new plotting function with different input and
        output arguments.
        * If ``plot`` is ``True``, the transformed function will automatically
          add the parcellated scalar dataset to the sequence of scalars to
          plot.
    """
    sink = f'{scalars}Parcellated'
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_f = Partial(
            parcellate_surf_scalars_p,
            scalars=scalars,
            parcellation_name=parcellation_name,
            sink=sink,
            plot=plot,
        )

        @splice_on(f, occlusion=parcellate_surf_scalars_p.output)
        def f_transformed(
            *,
            surf: CortexTriSurface,
            surf_scalars: Sequence[str] = (),
            **params: Mapping,
        ):
            return compositor(f, transformer_f)(**params)(
                surf=surf,
                surf_scalars=surf_scalars,
            )

        return f_transformed
    return transform


def scatter_into_parcels(
    scalars: str,
    parcellation_name: str,
    plot: bool = True,
) -> callable:
    """
    Add a parcel-valued scalar dataset to a surface by scattering the
    parcel-wise values into the vertices of the surface.

    Parameters
    ----------
    scalars : str
        The name of the scalar dataset to add to the surface.
    parcellation_name : str
        The name of the parcellation to use to parcellate the scalar dataset.
        The parcellation must be present on the surface.
    plot : bool (default: True)
        Indicates whether the parcel-valued scalar dataset should be plotted.

    Returns
    -------
    callable
        A transform function. Transform functions accept a plotting
        function and return a new plotting function with different input and
        output arguments.
        * If ``plot`` is ``True``, the transformed function will automatically
          add the parcel-valued scalar dataset to the sequence of scalars to
          plot.
        * The transformed plotter requires the ``<scalars>_parcellated``
          argument, where ``<scalars>`` is the name of the scalar dataset
          provided as an argument to this function. The value of this
          argument should be a tensor of shape ``(N,)`` where ``N`` is the
          number of parcels in the parcellation.
    """
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        paramstr = f'{sanitise(scalars)}_parcellated'
        transformer_f = Partial(
            scatter_into_parcels_p,
            scalars=scalars,
            parcellation_name=parcellation_name,
            plot=plot,
        )

        @splice_on(
            f,
            occlusion=scatter_into_parcels_p.output,
            expansion={paramstr: (Tensor, REQUIRED)},
        )
        def f_transformed(
            *,
            surf: CortexTriSurface,
            surf_scalars: Sequence[str] = (),
            **params: Mapping,
        ):
            try:
                parcellated = params.pop(paramstr)
            except KeyError:
                raise TypeError(
                    'Transformed plot function missing one required '
                    f'keyword-only argument: {paramstr}'
                )
            return compositor(f, transformer_f)(**params)(
                surf=surf,
                parcellated=parcellated,
                surf_scalars=surf_scalars,
            )

        return f_transformed
    return transform


def vertex_to_face(
    scalars: str,
    interpolation: Literal['mode', 'mean'] = 'mode',
) -> callable:
    """
    Resample a scalar dataset defined on the vertices of a surface to a
    scalar dataset defined on the faces of the surface. The vertex-valued
    scalar dataset must be defined on the surface (i.e., in its ``point_data``
    dictionary).

    Parameters
    ----------
    scalars : str
        The name of the scalar dataset to resample.
    interpolation : {'mode', 'mean'} (default: 'mode')
        The interpolation method to use. If ``'mode'``, the value of the
        scalar dataset for each face is the mode of the values of the
        vertices that make up the face. If ``'mean'``, the value of the
        scalar dataset for each face is the mean of the values of the
        vertices that make up the face.

    Returns
    -------
    callable
        A transform function. Transform functions accept a plotting
        function and return a new plotting function with different input and
        output arguments.
    """
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_f = Partial(
            vertex_to_face_p,
            scalars=scalars,
            interpolation=interpolation,
        )

        @splice_on(f, occlusion=vertex_to_face_p.output)
        def f_transformed(
            *,
            surf: CortexTriSurface,
            surf_scalars: Sequence[str] = (),
            **params: Mapping,
        ):
            return compositor(f, transformer_f)(**params)(
                surf=surf,
                surf_scalars=surf_scalars,
            )

        return f_transformed
    return transform


def add_surface_overlay(
    layer_name: str,
    *chains: Sequence[callable],
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> direct_compositor:
        result = add_surface_overlay_p(layer_name=layer_name, chains=chains)
        transformer_f, signature = result['prim'], result['signature']

        @splice_on(
            f,
            expansion={
                k: (v.annotation, v.default)
                for k, v in signature.parameters.items()
            },
        )
        def f_transformed(**params: Mapping):
            return compositor(f, transformer_f)()(params=params)

        return f_transformed
    return transform


def add_points_overlay(
    layer_name: str,
    *chains: Sequence[callable],
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> direct_compositor:
        result = add_points_overlay_p(layer_name=layer_name, chains=chains)
        transformer_f, signature = result['prim'], result['signature']

        @splice_on(
            f,
            expansion={
                k: (v.annotation, v.default)
                for k, v in signature.parameters.items()
            },
        )
        def f_transformed(**params: Mapping):
            return compositor(f, transformer_f)()(params=params)

        return f_transformed
    return transform


def add_network_overlay(
    layer_name: str,
    *chains: Sequence[callable],
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> direct_compositor:
        result = add_network_overlay_p(layer_name=layer_name, chains=chains)
        transformer_f, signature = result['prim'], result['signature']

        @splice_on(
            f,
            expansion={
                k: (v.annotation, v.default)
                for k, v in signature.parameters.items()
            },
        )
        def f_transformed(**params: Mapping):
            return compositor(f, transformer_f)()(params=params)

        return f_transformed
    return transform


def build_network(name: str = 'network') -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> direct_compositor:
        transformer_f = Partial(
            build_network_p,
            name=name,
        )

        @splice_on(f, occlusion=build_network_p.output)
        def f_transformed(
            *,
            node_coor: Tensor,
            node_values: Optional[pd.DataFrame] = None,
            edge_values: Optional[pd.DataFrame] = None,
            networks: Optional[NetworkDataCollection] = None,
            lh_mask: Optional[Tensor] = None,
            **params: Mapping,
        ):
            return compositor(f, transformer_f)(**params)(
                node_coor=node_coor,
                node_values=node_values,
                edge_values=edge_values,
                networks=networks,
                lh_mask=lh_mask,
            )

        return f_transformed
    return transform


def node_coor_from_parcels(parcellation: str) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> direct_compositor:
        transformer_f = Partial(
            node_coor_from_parcels_p,
            parcellation=parcellation,
        )

        @splice_on(f, occlusion=node_coor_from_parcels_p.output)
        def f_transformed(
            *,
            surf: CortexTriSurface,
            surf_projection: str,
            **params: Mapping,
        ):
            return compositor(f, transformer_f)(**params)(
                surf=surf,
                surf_projection=surf_projection,
            )

        return f_transformed
    return transform


def add_node_variable(
    name: str = 'node',
    threshold: Union[float, int] = 0.0,
    percent_threshold: bool = False,
    topk_threshold: bool = False,
    absolute: bool = True,
    node_selection: Optional[np.ndarray] = None,
    incident_edge_selection: Optional[np.ndarray] = None,
    removed_val: Optional[float] = None,
    surviving_val: Optional[float] = 1.0,
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        paramstr = f'{name}_nodal'
        transformer_f = Partial(
            add_node_variable_p,
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

        @splice_on(
            f,
            occlusion=add_node_variable_p.output,
            expansion={paramstr: (Union[pd.DataFrame, str], REQUIRED)},
        )
        def f_transformed(**params: Mapping):
            try:
                val = params.pop(paramstr)
            except KeyError:
                raise TypeError(
                    'Transformed plot function missing one required '
                    f'keyword-only argument: {name}_nodal'
                )
            return compositor(f, transformer_f)(**params)(val=val)

        return f_transformed
    return transform


def add_edge_variable(
    name: str,
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
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        paramstr = f'{name}_adjacency'
        transformer_f = Partial(
            add_edge_variable_p,
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

        @splice_on(
            f,
            occlusion=add_node_variable_p.output,
            expansion={paramstr: (Union[pd.DataFrame, str], REQUIRED)},
        )
        def f_transformed(**params: Mapping):
            try:
                adj = params.pop(paramstr)
            except KeyError:
                raise TypeError(
                    'Transformed plot function missing one required '
                    f'keyword-only argument: {name}_adjacency'
                )
            return compositor(f, transformer_f)(**params)(adj=adj)

        return f_transformed
    return transform


def scalar_focus_camera(
    kind: Literal["centroid", "peak"] = "centroid",
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_f = Partial(
            transform_postprocessor_p,
            name='snapshots',
            transformer=Partial(
                scalar_focus_camera_p,
                kind=kind,
                __allowed__=(
                    'surf',
                    'hemispheres',
                    'surf_scalars',
                    'surf_projection',
                    'close_plotter',
                ),
            ),
            auxwriter=Partial(
                scalar_focus_camera_aux_p,
                kind=kind,
                __allowed__=(
                    'hemisphere',
                ),
            ),
        )

        @splice_on(f, occlusion=transform_postprocessor_p.output)
        def f_transformed(
            postprocessors: Optional[Sequence[callable]] = None,
            **params: Mapping,
        ):
            return compositor(f, transformer_f)(**params)(
                postprocessors=postprocessors,
            )

        return f_transformed
    return transform


def closest_ortho_camera(
    n_ortho: int = 3,
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_f = Partial(
            transform_postprocessor_p,
            name='snapshots',
            transformer=Partial(
                closest_ortho_camera_p,
                n_ortho=n_ortho,
                __allowed__=(
                    'surf',
                    'hemispheres',
                    'surf_scalars',
                    'surf_projection',
                    'close_plotter',
                ),
            ),
            auxwriter=Partial(
                closest_ortho_camera_aux_p,
                n_ortho=n_ortho,
                __allowed__=(
                    'hemisphere',
                ),
            ),
        )

        @splice_on(f, occlusion=transform_postprocessor_p.output)
        def f_transformed(
            postprocessors: Optional[Sequence[callable]] = None,
            **params: Mapping,
        ):
            return compositor(f, transformer_f)(**params)(
                postprocessors=postprocessors,
            )

        return f_transformed
    return transform


def planar_sweep_camera(
    initial: Sequence,
    normal: Optional[Sequence[float]] = None,
    n_steps: int = 10,
    require_planar: bool = True,
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_f = Partial(
            transform_postprocessor_p,
            name='snapshots',
            transformer=Partial(
                planar_sweep_camera_p,
                initial=initial,
                normal=normal,
                n_steps=n_steps,
                require_planar=require_planar,
                __allowed__=(
                    'surf',
                    'hemispheres',
                    'close_plotter',
                ),
            ),
            auxwriter=Partial(
                planar_sweep_camera_aux_p,
                n_steps=n_steps,
                __allowed__=(
                    'hemisphere',
                ),
            ),
        )

        @splice_on(f, occlusion=transform_postprocessor_p.output)
        def f_transformed(
            postprocessors: Optional[Sequence[callable]] = None,
            **params: Mapping,
        ):
            return compositor(f, transformer_f)(**params)(
                postprocessors=postprocessors,
            )

        return f_transformed
    return transform


def auto_camera(
    n_ortho: int = 0,
    focus: Optional[Literal["centroid", "peak"]] = None,
    n_angles: int = 0,
    initial_angle: Tuple[float, float, float] = (1, 0, 0),
    normal_vector: Optional[Tuple[float, float, float]] = None,
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_f = Partial(
            transform_postprocessor_p,
            name='snapshots',
            transformer=Partial(
                auto_camera_p,
                n_ortho=n_ortho,
                focus=focus,
                n_angles=n_angles,
                initial_angle=initial_angle,
                normal_vector=normal_vector,
                __allowed__=(
                    'surf',
                    'hemispheres',
                    'surf_scalars',
                    'surf_projection',
                    'close_plotter',
                ),
            ),
            auxwriter=Partial(
                auto_camera_aux_p,
                n_ortho=n_ortho,
                focus=focus,
                n_angles=n_angles,
                __allowed__=(
                    'hemisphere',
                ),
            ),
        )

        @splice_on(f, occlusion=transform_postprocessor_p.output)
        def f_transformed(
            postprocessors: Optional[Sequence[callable]] = None,
            **params: Mapping,
        ):
            return compositor(f, transformer_f)(**params)(
                postprocessors=postprocessors,
            )

        return f_transformed
    return transform


def plot_to_image() -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_f = add_postprocessor_p
        _postprocessor = F(plot_to_image_f)
        _auxwriter = plot_to_image_aux_p

        @splice_on(f, occlusion=('window_size',))
        def f_transformed(
            *,
            views: Union[Sequence, Literal['__default__']] = '__default__',
            window_size: Tuple[int, int] = (1300, 1000),
            plot_scalar_bar: bool = False,
            **params,
        ):
            postprocessor = Partial(
                _postprocessor,
                views=views,
                window_size=window_size,
                plot_scalar_bar=plot_scalar_bar,
                __allowed__=('hemispheres', 'close_plotter'),
            )
            # The inconsistent naming of the `hemisphere` parameter is
            # intentional, but not ideal. The `hemispheres` variable is
            # defined in the local scope of the ``unified_plotter`` function
            # but remains undefined in the metadata function, so we currently
            # have to use the `hemisphere` parameter in the metadata function
            # as a workaround.
            auxwriter = Partial(
                _auxwriter,
                views=views,
                __allowed__=('hemisphere',),
            )
            postprocessors = params.get('postprocessors', None)
            return compositor(f, transformer_f)(
                window_size=window_size, **params
            )(
                name='snapshots',
                postprocessor=postprocessor,
                postprocessors=postprocessors,
                auxwriter=auxwriter,
            )

        return f_transformed
    return transform


def plot_final_image(
    n_scenes: int = 1,
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_f = add_postprocessor_p
        _postprocessor = F(plot_final_view_f)
        _auxwriter = plot_to_image_aux_p
        if n_scenes > 1:
            raise NotImplementedError(
                'The `plot_final_image` postprocessor does not currently '
                'support multiple scenes.'
            )

        @splice_on(f, occlusion=('off_screen', 'window_size'))
        def f_transformed(
            *,
            window_size: Tuple[int, int] = (1920, 1080),
            plot_scalar_bar: bool = False,
            **params,
        ):
            postprocessor = Partial(
                _postprocessor,
                n_scenes=n_scenes,
                window_size=window_size,
                plot_scalar_bar=plot_scalar_bar,
                __allowed__=('close_plotter',),
            )
            auxwriter = Partial(
                _auxwriter,
                views='__final__',
                n_scenes=n_scenes,
                __allowed__=('hemisphere',),
            )
            postprocessors = params.get('postprocessors', None)
            _ = params.pop('off_screen', False)
            return compositor(f, transformer_f)(
                off_screen=False,
                window_size=window_size,
                **params,
            )(
                name='snapshots',
                postprocessor=postprocessor,
                postprocessors=postprocessors,
                auxwriter=auxwriter,
            )

        return f_transformed
    return transform


def plot_to_html(
    fname_spec: Optional[str] = None,
    suffix: Optional[str] = 'scene',
    extension: str = 'html',
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_fi = add_postprocessor_p
        transformer_fo = Partial(
            save_html_p,
            fname_spec=fname_spec,
            suffix=suffix,
            extension=extension,
        )
        _postprocessor = F(plot_to_html_buffer_f)

        @splice_on(f, occlusion=('window_size',))
        def f_transformed(
            *,
            output_dir: str,
            window_size: Tuple[int, int] = (1920, 1080),
            **params,
        ):
            postprocessor = Partial(
                _postprocessor,
                window_size=window_size,
                __allowed__=('close_plotter',),
            )
            postprocessors = params.get('postprocessors', None)
            _f_transformed = compositor(
                transformer_fo,
                compositor(f, transformer_fi)(
                    window_size=window_size, **params
                ),
            )
            return _f_transformed(output_dir=output_dir)(
                name='html_buffer',
                postprocessor=postprocessor,
                postprocessors=postprocessors,
            )

        return f_transformed
    return transform


def plot_to_display(
    window_size: Tuple[int, int] = (1300, 1000),
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_f = Partial(
            plot_to_display_p,
            window_size=window_size,
        )

        @splice_on(f, occlusion=('off_screen',))
        def f_transformed(**params):
            _ = params.pop('off_screen', False)
            return compositor(transformer_f, f)()(
                off_screen=False,
                window_size=window_size,
                **params,
            )

        return f_transformed
    return transform


def save_snapshots(
    fname_spec: Optional[str] = None,
    suffix: Optional[str] = 'scene',
    extension: str = 'png',
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_f = Partial(
            save_snapshots_p,
            fname_spec=fname_spec,
            suffix=suffix,
            extension=extension,
        )

        @splice_on(f)
        def f_transformed(output_dir: str, **params):
            return compositor(transformer_f, f)(
                output_dir=output_dir,
            )(**params)

        return f_transformed
    return transform


def save_figure(
    layout: CellLayout,
    canvas_size: Tuple[int, int] = (2048, 2048),
    sort_by: Optional[Sequence[str]] = None,
    padding: int = 0,
    canvas_color: Any = (255, 255, 255, 255),
    fname_spec: Optional[str] = None,
    scalar_bar_action: Literal['overlay', 'collect'] = 'overlay',
    suffix: Optional[str] = 'scene',
    extension: str = 'svg',
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_f = Partial(
            save_figure_p,
            layout=layout,
            canvas_size=canvas_size,
            sort_by=sort_by,
            padding=padding,
            canvas_color=canvas_color,
            fname_spec=fname_spec,
            suffix=suffix,
            extension=extension,
        )

        @splice_on(f, occlusion=('sbprocessor',))
        def f_transformed(output_dir: str, **params):
            if scalar_bar_action == 'collect':
                sbprocessor = _null_sbprocessor
            elif scalar_bar_action == 'overlay':
                sbprocessor = overlay_scalar_bars
            return compositor(transformer_f, f)(
                output_dir=output_dir,
            )(sbprocessor=sbprocessor, **params)

        return f_transformed
    return transform


def save_grid(
    n_rows: int,
    n_cols: int,
    order: Literal['row', 'column'] = 'row',
    sort_by: Optional[Sequence[str]] = None,
    annotations: Optional[Mapping[int, Mapping]] = None,
    canvas_size: Tuple[int, int] = (2048, 2048),
    padding: int = 0,
    canvas_color: Any = (255, 255, 255, 255),
    fname_spec: Optional[str] = None,
    scalar_bar_action: Literal['overlay', 'collect'] = 'overlay',
    suffix: Optional[str] = 'scene',
    extension: str = 'svg',
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_f = Partial(
            save_grid_p,
            n_rows=n_rows,
            n_cols=n_cols,
            order=order,
            sort_by=sort_by,
            annotations=annotations,
            canvas_size=canvas_size,
            padding=padding,
            canvas_color=canvas_color,
            fname_spec=fname_spec,
            suffix=suffix,
            extension=extension,
        )

        @splice_on(f, occlusion=('sbprocessor',))
        def f_transformed(output_dir: str, **params):
            if scalar_bar_action == 'collect':
                sbprocessor = _null_sbprocessor
            elif scalar_bar_action == 'overlay':
                sbprocessor = overlay_scalar_bars
            return compositor(transformer_f, f)(
                output_dir=output_dir,
            )(sbprocessor=sbprocessor, **params)

        return f_transformed
    return transform
