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
from itertools import chain
from math import ceil
from pkg_resources import resource_filename as pkgrf
from typing import Literal, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import nibabel as nb
import pandas as pd
from lytemaps.transforms import mni152_to_fsaverage, mni152_to_fslr
import matplotlib.pyplot as plt
import pyvista as pv

from conveyant import (
    direct_compositor,
    SanitisedPartialApplication as Partial,
    SanitisedFunctionWrapper as F,
)
from .const import Tensor
from .prim import (
    surf_from_archive_p,
    scalars_from_cifti_p,
    scalars_from_gifti_p,
    scalars_from_nifti_p,
    resample_to_surface_p,
    parcellate_colormap_p,
    parcellate_scalars_p,
    scatter_into_parcels_p,
    vertex_to_face_p,
    add_node_variable_p,
    add_edge_variable_p,
    add_postprocessor_p,
    transform_postprocessor_p,
    plot_to_image_f,
    plot_to_image_aux_p,
    scalar_focus_camera_p,
    scalar_focus_camera_aux_p,
    plot_to_html_buffer_f,
    save_screenshots_p,
    save_html_p,
)
from .surf import CortexTriSurface
from .util import (
    auto_focus,
    filter_adjacency_data,
    filter_node_data,
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


def scalars_from_cifti(
    scalars: str,
    is_masked: bool = True,
    apply_mask: bool = False,
    null_value: Optional[float] = 0.0,
    plot: bool = False,
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
        transformer_f = Partial(
            scalars_from_cifti_p,
            scalars=scalars,
            is_masked=is_masked,
            apply_mask=apply_mask,
            null_value=null_value,
            plot=plot,
        )

        def f_transformed(
            *,
            surf: CortexTriSurface,
            **params: Mapping,
        ):
            try:
                cifti = params.pop(f'{scalars}_cifti')
            except KeyError:
                raise TypeError(
                    'Transformed plot function missing one required '
                    f'keyword-only argument: {scalars}_cifti'
                )
            surf_scalars = params.pop('surf_scalars', ())
            return compositor(f, transformer_f)(**params)(
                cifti=cifti,
                surf=surf,
                surf_scalars=surf_scalars,
            )

        return f_transformed
    return transform


def scalars_from_gifti(
    scalars: str,
    is_masked: bool = True,
    apply_mask: bool = False,
    null_value: Optional[float] = 0.0,
    select: Optional[Sequence[int]] = None,
    exclude: Optional[Sequence[int]] = None,
    plot: bool = False,
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
        transformer_f = Partial(
            scalars_from_gifti_p,
            scalars=scalars,
            is_masked=is_masked,
            apply_mask=apply_mask,
            null_value=null_value,
            select=select,
            exclude=exclude,
            plot=plot,
        )

        def f_transformed(
            *,
            surf: CortexTriSurface,
            **params: Mapping,
        ):
            left_gifti = params.pop(f'{scalars}_gifti_left', None)
            right_gifti = params.pop(f'{scalars}_gifti_right', None)
            if left_gifti is None and right_gifti is None:
                raise TypeError(
                    'Transformed plot function missing a required '
                    f'keyword-only argument: either {scalars}_gifti_left or '
                    f'{scalars}_gifti_right'
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


#TODO: replace threshold arg with the option to provide one of our hypermaths
#      expressions.
def scalars_from_nifti(
    scalars: str,
    threshold: float = 0.0,
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_f = Partial(
            scalars_from_nifti_p,
            threshold=threshold,
        )

        def f_transformed(**params: Mapping):
            try:
                nifti = params.pop(f'{scalars}_nifti')
            except KeyError:
                raise TypeError(
                    'Transformed plot function missing one required '
                    f'keyword-only argument: {scalars}_nifti'
                )
            return compositor(f, transformer_f)(**params)(nifti=nifti)

        return f_transformed
    return transform


def resample_to_surface(
    scalars: str,
    template: str = 'fsLR',
    null_value: Optional[float] = 0.0,
    select: Optional[Sequence[int]] = None,
    exclude: Optional[Sequence[int]] = None,
    plot: bool = False,
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
        'fsLR': mni152_to_fslr,
        'fsaverage': mni152_to_fsaverage,
    }
    f_resample = templates[template]
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_f = Partial(
            resample_to_surface_p,
            scalars=scalars,
            f_resample=f_resample,
            null_value=null_value,
            select=select,
            exclude=exclude,
            plot=plot,
        )

        def f_transformed(
            *,
            surf: CortexTriSurface,
            **params: Mapping,
        ):
            try:
                nifti = params.pop(f'{scalars}_nifti')
            except KeyError:
                raise TypeError(
                    'Transformed plot function missing one required '
                    f'keyword-only argument: {scalars}_nifti'
                )
            surf_scalars = params.pop('surf_scalars', ())
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

        def f_transformed(
            *,
            surf: CortexTriSurface,
            **params: Mapping,
        ):
            return compositor(f, transformer_f)(**params)(surf=surf)

        return f_transformed
    return transform


def parcellate_scalars(
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
          add the parcellated scalar dataset to the sequence of scalars to plot.
    """
    sink = f'{scalars}Parcellated'
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_f = Partial(
            parcellate_scalars_p,
            scalars=scalars,
            parcellation_name=parcellation_name,
            sink=sink,
            plot=plot,
        )

        def f_transformed(
            *,
            surf: CortexTriSurface,
            **params: Mapping,
        ):
            surf_scalars = params.pop('surf_scalars', ())
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
        transformer_f = Partial(
            scatter_into_parcels_p,
            scalars=scalars,
            parcellation_name=parcellation_name,
            plot=plot,
        )

        def f_transformed(
            *,
            surf: CortexTriSurface,
            **params: Mapping,
        ):
            try:
                parcellated = params.pop(f'{scalars}_parcellated')
            except KeyError:
                raise TypeError(
                    'Transformed plot function missing one required '
                    f'keyword-only argument: {scalars}_parcellated'
                )
            surf_scalars = params.pop('surf_scalars', ())
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


def add_node_variable(
    name: str = "node",
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

        def f_transformed(**params: Mapping):
            try:
                val = params[f"{name}_nodal"]
            except KeyError:
                raise TypeError(
                    'Transformed plot function missing one required '
                    f'keyword-only argument: {name}_nodal'
                )
            params = {
                k: v for k, v in params.items()
                if k != f"{name}_nodal"
            }
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

        def f_transformed(**params: Mapping):
            try:
                adj = params[f"{name}_adjacency"]
            except KeyError:
                raise TypeError(
                    'Transformed plot function missing one required '
                    f'keyword-only argument: {name}_adjacency'
                )
            params = {
                k: v for k, v in params.items()
                if k != f"{name}_adjacency"
            }
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
            name='screenshots',
            transformer=Partial(
                scalar_focus_camera_p,
                kind=kind,
                __allowed__=(
                    'surf',
                    'hemispheres',
                    'surf_scalars',
                    'surf_projection',
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
                __allowed__=('hemispheres',),
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
            return compositor(f, transformer_f)(**params)(
                name='screenshots',
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

        def f_transformed(
            *,
            output_dir: str,
            window_size: Tuple[int, int] = (1920, 1080),
            **params,
        ):
            postprocessor = Partial(
                _postprocessor,
                window_size=window_size,
                __allowed__=(),
            )
            postprocessors = params.get('postprocessors', None)
            _f_transformed = compositor(
                transformer_fo,
                compositor(f, transformer_fi)(**params),
            )
            return _f_transformed(output_dir=output_dir)(
                name='html_buffer',
                postprocessor=postprocessor,
                postprocessors=postprocessors,
            )

        return f_transformed
    return transform


def save_screenshots(
    fname_spec: Optional[str] = None,
    suffix: Optional[str] = 'scene',
    extension: str = 'png',
) -> callable:
    def transform(
        f: callable,
        compositor: callable = direct_compositor,
    ) -> callable:
        transformer_f = Partial(
            save_screenshots_p,
            fname_spec=fname_spec,
            suffix=suffix,
            extension=extension,
        )

        def f_transformed(output_dir: str, **params):
            return compositor(transformer_f, f)(
                output_dir=output_dir,
            )(**params)

        return f_transformed
    return transform
