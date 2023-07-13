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
)
from .const import Tensor
from .prim import (
    surf_from_archive_p,
    resample_to_surface_p,
    plot_to_image_p,
)
from .surf import CortexTriSurface, make_cmap
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
        "templateflow": CortexTriSurface.from_tflow,
        "neuromaps": CortexTriSurface.from_nmaps
    }
    archives = {k: v for k, v in archives.items() if k in allowed}
    def transform(
        f: callable,
        compositor: callable = direct_compositor
    ) -> callable:
        transformer_f = Partial(surf_from_archive_p, archives=archives)

        def f_transformed(
            *,
            template: str = 'fsLR',
            load_mask: bool = True,
            projections: Optional[Sequence[str]] = ('veryinflated',),
            **params: Mapping,
        ):
            return compositor(f, transformer_f)(**params)(
                template=template,
                load_mask=load_mask,
                projections=projections,
            )

        return f_transformed
    return transform


def resample_to_surface(
    scalars: str,
    template: str = 'fsLR',
    null_value: Optional[float] = 0.,
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
        compositor: callable = direct_compositor
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
                nii = params.pop(f"{scalars}_nifti")
            except KeyError:
                raise TypeError(
                    'Transformed plot function missing one required '
                    f'keyword-only argument: {scalars}_nifti'
                )
            scalars_to_plot = params.get('scalars', []) if plot else None
            return compositor(f, transformer_f)(**params)(
                nii=nii,
                surf=surf,
                scalars_to_plot=scalars_to_plot
            )

        return f_transformed
    return transform


def plot_to_image():
    def transform(
        f: callable,
        compositor: callable = direct_compositor
    ) -> callable:
        transformer_f = plot_to_image_p

        def f_transformed(
            *,
            basename: str = None,
            views: Sequence = (
                'medial',
                'lateral',
                'dorsal',
                'ventral',
                'anterior',
                'posterior',
            ),
            window_size: Tuple[int, int] = (1300, 1000),
            hemisphere: Optional[Literal['left', 'right', 'both']] = None,
            **params,
        ):
            return compositor(transformer_f, f)(
                basename=basename,
                views=views,
                window_size=window_size,
                hemisphere=hemisphere,
            )(hemisphere=hemisphere, **params)

        return f_transformed
    return transform
